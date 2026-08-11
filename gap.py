"""
欠落スロットの検出と質問生成。

IR とフレームを突き合わせ、必須スロットのうち空のものを列挙する。
空いていれば質問文を作る。文言は data/questions.toml に置き、
コードには書かない。

【主語を補完して質問しない条件】
    GA が空 かつ 述語に POLITE がある かつ 述語が意志動詞

「行きません」から主語が発話者だと分かるのは、この 3 つがそろうからである。
丁寧語は聞き手に向けた発話であることを示し、意志動詞は主体が
自分の判断で行う行為であることを示す。主語が明示されていないなら、
それは発話者自身と考えるのが自然になる。

どれか 1 つでも欠けると成立しない。
    「雨が降ります」  → 意志動詞でない。主語は雨で、省略ではない
    「行く」          → 丁寧語でない。独り言や引用の可能性がある

【疑問詞は埋まっていないものとして扱う】
「どこに行きますか」の NI には「どこ」が入るが、これは値ではなく
値を尋ねる印である。埋まったとみなすと、質問に質問で答えられなくなる。
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .frames import FrameDict
from .ir import IR, POLITE, Clause, Token

# 疑問詞。スロットに入っていても「空」として扱う。
INTERROGATIVES: frozenset[str] = frozenset(
    {"どこ", "誰", "だれ", "何", "なに", "いつ", "どちら", "どれ", "いくつ"}
)

# 主語を補ったときに入れる印。実在の Token ではないので surface で示す。
SPEAKER = "話者"


class QuestionsError(ValueError):
    """質問テンプレートが読めないときに送出する。"""


@dataclass
class Gap:
    """埋まっていない必須スロット 1 件。"""

    clause_index: int
    slot: str
    lemma: str
    question: str

    def __str__(self) -> str:
        return f"{self.lemma}[{self.slot}] {self.question}"


@dataclass
class Filled:
    """補完によって埋めたスロット。質問はしない。"""

    clause_index: int
    slot: str
    value: str
    reason: str


@dataclass
class GapReport:
    """1 つの IR に対する検出結果。"""

    gaps: list[Gap]
    filled: list[Filled]

    @property
    def has_gap(self) -> bool:
        return bool(self.gaps)

    def questions(self) -> list[str]:
        return [gap.question for gap in self.gaps]


class QuestionTemplates:
    """質問文のテンプレート。data/questions.toml から読む。"""

    def __init__(self, default: dict[str, str], by_lemma: dict[str, dict[str, str]]):
        self.default = default
        self.by_lemma = by_lemma

    @classmethod
    def load(cls, path: str | Path) -> "QuestionTemplates":
        target = Path(path)
        if not target.exists():
            raise QuestionsError(f"ファイルが無い: {target}")
        with target.open("rb") as handle:
            data = tomllib.load(handle)
        return cls(
            default={k: str(v) for k, v in (data.get("default") or {}).items()},
            by_lemma={
                lemma: {k: str(v) for k, v in table.items()}
                for lemma, table in (data.get("lemma") or {}).items()
            },
        )

    def question_for(self, lemma: str, slot: str) -> str:
        override = self.by_lemma.get(lemma, {})
        if slot in override:
            return override[slot]
        return self.default.get(slot, f"{slot}は？")


def is_interrogative(token: Token | None) -> bool:
    if token is None:
        return False
    return token.lemma in INTERROGATIVES or token.surface in INTERROGATIVES


def _slot_is_empty(clause: Clause, slot: str) -> bool:
    token = clause.slots.get(slot)
    if token is None:
        return True
    # 疑問詞は値ではなく「値を尋ねる印」なので埋まっていない扱い
    return is_interrogative(token)


def detect(
    ir: IR, frames: FrameDict, templates: QuestionTemplates
) -> GapReport:
    """IR とフレームを照合し、空の必須スロットと補完結果を返す。"""
    gaps: list[Gap] = []
    filled: list[Filled] = []

    for index, clause in enumerate(ir.clauses):
        predicate = clause.predicate
        if predicate is None:
            continue
        frame = frames.get(predicate.lemma)
        if frame is None:
            # フレームを知らない述語は判定しない。
            # 知らないことを「欠落あり」と報告しないため。
            continue

        # 主語の補完は required かどうかに関わらず先に判定する。
        # 「行く」の GA は optional だが、「行きません」の主語が発話者だと
        # 分かること自体は必須スロットの有無と無関係である。
        # required だけを見ていると、この補完が起きない。
        completed_subject = False
        if _slot_is_empty(clause, "GA") and _speaker_is_subject(predicate, frame):
            filled.append(
                Filled(
                    clause_index=index,
                    slot="GA",
                    value=SPEAKER,
                    reason="丁寧語・主語なし・意志動詞の組み合わせ",
                )
            )
            completed_subject = True

        for slot in frame.required:
            if not _slot_is_empty(clause, slot):
                continue
            if slot == "GA" and completed_subject:
                continue

            gaps.append(
                Gap(
                    clause_index=index,
                    slot=slot,
                    lemma=predicate.lemma,
                    question=templates.question_for(predicate.lemma, slot),
                )
            )

    return GapReport(gaps=gaps, filled=filled)


def _speaker_is_subject(predicate: Token, frame) -> bool:
    """主語を発話者として補ってよいか。

    経路が 2 つある。

    1. 感情・感覚の述語（experiencer）
       「疲れました」「眠いです」「痛いです」の主語は話者である。
       疲れるのは意志ではないので volitional では捉えられない。
       丁寧語も要らない。「疲れた」も主語は話者である。
       感情や感覚は本人にしか分からないので、明示が無ければ本人の話になる。

    2. 丁寧語 + 意志動詞
       「行きません」の主語が話者だと分かるのは、聞き手に向けた発話で
       あること（丁寧語）と、自分の判断で行う行為であること（意志動詞）が
       そろうからである。詳しくはモジュール冒頭の説明を参照。
    """
    if frame.experiencer:
        return True
    return predicate.has(POLITE) and frame.volitional
