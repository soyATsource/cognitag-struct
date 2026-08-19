"""
文型に単語を埋めて返答を作る。

【固定文を選ぶのをやめる】
「{target}を{predicate}、ということだな」のような穴あきの文を並べても、
書いた数しか言い方は増えない。ここでは文を持たず、
「単語」「助詞」「単語」の並び（文型）だけを持つ。

    文型      [WO, を, PRED]
    入力      鍵をなくした
    出力      鍵をなくしたんだな / 鍵をなくしたのか / 鍵をなくそう …

単語は相手の発話から取り、述語は活用させる。したがって
返答の数は 文型 × 埋まっている格 × 活用の組み合わせ になる。
文型を 1 本足すと、条件を満たす全入力に効く。

【作れないものは作らない】
必要な格が空なら、その文型は使わない。活用型を知らない述語も使わない。
無理に埋めると、存在しない語や意味の通らない文ができる。
候補が 1 つも作れなければ None を返し、呼び出し側が別の道を選ぶ。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .inflect import Conjugator, Spec
from .select import select
from .ir import Clause

# elements に書ける役割名。これ以外は文字列としてそのまま置く。
SLOT_NAMES: frozenset[str] = frozenset(
    {"GA", "WO", "NI", "HE", "KARA", "MADE", "TOPIC"}
)
PREDICATE = "PRED"

# 空いている格に、辞書から引いた語を置く印。
#
# 「行きます」の NI は空いている。「行く」の NI は場所を期待すると
# frames.jsonl が言っているので、場所カテゴリから 1 語を引いて置く。
# 相手が言っていない語なので、これを置くと返答が相手の言葉の
# 言い換えではなくなる。ここが「関連語取得」にあたる。
CANDIDATE = "CAND"
# その格の助詞。CAND と組で使う。「駅に」「本を」。
CANDIDATE_PARTICLE = "CAND_P"

# 格 -> 助詞。CAND_P を置くときに使う。
# 候補語として残す上位の件数。頻度の高い順に、この数だけ見る。
# 少なすぎると同じ語ばかり出る。多すぎると専門語が混ざる。
COMMON_KEEP = 12

PARTICLES: dict[str, str] = {
    "GA": "が", "WO": "を", "NI": "に",
    "HE": "へ", "KARA": "から", "MADE": "まで",
}


class PatternError(ValueError):
    """文型の定義が不正なときに送出する。"""


@dataclass
class Pattern:
    """返答の文型 1 つ。"""

    id: str
    elements: list[str]
    need: list[str] = field(default_factory=list)
    spec: dict = field(default_factory=dict)
    note: str = ""
    # CAND をどの格として置くか。「NI」なら場所、「WO」なら対象。
    # フレームの content_constraints をこの格で引く。
    suggest_slot: str = ""


@dataclass
class Candidate:
    """組み立てた返答 1 件。どの文型から作ったかを残す。"""

    text: str
    pattern_id: str


class PatternSet:
    """文型の一覧。data/patterns.jsonl から読む。"""

    def __init__(self, patterns: list[Pattern]) -> None:
        self.patterns = patterns

    @classmethod
    def load(cls, path: str | Path) -> "PatternSet":
        target = Path(path)
        if not target.exists():
            raise PatternError(f"ファイルが無い: {target}")

        patterns: list[Pattern] = []
        seen: set[str] = set()
        for line_no, line in enumerate(
            target.read_text(encoding="utf-8").splitlines(), 1
        ):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise PatternError(f"{target}:{line_no} JSON が壊れている: {exc}")

            identifier = raw.get("id")
            elements = raw.get("elements")
            if not identifier or not isinstance(elements, list) or not elements:
                raise PatternError(f"{target}:{line_no} id と elements が要る")
            if identifier in seen:
                raise PatternError(f"{target}:{line_no} id が重複: {identifier}")
            seen.add(identifier)
            if CANDIDATE in elements and not raw.get("suggest_slot"):
                raise PatternError(
                    f"{target}:{line_no} CAND を使うなら suggest_slot が要る: "
                    f"{identifier}"
                )
            if PREDICATE not in elements:
                raise PatternError(
                    f"{target}:{line_no} 述語の無い文型は作れない: {identifier}"
                )
            patterns.append(
                Pattern(
                    id=str(identifier),
                    elements=[str(e) for e in elements],
                    need=[str(n) for n in (raw.get("need") or [])],
                    spec=dict(raw.get("spec") or {}),
                    note=str(raw.get("note", "")),
                    suggest_slot=str(raw.get("suggest_slot", "")),
                )
            )
        return cls(patterns)

    def __len__(self) -> int:
        return len(self.patterns)


class Composer:
    """節から返答の候補を組み立てる。

    【関連語を引く】
    空いている格には、辞書から引いた語を置ける。どの語が置けるかは
    フレームが宣言している（「行く」の NI は場所）。カテゴリ辞書を
    その値で引き、Facet で 1 件に決める。

    引いた語は相手が言っていない語なので、返答は言い換えではなくなる。
    そのぶん、置く場所を間違えると意味の通らない文になる。置けるのは
    「その格に何が入るか」をフレームが宣言している場合だけにしてある。
    """

    def __init__(
        self,
        patterns: PatternSet,
        conjugator: Conjugator,
        category_dict=None,
        frames=None,
    ) -> None:
        self.patterns = patterns
        self.conjugator = conjugator
        self.category_dict = category_dict
        self.frames = frames

    def suggest(self, clause: Clause, slot: str, avoid: set[str]) -> str | None:
        """空いている格に置ける語を 1 件引く。無ければ None。

        avoid には相手が使った語を渡す。相手の言葉をそのまま返すのは
        言い換えにしかならないので、候補から外す。
        """
        if self.category_dict is None or self.frames is None:
            return None
        if clause.predicate is None or slot in clause.slots:
            return None
        frame = self.frames.get(clause.predicate.lemma)
        if frame is None:
            return None
        wants = frame.content_constraints.get(slot)
        if not wants:
            # その格に何が入るかを宣言していない述語では引かない。
            # 推測で置くと「鍵をカレーした」のような文ができる。
            return None

        pool = []
        for value in wants:
            for entry in self.category_dict.members(content=value):
                if entry.surface not in avoid:
                    pool.append(entry)
        if not pool:
            return None

        # 会話で使う語に絞る。
        #
        # カテゴリには専門語も入っている（「情報」に 特徴量、「金銭」に
        # ボラティリティ）。頻度を見ないと「特徴量を作りますか」になる。
        # 上位を残すのは、閾値を決め打ちするとカテゴリごとに効き方が
        # 変わるため。順位で切れば、どのカテゴリでも「よく出る側」が残る。
        pool.sort(key=lambda e: (not e.everyday, -e.frequency, e.id))
        # 日常語があるなら、その中だけから選ぶ。
        #
        # 順位で切るだけでは足りない。手で選んだ日常語は Facet を
        # 持たないものが多く、距離を測ると専門語（V2 由来で Facet 付き）に
        # 負ける。「香りを食べますか」はそれで出ていた。
        # 会話で使う語が 1 つでもあるなら、そちらを優先する。
        daily = [e for e in pool if e.everyday]
        pool = daily if daily else pool[: max(COMMON_KEEP, 1)]

        # 基準点をどこに置くかで、引ける語が変わる。
        #
        # 原点（すべて 0.0）を基準にすると、Facet の値が小さい語ばかりが
        # 選ばれる。「会う」の相手に「運転手」が出てくるのはこれが理由で、
        # 意味ではなく数値の小ささで決まってしまっていた。
        #
        # 相手の発話に既知の語があれば、その重心を基準にする（話題が近い語）。
        # 無ければカテゴリ自身の重心を基準にする（そのカテゴリらしい語）。
        target = self._centroid(
            [t for t in (clause.slots.values()) if t.entry_id]
        ) or self._centroid_of_entries(pool)
        chosen = select(pool, target)
        return chosen.surface if chosen else None

    @staticmethod
    def _centroid(tokens) -> dict[str, float]:
        """トークンが持つ Facet の平均。取れなければ空。"""
        facets = []
        for token in tokens:
            if getattr(token, "facet", None):
                facets.append(token.facet)
        return Composer._mean(facets)

    @staticmethod
    def _centroid_of_entries(entries) -> dict[str, float]:
        return Composer._mean([e.facet for e in entries if e.facet])

    @staticmethod
    def _mean(facets: list[dict]) -> dict[str, float]:
        if not facets:
            return {}
        axes = ("physical", "psychological", "temporal", "logical")
        return {
            axis: sum(float(f.get(axis, 0.0)) for f in facets) / len(facets)
            for axis in axes
        }

    def candidates(self, clause: Clause, avoid: set[str] | None = None
                   ) -> list[Candidate]:
        """使える文型をすべて適用する。並びは文型の登録順。

        順序を保つのは、選ぶ側が決定的に選べるようにするため。
        集合にすると実行ごとに変わりうる。
        """
        predicate = clause.predicate
        if predicate is None:
            return []
        if not self.conjugator.knows(predicate.conjugation):
            # 活用させられない述語では文型を埋められない。
            return []

        filled: list[Candidate] = []
        for pattern in self.patterns.patterns:
            text = self._apply(pattern, clause, avoid or set())
            if text is not None:
                filled.append(Candidate(text=text, pattern_id=pattern.id))
        return filled

    def _apply(self, pattern: Pattern, clause: Clause,
               avoid: set[str]) -> str | None:
        words = self._words(clause)
        if any(name not in words for name in pattern.need):
            return None

        # 関連語を置く文型なら、置ける語を先に引く。
        # 引けなければこの文型は使わない（空欄のまま出さない）。
        suggested = None
        if CANDIDATE in pattern.elements:
            slot = pattern.suggest_slot
            if not slot:
                return None
            suggested = self.suggest(clause, slot, avoid | set(words.values()))
            if suggested is None:
                return None

        predicate = clause.predicate
        spec = self._spec(pattern, predicate)
        realized = self.conjugator.realize(
            predicate.lemma, predicate.conjugation, spec
        )
        if realized is None:
            return None

        parts: list[str] = []
        for element in pattern.elements:
            if element == PREDICATE:
                parts.append(realized)
            elif element == CANDIDATE:
                parts.append(suggested)
            elif element == CANDIDATE_PARTICLE:
                parts.append(PARTICLES.get(pattern.suggest_slot, ""))
            elif element in SLOT_NAMES:
                word = words.get(element)
                if word is None:
                    return None
                parts.append(word)
            else:
                parts.append(element)
        return "".join(parts)

    @staticmethod
    def _words(clause: Clause) -> dict[str, str]:
        """節から埋められる単語を集める。表層形をそのまま使う。"""
        words = {
            name: token.surface for name, token in clause.slots.items()
        }
        if clause.topic is not None:
            words["TOPIC"] = clause.topic.surface
        return words

    @staticmethod
    def _spec(pattern: Pattern, predicate) -> Spec:
        """文型の指定と、相手の発話の素性を合わせる。

        文型が書いていない素性は相手の発話から引き継ぐ。
        「鍵をなくした」への確認は「なくしたんだな」であって、
        時制を勝手に変えない。時制を動かすと決めた文型だけが上書きする。
        """
        from .ir import NEGATIVE, PAST, POLITE

        spec = pattern.spec
        return Spec(
            polite=bool(spec.get("polite", predicate.has(POLITE))),
            negative=bool(spec.get("negative", predicate.has(NEGATIVE))),
            past=bool(spec.get("past", predicate.has(PAST))),
            mood=str(spec.get("mood", "plain")),
            final=str(spec.get("final", "")),
        )
