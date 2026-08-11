"""
IR から日本語文を生成する。

【核から周辺へ足す】
核となる文をまず作り、verbosity の値だけ増分を足す。
どの段階で止めても文として成立する。

    0: Piが熱い
    1: Piがちょっと熱い
    2: Piが今日はちょっと熱い
    3: Pi、今日はちょっと熱い。48度

理解では周辺を削って核を取り出し、生成では核から周辺を足す。
同じ構造を逆向きに使うので、解析と生成で表現を共有できる。
parse.py が作った IR も、Sentinel の観測値から直接組み立てた IR も、
同じようにここへ渡せる。

【出力しなかった要素を捨てない】
verbosity を下げて省いた要素は Utterance.withheld に残す。
「今日はちょっと熱い」と言ったあとに「何度？」と問われたら
48度と答えられなければならない。破棄すると答えようがなくなる。

【段階を上げても下位段階を壊さない】
足すのは要素であって、書き換えではない。
verbosity 3 で主格が「が」から読点に変わるのは、
要素 Pi が失われたのではなく、助詞の付け方という表層の規則が
変わっただけである。要素の集合は増える一方になる。
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .ir import IR, Clause, Token
# 格助詞の対応は parse.py の 1 か所に置く。ここで定義し直すと、
# 解析と生成で助詞がずれたときに気づけない。
from .parse import CASE_PARTICLES

MIN_VERBOSITY = 0
MAX_VERBOSITY = 3

# 役割の名前。generation_style.toml の elements に書く値と対応する。
SUBJECT = "subject"
TOPIC = "topic"
CASES = "cases"
DEGREE = "degree"
PREDICATE = "predicate"
SUPPLEMENT = "supplement"

# 出力する順序。style は「どれを出すか」を決め、順序はここで決める。
ELEMENT_ORDER: tuple[str, ...] = (
    SUBJECT,
    TOPIC,
    CASES,
    DEGREE,
    PREDICATE,
    SUPPLEMENT,
)

# スロット名 -> 格助詞（parse.py の逆引き）
SLOT_PARTICLES: dict[str, str] = {
    slot: particle for particle, slot in CASE_PARTICLES.items()
}


class StyleError(ValueError):
    """スタイル定義が読めないときに送出する。"""


@dataclass
class VerbosityLevel:
    """1 段階分の構成。"""

    elements: list[str]
    subject_particle: str = "が"
    subject_comma: bool = False
    topic_particle: str = "は"
    supplement_separator: str = "。"


@dataclass
class Style:
    """文の組み立て方。語彙の好みは persona.toml 側の担当。"""

    first_person: str = "私"
    idiom_rate: float = 0.1
    levels: dict[int, VerbosityLevel] = field(default_factory=dict)
    # 応答の型。chat.py が使う。キーは発話の種類（小文字）。
    reply: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "Style":
        target = Path(path)
        if not target.exists():
            raise StyleError(f"ファイルが無い: {target}")
        with target.open("rb") as handle:
            data = tomllib.load(handle)

        general = data.get("general") or {}
        levels: dict[int, VerbosityLevel] = {}
        for key, table in (data.get("verbosity") or {}).items():
            try:
                index = int(key)
            except ValueError:
                raise StyleError(f"verbosity のキーが数値でない: {key}")
            levels[index] = VerbosityLevel(
                elements=list(table.get("elements") or []),
                subject_particle=str(table.get("subject_particle", "が")),
                subject_comma=bool(table.get("subject_comma", False)),
                topic_particle=str(table.get("topic_particle", "は")),
                supplement_separator=str(table.get("supplement_separator", "。")),
            )
        if not levels:
            raise StyleError(f"{target} に [verbosity.N] が無い")

        return cls(
            first_person=str(general.get("first_person", "私")),
            idiom_rate=float(general.get("idiom_rate", 0.1)),
            levels=levels,
            reply={k: str(v) for k, v in (data.get("reply") or {}).items()},
        )

    def level(self, verbosity: int) -> VerbosityLevel:
        return self.levels[clamp_verbosity(verbosity, self.levels)]


@dataclass
class Utterance:
    """生成結果。"""

    text: str
    verbosity: int
    # 出力しなかった要素。役割 -> Token の一覧。
    # 表層形だけでなく Token ごと保持するのは、後から問われたときに
    # 品詞や位置まで含めて答えられるようにするため。
    withheld: dict[str, list[Token]] = field(default_factory=dict)
    used_idiom: str | None = None

    def withheld_surfaces(self) -> dict[str, list[str]]:
        """目視と検査のために表層形だけ取り出す。"""
        return {
            role: [t.surface for t in tokens]
            for role, tokens in self.withheld.items()
        }


def clamp_verbosity(verbosity: int, levels: dict[int, VerbosityLevel]) -> int:
    """範囲外は例外にせず最近傍へ丸める。

    呼び出し側が 5 や -1 を渡すのは、たいてい計算の結果であって
    誤りではない。落とすより、いちばん近い段階で喋る方がよい。
    """
    available = sorted(levels)
    low, high = available[0], available[-1]
    if verbosity < low:
        return low
    if verbosity > high:
        return high
    if verbosity in levels:
        return verbosity
    return min(available, key=lambda v: abs(v - verbosity))


def elements_of(clause: Clause) -> dict[str, list[Token]]:
    """節を役割ごとの Token に分解する。

    出力するかどうかはここでは決めない。style が選び、
    選ばれなかったものが withheld へ回る。
    """
    subject = clause.slots.get("GA")
    return {
        SUBJECT: [subject] if subject is not None else [],
        TOPIC: [clause.topic] if clause.topic is not None else [],
        CASES: [
            token for slot, token in clause.slots.items() if slot != "GA"
        ],
        DEGREE: [clause.degree] if clause.degree is not None else [],
        PREDICATE: [clause.predicate] if clause.predicate is not None else [],
        SUPPLEMENT: list(clause.supplements),
    }


def _case_phrase(clause: Clause) -> str:
    """主格以外の格要素を並べる。スロット名の順序で決定的にする。"""
    parts = []
    for slot in SLOT_PARTICLES:
        token = clause.slots.get(slot)
        if slot == "GA" or token is None:
            continue
        parts.append(f"{token.surface}{SLOT_PARTICLES[slot]}")
    return "".join(parts)


def _render_clause(
    clause: Clause, level: VerbosityLevel
) -> tuple[str, str]:
    """1 つの節を (本文, 補足) に整形する。

    補足を分けて返すのは、別文として句点の後ろに置くため。
    """
    chosen = set(level.elements)
    body: list[str] = []
    supplement = ""

    for role in ELEMENT_ORDER:
        if role not in chosen:
            continue

        if role == SUBJECT:
            token = clause.slots.get("GA")
            if token is None:
                continue
            if level.subject_comma:
                body.append(f"{token.surface}、")
            else:
                body.append(f"{token.surface}{level.subject_particle}")

        elif role == TOPIC:
            if clause.topic is not None:
                body.append(f"{clause.topic.surface}{level.topic_particle}")

        elif role == CASES:
            body.append(_case_phrase(clause))

        elif role == DEGREE:
            if clause.degree is not None:
                body.append(clause.degree.surface)

        elif role == PREDICATE:
            if clause.predicate is not None:
                body.append(clause.predicate.surface)

        elif role == SUPPLEMENT:
            if clause.supplements:
                joined = "、".join(t.surface for t in clause.supplements)
                supplement = f"{level.supplement_separator}{joined}"

    return "".join(body), supplement


def generate(ir: IR, verbosity: int, style: Style) -> Utterance:
    """IR を文にする。

    verbosity が範囲外でも例外にせず丸める。
    出力しなかった要素は withheld に残して破棄しない。

    used_idiom は常に None。慣用句をどの位置に差し込むかは
    文体の設計そのもので、まだ決まっていない。
    choose_idiom() は idiom.py に独立した部品として用意してある。
    """
    resolved = clamp_verbosity(verbosity, style.levels)
    level = style.levels[resolved]
    chosen = set(level.elements)

    bodies: list[str] = []
    supplements: list[str] = []
    withheld: dict[str, list[Token]] = {}

    for clause in ir.clauses:
        body, supplement = _render_clause(clause, level)
        if body:
            bodies.append(body)
        if supplement:
            supplements.append(supplement)

        for role, tokens in elements_of(clause).items():
            if role in chosen or not tokens:
                continue
            withheld.setdefault(role, []).extend(tokens)

    text = "".join(bodies) + "".join(supplements)
    return Utterance(
        text=text,
        verbosity=resolved,
        withheld=withheld,
        used_idiom=None,
    )
