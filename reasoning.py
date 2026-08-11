"""
入力をタグにまとめ、タグから含意を引く。

返答が「理解した」だけで終わっていたのは、内容のタグが取れていなかった
ためである。何の話かが分からなければ、それについて言えることも無い。

    疲れた
      → タグ #疲労 #感情 #過去
      → 含意 「疲れなら無理をしない方がいい」
      → 返答 「疲れの話だな。休息が足りていないなら、無理をしない方がいい」

【タグの出どころは 3 つ】
    述語のフレーム    #疲労 #移動 #作業 …（frames.jsonl の tags）
    文の形（モダリティ） #意志 #願望 #伝聞 …
    素性              #否定 #過去
    節の修飾          #仮定 #不確定

名詞のタグは今は取れない。CogniTag 本体辞書（11,980 語）を繋げば
「名古屋 → #場所」なども入るが、それは別の段階の作業である。

【なぜタグを挟むか】
述語ごとに返答を書くと 118 通り要る。語を足すたびに返答も書く羽目になる。
タグを挟めば含意は 14 通りで済み、frames.jsonl にタグを 1 つ足すだけで
返答が付いてくる。
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .frames import FrameDict
from .ir import IR, NEGATIVE, PAST, Token
from .modality import Modality, ModalityResult

# モダリティ由来のタグ。文の形から付く。
MODALITY_TAGS: dict[Modality, str] = {
    Modality.VOLITION: "#意志",
    Modality.DESIRE: "#願望",
    Modality.SPECULATION: "#不確定",
}

# 素性由来のタグ。
FEATURE_TAGS: dict[str, str] = {
    NEGATIVE: "#否定",
    PAST: "#過去",
}

# 伝聞は推量の一種だが、含意が違う（出どころを問える）ので分ける。
HEARSAY_TAG = "#伝聞"

# 名詞の種類から付くタグ。Sudachi の品詞 3 番目を見るだけで取れる。
#
# CogniTag 本体辞書（11,980 語）を繋げばもっと細かい分類が入るが、
# 「名古屋は場所」「田中は人」程度の区別は解析器が既に知っている。
# 辞書を読み込まずに済む分、起動も速い。
NOUN_TAGS: dict[str, str] = {
    "地名": "#場所",
    "人名": "#人",
    "組織名": "#組織",
    "副詞可能": "#時間",
    "助数詞可能": "#数量",
    "数詞": "#数量",
}


class ReasoningError(ValueError):
    """含意表が読めないときに送出する。"""


@dataclass
class Implication:
    """1 つのタグについて言えること。"""

    tag: str
    label: str
    so: str
    because: str = ""

    def as_sentence(self) -> str:
        """「これは<label>だ。<because>なら<so>」の形にする。"""
        if self.because:
            return f"{self.label}だな。{self.because}なら、{self.so}"
        return f"{self.label}だな。{self.so}"


@dataclass
class Reasoning:
    """入力から取れたタグと、そこから言えること。"""

    tags: list[str] = field(default_factory=list)
    implications: list[Implication] = field(default_factory=list)

    @property
    def has_content(self) -> bool:
        return bool(self.implications)

    def summary(self) -> str:
        """タグの一覧。trace 表示用。"""
        return " ".join(self.tags) if self.tags else "（タグなし）"


class ImplicationTable:
    """reasoning.toml を読んで、タグから含意を引く。"""

    def __init__(self, table: dict[str, Implication]) -> None:
        self.table = table

    @classmethod
    def load(cls, path: str | Path) -> "ImplicationTable":
        target = Path(path)
        if not target.exists():
            raise ReasoningError(f"ファイルが無い: {target}")
        with target.open("rb") as handle:
            data = tomllib.load(handle)

        table: dict[str, Implication] = {}
        for tag, body in data.items():
            if not isinstance(body, dict):
                continue
            table[tag] = Implication(
                tag=tag,
                label=str(body.get("label", tag)),
                so=str(body.get("so", "")),
                because=str(body.get("because", "")),
            )
        return cls(table)

    def get(self, tag: str) -> Implication | None:
        return self.table.get(tag)

    def __len__(self) -> int:
        return len(self.table)


def collect_tags(
    ir: IR | None,
    modality: ModalityResult | None,
    frames: FrameDict,
    tokens: list[Token] | None = None,
) -> list[str]:
    """入力からタグを集める。重複は畳み、出現順を保つ。

    順序を保つのは、返答に使う 1 件を選ぶときに「先に出たものを優先」で
    決められるようにするため。集合にすると実行ごとに変わる。
    """
    tags: list[str] = []

    def add(tag: str) -> None:
        if tag and tag not in tags:
            tags.append(tag)

    # 1. 述語のフレームから内容タグ
    if isinstance(ir, IR):
        for clause in ir.clauses:
            if clause.predicate is None:
                continue
            frame = frames.get(clause.predicate.lemma)
            if frame:
                for tag in frame.tags:
                    add(tag)
            # 2. 節の修飾（#仮定 #不確定 #質問）
            for modifier in clause.modifiers:
                add(modifier)

    # 2.5 名詞の種類。解析器が知っている区別をそのまま使う。
    #     句辞書由来のトークンは form / content を持つのでそれも拾う。
    for token in tokens or []:
        if token.is_phrase:
            for tag in token.tags():
                add(tag)
        elif token.pos == "名詞":
            add(NOUN_TAGS.get(token.subpos2, ""))
            add(NOUN_TAGS.get(token.subpos, ""))

    if modality is not None:
        # 3. 文の形。
        #    伝聞を先に見る。「らしい」は推量の一種として扱っているが、
        #    含意は「出どころを知りたい」で #不確定 より具体的なので、
        #    上限で切られる前に入れておきたい。
        if any("伝聞" in reason for reason in modality.evidence):
            add(HEARSAY_TAG)
        add(MODALITY_TAGS.get(modality.modality, ""))
        # 4. 素性
        if modality.negative:
            add(FEATURE_TAGS[NEGATIVE])
        if modality.past:
            add(FEATURE_TAGS[PAST])

    return tags


# 粗いタグと、それを言い換えたより具体的なタグの対応。
#
# 両方の含意を並べると重複する。「気持ちの話だな。まず受け止めたい。
# しんどい話だな。正論より先に…」は同じことを 2 回言っている。
# 具体的な方があるなら、粗い方の含意は落とす。
#
# タグ自体は両方残す。落とすのは含意（返答に出る文）だけ。
# タグは分類の記録なので、消すと後から追えなくなる。
SUBSUMED_BY: dict[str, frozenset[str]] = {
    "#感情": frozenset({"#つらさ", "#喜び", "#疲労"}),
    "#変化": frozenset({"#困難", "#達成"}),
}


def reason(
    ir: IR | None,
    modality: ModalityResult | None,
    frames: FrameDict,
    table: ImplicationTable,
    limit: int = 2,
    tokens: list[Token] | None = None,
) -> Reasoning:
    """タグを集め、含意を引く。

    limit を超える含意は返さない。全部並べると返答が長くなるだけで、
    読み手には響かない。先に出たタグ（＝述語由来の内容タグ）を優先する。
    """
    tags = collect_tags(ir, modality, frames, tokens)
    present = set(tags)

    implications = []
    for tag in tags:
        # より具体的なタグが同席しているなら、粗い方の含意は出さない
        specific = SUBSUMED_BY.get(tag)
        if specific and present & specific:
            continue
        found = table.get(tag)
        if found is not None and found.so:
            implications.append(found)
        if len(implications) >= limit:
            break
    return Reasoning(tags=tags, implications=implications)
