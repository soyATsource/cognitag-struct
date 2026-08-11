"""
中間表現（IR）の型定義。

CogniTag 本体はテキストを形態素に割って 4 軸 Facet の TF-IDF 加重平均を返す。
話題の検出には足りるが、語順・格関係・否定を捨てるため命題を扱えない。
その手前に構造層を置き、Facet は「構造層が絞り込んだ候補から
どの語を選ぶか」を決める役割へ移す。

このモジュールは型の定義だけを持つ。値を埋める処理（格解析・生成）は
次の段階で実装するため、ここには parse 相当のロジックを書かないこと。

【tags を持たない理由】
form / content という軸の値と、"#ことわざ" のようなタグ表記の両方を
フィールドとして保持すると、片方だけ更新されたときに矛盾する。
タグが必要な箇所は tags() で導出する。導出なら不整合が起こりえない。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- 機能素 ---------------------------------------------------------------
#
# 意味を変えない文法的な装飾だけをここに置く。
# 可能・受身・使役は意味を変えるので機能素にしない（tokenizer.py 参照）。
POLITE = "POLITE"
NEGATIVE = "NEGATIVE"
PAST = "PAST"

FEATURES: tuple[str, ...] = (POLITE, NEGATIVE, PAST)


# --- 格 -------------------------------------------------------------------
#
# Clause.slots のキー。助詞そのものではなく格の名前で持つ。
# 「は」は格ではなく主題の提示なので、ここには入れず Clause.topic に置く。
GA = "GA"      # が
WO = "WO"      # を
NI = "NI"      # に
HE = "HE"      # へ
KARA = "KARA"  # から
MADE = "MADE"  # まで

CASES: tuple[str, ...] = (GA, WO, NI, HE, KARA, MADE)


# --- 節の関係 -------------------------------------------------------------
CONDITION = "CONDITION"
PURPOSE = "PURPOSE"
NONE = "NONE"

RELATIONS: tuple[str, ...] = (CONDITION, PURPOSE, NONE)


# --- 既定値 ---------------------------------------------------------------
#
# 句辞書に無い語はここに落ちる。既存 CogniTag 辞書のカテゴリとの
# 対応付けは次の段階で行う。
DEFAULT_FORM = "一般語"


@dataclass
class Token:
    """1 トークン。句辞書由来か、形態素解析器由来かのどちらか。

    助動詞は独立したトークンにせず、直前の用言へ features として吸収する
    （tokenizer.py 参照）。したがって span は吸収した範囲全体を指す。
    """

    surface: str
    lemma: str
    pos: str
    # 品詞細分類。SudachiPy の part_of_speech()[1]。
    #
    # 格助詞（が/を/に）と係助詞（は）は pos がどちらも「助詞」で区別できない。
    # 「は」は格ではなく主題の提示であり、Clause.slots ではなく
    # Clause.topic へ振り分けなければならない。その判別に subpos が要る。
    #
    # 形態素由来のトークンには Sudachi の値をそのまま入れる（該当なしは "*"）。
    # 句辞書由来のトークンは既定の空文字のままにする。
    subpos: str = ""
    # 品詞のさらに細かい区分。SudachiPy の part_of_speech()[2]。
    #
    # 名詞の中身を見分けるのに要る。
    #     名古屋 → 固有名詞 / 地名     → #場所
    #     明日   → 普通名詞 / 副詞可能 → 時を表す。格に入れない
    #     連絡   → 普通名詞 / サ変可能 → する と結んで 1 語にする
    # pos と subpos だけでは「名詞」までしか分からない。
    subpos2: str = ""
    # 活用形。SudachiPy の part_of_speech()[5]。
    #
    # 「たら」と「た」はどちらも見出し語が「た」で、活用形だけが違う
    # （仮定形-一般 / 終止形-一般）。条件と過去の区別はここにしか無い。
    inflection: str = ""
    form: str = DEFAULT_FORM
    content: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    is_phrase: bool = False
    entry_id: str | None = None
    span: tuple[int, int] = (0, 0)

    def tags(self) -> list[str]:
        """form / content からタグ表記を導出する。

        フィールドとして持たないのは、軸の値との二重管理を避けるため。
        """
        return [f"#{self.form}"] + [f"#{value}" for value in self.content]

    def has(self, feature: str) -> bool:
        return feature in self.features


@dataclass
class Clause:
    """1 つの節。述語と、それに係る格要素。

    topic を slots と分けているのは、「は」が格ではないため。
    「象は鼻が長い」の「象」は主格ではなく主題であり、
    同じ入れ物に入れると格解析が壊れる。
    """

    predicate: Token | None = None
    slots: dict[str, Token] = field(default_factory=dict)
    modifiers: list[str] = field(default_factory=list)
    topic: Token | None = None
    # 程度修飾。「ちょっと熱い」の「ちょっと」。
    # 生成の verbosity 1 で足す要素で、命題の真偽には関わらない。
    degree: Token | None = None
    # 補足情報。「熱い」に対する「48度」のような、節の根拠になる値。
    # verbosity 3 で別文として添える。IR ではなく Clause に置くのは、
    # 節が 2 つある文でどちらの補足かを保つため。
    supplements: list[Token] = field(default_factory=list)


@dataclass
class IR:
    """1 入力分の中間表現。"""

    clauses: list[Clause] = field(default_factory=list)
    relation: str | None = None
    source_text: str = ""
