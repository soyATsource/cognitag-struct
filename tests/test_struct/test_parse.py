"""
格解析の検証。依頼された 8 例をすべて含む。
"""

from __future__ import annotations

from cognitag_struct.ir import CONDITION, NONE, PURPOSE
from cognitag_struct.parse import (
    HYPOTHETICAL,
    QUESTION,
    UNCERTAIN,
    Unparsable,
    parse,
)


def analyze(tokenizer, frames, text):
    return parse(tokenizer.tokenize(text), source_text=text, frames=frames)


class Test1_格と主題:
    def test_明日は名古屋に行きません(self, tokenizer, frames):
        ir = analyze(tokenizer, frames, "明日は名古屋に行きません")

        assert not isinstance(ir, Unparsable)
        assert len(ir.clauses) == 1
        clause = ir.clauses[0]

        assert clause.predicate.lemma == "行く"
        assert clause.slots["NI"].surface == "名古屋"
        assert clause.topic.surface == "明日"
        # 「は」は格ではないので格スロットには入らない
        assert "GA" not in clause.slots

    def test_明日は格スロットに入らない(self, tokenizer, frames):
        ir = analyze(tokenizer, frames, "明日は名古屋に行きません")
        slots = ir.clauses[0].slots
        assert all(t.surface != "明日" for t in slots.values())


class Test2_格の入れ替え:
    def test_犬が男を噛んだ(self, tokenizer, frames):
        ir = analyze(tokenizer, frames, "犬が男を噛んだ")

        clause = ir.clauses[0]
        assert clause.slots["GA"].surface == "犬"
        assert clause.slots["WO"].surface == "男"

    def test_男が犬を噛んだ(self, tokenizer, frames):
        ir = analyze(tokenizer, frames, "男が犬を噛んだ")

        clause = ir.clauses[0]
        assert clause.slots["GA"].surface == "男"
        assert clause.slots["WO"].surface == "犬"

    def test_語順が同じでも格が違えばIRが異なる(self, tokenizer, frames):
        """語彙は同一。格の割り当てだけが違う。

        Facet の加重平均ではこの 2 文を区別できない。構造層を置く理由。
        """
        a = analyze(tokenizer, frames, "犬が男を噛んだ").clauses[0]
        b = analyze(tokenizer, frames, "男が犬を噛んだ").clauses[0]

        assert a.slots["GA"].surface != b.slots["GA"].surface
        assert a.slots["WO"].surface != b.slots["WO"].surface


class Test3_必須スロットが空:
    def test_行きますはNIが空(self, tokenizer, frames):
        ir = analyze(tokenizer, frames, "行きます")

        assert len(ir.clauses) == 1
        assert ir.clauses[0].predicate.lemma == "行く"
        assert "NI" not in ir.clauses[0].slots


class Test4_不確定:
    def test_かもしれませんで1節のまま(self, tokenizer, frames):
        """「かもしれません」は動詞「しれる」を含むが、用言として数えない。

        数えると 2 節になり、「しれる」が述語の節ができてしまう。
        """
        ir = analyze(tokenizer, frames, "明日名古屋に行くかもしれません")

        assert not isinstance(ir, Unparsable)
        assert len(ir.clauses) == 1
        assert ir.clauses[0].predicate.lemma == "行く"

    def test_不確定タグが付く(self, tokenizer, frames):
        ir = analyze(tokenizer, frames, "明日名古屋に行くかもしれません")
        assert UNCERTAIN in ir.clauses[0].modifiers

    def test_不確定でも節を破棄しない(self, tokenizer, frames):
        """予定としては成立するので保持する。"""
        ir = analyze(tokenizer, frames, "明日名古屋に行くかもしれません")
        assert ir.clauses[0].slots["NI"].surface == "名古屋"

    def test_形容詞でも1節のまま(self, tokenizer, frames):
        ir = analyze(tokenizer, frames, "熱いかもしれません")
        assert len(ir.clauses) == 1
        assert ir.clauses[0].predicate.lemma == "熱い"


class Test5_質問:
    def test_質問タグが付く(self, tokenizer, frames):
        ir = analyze(tokenizer, frames, "明日はどこに行きますか")
        assert QUESTION in ir.clauses[0].modifiers

    def test_疑問詞はNIに入る(self, tokenizer, frames):
        """構造としては埋まる。空スロット扱いにするのは gap.py の役目。"""
        ir = analyze(tokenizer, frames, "明日はどこに行きますか")
        assert ir.clauses[0].slots["NI"].surface == "どこ"

    def test_かもしれないのかと混ざらない(self, tokenizer, frames):
        """「かもしれない」の「か」は副助詞、文末の「か」は終助詞。"""
        ir = analyze(tokenizer, frames, "明日名古屋に行くかもしれません")
        assert QUESTION not in ir.clauses[0].modifiers


class Test6_条件節:
    def test_2節になる(self, tokenizer, frames):
        ir = analyze(tokenizer, frames, "雨が降ったら名古屋には行きません")
        assert len(ir.clauses) == 2

    def test_relationがCONDITION(self, tokenizer, frames):
        ir = analyze(tokenizer, frames, "雨が降ったら名古屋には行きません")
        assert ir.relation == CONDITION

    def test_条件節が前で帰結節が後(self, tokenizer, frames):
        ir = analyze(tokenizer, frames, "雨が降ったら名古屋には行きません")

        assert ir.clauses[0].predicate.lemma == "降る"
        assert HYPOTHETICAL in ir.clauses[0].modifiers
        assert ir.clauses[1].predicate.lemma == "行く"

    def test_2節目で名古屋がNIかつtopic(self, tokenizer, frames):
        """「には」は格と主題を同時に持つ。両者は排他ではない。"""
        ir = analyze(tokenizer, frames, "雨が降ったら名古屋には行きません")
        second = ir.clauses[1]

        assert second.slots["NI"].surface == "名古屋"
        assert second.topic.surface == "名古屋"

    def test_1節目の雨はGA(self, tokenizer, frames):
        ir = analyze(tokenizer, frames, "雨が降ったら名古屋には行きません")
        assert ir.clauses[0].slots["GA"].surface == "雨"

    def test_ばでも条件になる(self, tokenizer, frames):
        ir = analyze(tokenizer, frames, "雨が降れば行きます")
        assert ir.relation == CONDITION
        assert HYPOTHETICAL in ir.clauses[0].modifiers


class Test7_目的:
    def test_2節になる(self, tokenizer, frames):
        ir = analyze(tokenizer, frames, "明日友人に会いに名古屋へ行きます")
        assert len(ir.clauses) == 2

    def test_relationがPURPOSE(self, tokenizer, frames):
        ir = analyze(tokenizer, frames, "明日友人に会いに名古屋へ行きます")
        assert ir.relation == PURPOSE

    def test_各節の述語(self, tokenizer, frames):
        ir = analyze(tokenizer, frames, "明日友人に会いに名古屋へ行きます")

        assert ir.clauses[0].predicate.lemma == "会う"
        assert ir.clauses[1].predicate.lemma == "行く"
        assert ir.clauses[0].slots["NI"].surface == "友人"
        assert ir.clauses[1].slots["HE"].surface == "名古屋"

    def test_名詞プラスにはPURPOSEにしない(self, tokenizer, frames):
        """「名古屋に行きます」の「に」は連用形の後ろではない。"""
        ir = analyze(tokenizer, frames, "名古屋に行きます")
        assert ir.relation is None  # 1 節なので relation 自体が無い


class Test8_解釈不能:
    def test_3節以上は解釈しない(self, tokenizer, frames):
        result = analyze(
            tokenizer, frames, "雨が降ったら友人に会いに名古屋へ行きます"
        )

        assert isinstance(result, Unparsable)
        assert "3" in result.reason
        assert result.source_text == "雨が降ったら友人に会いに名古屋へ行きます"

    def test_例外ではなく戻り値(self, tokenizer, frames):
        """呼び出し側が通常の分岐として扱えること。"""
        result = analyze(tokenizer, frames, "雨が降ったら友人に会いに名古屋へ行きます")
        assert not result  # __bool__ が False

    def test_述語が無ければ解釈不能(self, tokenizer, frames):
        result = analyze(tokenizer, frames, "名古屋")
        assert isinstance(result, Unparsable)

    def test_空入力(self, tokenizer, frames):
        assert isinstance(parse([], frames=frames), Unparsable)


class Test無助詞の名詞:
    def test_副詞可能名詞は格に入れない(self, tokenizer, frames):
        """「明日」は助詞なしで時を表す。格ではないので GA に入れない。"""
        ir = analyze(tokenizer, frames, "明日名古屋に行くかもしれません")
        clause = ir.clauses[0]

        assert "GA" not in clause.slots
        assert all(t.surface != "明日" for t in clause.slots.values())

    def test_助詞が落ちた名詞は格を補う(self, tokenizer, frames):
        """「僕行きます」の「僕」は副詞可能名詞ではないので補完対象。"""
        ir = analyze(tokenizer, frames, "僕行きます")
        assert ir.clauses[0].slots["NI"].surface == "僕" or (
            ir.clauses[0].slots.get("GA") is not None
        )

    def test_フレームが無ければ補完しない(self, tokenizer, frames):
        """根拠が無いまま推測しない。"""
        ir = analyze(tokenizer, frames, "犬が男を噛んだ")
        assert frames.get("噛む") is None
        assert set(ir.clauses[0].slots) == {"GA", "WO"}
