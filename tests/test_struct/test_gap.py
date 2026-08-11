"""
欠落スロットの検出と質問生成の検証。
"""

from __future__ import annotations

from cognitag_struct.gap import SPEAKER, detect
from cognitag_struct.parse import parse


def report(tokenizer, frames, questions, text):
    ir = parse(tokenizer.tokenize(text), source_text=text, frames=frames)
    return detect(ir, frames, questions)


class Test空スロットの検出:
    def test_行きますはNIが空で質問される(self, tokenizer, frames, questions):
        result = report(tokenizer, frames, questions, "行きます")

        assert result.has_gap
        assert [g.slot for g in result.gaps] == ["NI"]
        assert result.questions() == ["どこに？"]

    def test_行き先があれば質問しない(self, tokenizer, frames, questions):
        result = report(tokenizer, frames, questions, "名古屋に行きます")
        assert not result.has_gap

    def test_述語ごとに質問文を変えられる(self, tokenizer, frames, questions):
        """「会う」の NI は場所ではなく相手なので「誰に？」。"""
        result = report(tokenizer, frames, questions, "会います")

        assert [g.slot for g in result.gaps] == ["NI"]
        assert result.questions() == ["誰に？"]

    def test_文言はtomlから読む(self, questions):
        """コードに直書きしていないこと。"""
        assert questions.question_for("行く", "NI") == "どこに？"
        assert questions.question_for("会う", "NI") == "誰に？"
        assert questions.question_for("熱い", "GA") == "何が？"

    def test_知らない述語は判定しない(self, tokenizer, frames, questions):
        """フレームが無いものを「欠落あり」と報告しない。"""
        result = report(tokenizer, frames, questions, "犬が男を噛んだ")
        assert not result.has_gap


class Test主語の補完:
    def test_丁寧語と意志動詞なら発話者を補う(self, tokenizer, frames, questions):
        """「明日は名古屋に行きません」の主語は発話者。質問しない。"""
        result = report(tokenizer, frames, questions, "明日は名古屋に行きません")

        assert not result.has_gap
        assert [f.slot for f in result.filled] == ["GA"]
        assert result.filled[0].value == SPEAKER

    def test_丁寧語が無ければ補わない(self, tokenizer, frames, questions):
        """「行く」だけでは独り言や引用の可能性がある。

        GA は optional なので質問はされないが、補完もされない。
        """
        result = report(tokenizer, frames, questions, "名古屋に行く")
        assert result.filled == []

    def test_意志動詞でなければ補わない(self, tokenizer, frames, questions):
        """「落ちます」の主語は発話者ではない。GA は必須なので質問される。"""
        result = report(tokenizer, frames, questions, "落ちます")

        assert result.filled == []
        assert [g.slot for g in result.gaps] == ["GA"]
        assert result.questions() == ["何が？"]

    def test_可能動詞は意志動詞でない(self, tokenizer, frames, questions):
        """「行けません」は能力の話で、意志の表明ではない。"""
        result = report(tokenizer, frames, questions, "名古屋に行けません")
        assert result.filled == []


class Test疑問詞:
    def test_疑問詞は空スロット扱い(self, tokenizer, frames, questions):
        """「どこに行きますか」の NI は値ではなく、値を尋ねる印。

        埋まったとみなすと、質問に質問で答えられなくなる。
        """
        result = report(tokenizer, frames, questions, "明日はどこに行きますか")

        assert result.has_gap
        assert [g.slot for g in result.gaps] == ["NI"]

    def test_具体的な地名なら埋まっている(self, tokenizer, frames, questions):
        result = report(tokenizer, frames, questions, "明日はどこに行きますか")
        filled = report(tokenizer, frames, questions, "明日は名古屋に行きますか")

        assert result.has_gap
        assert not filled.has_gap


class Test複数節:
    def test_節ごとに判定する(self, tokenizer, frames, questions):
        result = report(
            tokenizer, frames, questions, "雨が降ったら名古屋には行きません"
        )

        # 「降る」はフレームが無いので判定対象外。
        # 「行く」は NI が埋まり GA は発話者補完。
        assert not result.has_gap
        assert [f.clause_index for f in result.filled] == [1]

    def test_gapに節の番号が入る(self, tokenizer, frames, questions):
        result = report(tokenizer, frames, questions, "雨が降ったら行きますか")
        for gap in result.gaps:
            assert gap.clause_index == 1


class Test出力の形:
    def test_Gapは読める文字列になる(self, tokenizer, frames, questions):
        result = report(tokenizer, frames, questions, "行きます")
        assert str(result.gaps[0]) == "行く[NI] どこに？"

    def test_質問が無ければ空リスト(self, tokenizer, frames, questions):
        result = report(tokenizer, frames, questions, "名古屋に行きます")
        assert result.questions() == []
