"""
モダリティ判定の検証。

辞書に依存しないので、どんな入力にも必ず答えが出ることを重視する。
判定順序を変えると結果が変わる箇所（依頼 > 推量 > 疑問）を重点的に見る。
"""

from __future__ import annotations

import pytest

from cognitag_struct.modality import (
    AGREEMENT,
    INFORMING,
    Modality,
    detect_modality,
    needs_knowledge,
)


def detect(tokenizer, text: str):
    return detect_modality(tokenizer.tokenize(text), text)


class Test依頼:
    @pytest.mark.parametrize(
        "text",
        [
            "資料を作ってください",
            "手伝ってくれる",
            "手伝ってもらえますか",
            "お願いします",
            "座れ",
        ],
    )
    def test_依頼と判定する(self, tokenizer, text):
        assert detect(tokenizer, text).modality is Modality.REQUEST

    def test_疑問形の依頼は依頼が勝つ(self, tokenizer):
        """「手伝ってもらえますか」は疑問形だが、答えるべきは肯否ではない。"""
        result = detect(tokenizer, "手伝ってもらえますか")
        assert result.modality is Modality.REQUEST
        assert result.modality is not Modality.Q_YESNO


class Test推量:
    @pytest.mark.parametrize(
        "text",
        [
            "明日は雨だろう",
            "行くかな",
            "たぶん行くと思う",
            "明日名古屋に行くかもしれません",
        ],
    )
    def test_推量と判定する(self, tokenizer, text):
        assert detect(tokenizer, text).modality is Modality.SPECULATION

    def test_かなを疑問と取り違えない(self, tokenizer):
        """「行くかな」の「か」は終助詞だが問いではない。

        Q_YESNO より先に推量を見る理由がここにある。
        """
        result = detect(tokenizer, "行くかな")
        assert result.modality is Modality.SPECULATION
        assert "かな" in result.evidence[0]

    def test_かもしれないも推量(self, tokenizer):
        result = detect(tokenizer, "熱いかもしれません")
        assert result.modality is Modality.SPECULATION


class Test疑問:
    def test_疑問詞つきはQ_OPEN(self, tokenizer):
        result = detect(tokenizer, "どこに行きますか")

        assert result.modality is Modality.Q_OPEN
        assert result.interrogatives == ["どこ"]

    def test_疑問詞なしはQ_YESNO(self, tokenizer):
        result = detect(tokenizer, "名古屋に行きますか")

        assert result.modality is Modality.Q_YESNO
        assert result.interrogatives == []

    def test_疑問符でも判定できる(self, tokenizer):
        """終助詞が無くても「？」があれば問い。"""
        assert detect(tokenizer, "名古屋に行く？").modality is Modality.Q_YESNO

    @pytest.mark.parametrize(
        "text,word",
        [("何を作りますか", "何"), ("なぜ行きませんか", "なぜ"), ("いつ来ますか", "いつ")],
    )
    def test_各疑問詞を拾う(self, tokenizer, text, word):
        result = detect(tokenizer, text)
        assert result.modality is Modality.Q_OPEN
        assert word in result.interrogatives


class Test願望:
    def test_たいは願望(self, tokenizer):
        assert detect(tokenizer, "名古屋に行きたい").modality is Modality.DESIRE

    def test_ほしいも願望(self, tokenizer):
        assert detect(tokenizer, "資料がほしい").modality is Modality.DESIRE

    def test_願望の疑問は問いが勝つ(self, tokenizer):
        """「どこに行きたいですか」は相手の願望を尋ねている。答えが要る。"""
        result = detect(tokenizer, "どこに行きたいですか")
        assert result.modality is Modality.Q_OPEN


class Test平叙:
    @pytest.mark.parametrize(
        "text", ["明日は名古屋に行きません", "犬が男を噛んだ", "Piが熱い"]
    )
    def test_平叙と判定する(self, tokenizer, text):
        assert detect(tokenizer, text).modality is Modality.STATEMENT


class Test対人態度:
    def test_ねは同意要求(self, tokenizer):
        result = detect(tokenizer, "疲れたね")
        assert AGREEMENT in result.attitude
        assert result.modality is Modality.STATEMENT

    def test_よは情報提供(self, tokenizer):
        result = detect(tokenizer, "もう終わったよ")
        assert INFORMING in result.attitude

    def test_態度はモダリティと独立(self, tokenizer):
        """平叙でも問いでも態度は付きうる。"""
        assert detect(tokenizer, "疲れたね").attitude == [AGREEMENT]
        assert detect(tokenizer, "行きますか").attitude == []


class Test素性のまとめ:
    def test_否定と丁寧を拾う(self, tokenizer):
        result = detect(tokenizer, "明日は名古屋に行きません")

        assert result.negative is True
        assert result.polite is True
        assert result.past is False

    def test_過去を拾う(self, tokenizer):
        result = detect(tokenizer, "もう終わったよ")
        assert result.past is True


class Test辞書に依存しない:
    def test_解析できない文でも判定できる(self, tokenizer, frames):
        """3 節以上で parse() が Unparsable を返す文でも動く。

        ルーティングは全ての入力に方針を出さなければならないので、
        構文解析の成否に依存してはいけない。
        """
        from cognitag_struct.parse import Unparsable, parse

        text = "雨が降ったら友人に会いに名古屋へ行きますか"
        tokens = tokenizer.tokenize(text)

        assert isinstance(parse(tokens, text, frames), Unparsable)
        assert detect_modality(tokens, text).modality is Modality.Q_YESNO

    def test_辞書に無い語だけの文でも動く(self, tokenizer):
        result = detect(tokenizer, "にゃんにゃんしますか")
        assert result.modality is Modality.Q_YESNO

    def test_空入力でも例外を出さない(self):
        result = detect_modality([], "")
        assert result.modality is Modality.STATEMENT


class Test知識要求の判定:
    def test_Q_OPENは知識が要る(self, tokenizer):
        assert needs_knowledge(detect(tokenizer, "どこに行きますか")) is True

    def test_それ以外は要らない(self, tokenizer):
        for text in ("行きますか", "作ってください", "行きたい", "疲れたね"):
            assert needs_knowledge(detect(tokenizer, text)) is False


class Test決定性:
    def test_10回実行して一致する(self, tokenizer):
        text = "どこに行きますか"
        results = {str(detect(tokenizer, text)) for _ in range(10)}
        assert len(results) == 1
