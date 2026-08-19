"""
タグの収集と含意の検証。

返答が「理解した」だけで終わっていたのは、内容のタグが取れて
いなかったため。ここが機能しているかがそのまま返答の中身になる。
"""

from __future__ import annotations

import pytest

from cognitag_struct import CogniTag, Modality


@pytest.fixture(scope="module")
def ct() -> CogniTag:
    return CogniTag()


def tags_of(ct: CogniTag, text: str) -> list[str]:
    return ct.analyze(text).reasoning.tags


class Test述語からタグが付く:
    @pytest.mark.parametrize(
        "text,tag",
        [
            ("名古屋に行きます", "#移動"),
            ("資料を作ります", "#作業"),
            ("疲れた", "#疲労"),
            ("悩んでる", "#思考"),
            ("壊れました", "#変化"),
            ("友人に連絡します", "#対人"),
            ("説明します", "#伝達"),
            ("難しいです", "#評価"),
        ],
    )
    def test_内容タグが取れる(self, ct: CogniTag, text: str, tag: str):
        assert tag in tags_of(ct, text)

    def test_複数のタグが付きうる(self, ct: CogniTag):
        """「疲れた」は感情でもあり疲労でもある。"""
        tags = tags_of(ct, "疲れた")
        assert "#感情" in tags
        assert "#疲労" in tags

    def test_フレームが無い述語はタグなし(self, ct: CogniTag):
        assert "#移動" not in tags_of(ct, "犬が男を噛んだ")


class Test文の形からタグが付く:
    def test_意志(self, ct: CogniTag):
        assert "#意志" in tags_of(ct, "どうしよう")

    def test_願望(self, ct: CogniTag):
        assert "#願望" in tags_of(ct, "名古屋に行きたい")

    def test_伝聞(self, ct: CogniTag):
        assert "#伝聞" in tags_of(ct, "雨が降るらしい")

    def test_否定と過去(self, ct: CogniTag):
        tags = tags_of(ct, "行きませんでした")
        assert "#否定" in tags
        assert "#過去" in tags

    def test_仮定(self, ct: CogniTag):
        assert "#仮定" in tags_of(ct, "雨が降ったら行きます")

    def test_伝聞は不確定より先に来る(self, ct: CogniTag):
        """含意の上限で切られる前に入れておきたい。

        「出どころを知りたい」は「確かめる余地がある」より具体的。
        """
        tags = tags_of(ct, "雨が降るらしい")
        assert tags.index("#伝聞") < tags.index("#不確定")


class Test含意が引ける:
    def test_疲労から休息の含意(self, ct: CogniTag):
        reasoning = ct.analyze("疲れた").reasoning

        assert reasoning.has_content
        texts = [i.as_sentence() for i in reasoning.implications]
        assert any("無理をしない" in t for t in texts)

    def test_移動から行き先の含意(self, ct: CogniTag):
        reasoning = ct.analyze("行きます").reasoning
        assert any("どこへ" in i.so for i in reasoning.implications)

    def test_becauseがあれば理由が入る(self, ct: CogniTag):
        """「これは<X>だ。<理由>なら<結論>」の形になる。"""
        reasoning = ct.analyze("名古屋に行きます").reasoning
        sentence = reasoning.implications[0].as_sentence()
        assert "なら、" in sentence

    def test_含意は絞られる(self, ct: CogniTag):
        """全部並べると長いだけなので上限をかけている。"""
        reasoning = ct.analyze("壊れました").reasoning
        assert len(reasoning.implications) <= 2

    def test_内容のタグが無ければ内容の含意も無い(self, ct: CogniTag):
        """「噛む」はフレームに無いので内容のタグは付かない。

        素性由来の #過去 は付く（「噛んだ」の「だ」を過去として扱うため）。
        内容について言えることが無い、という状態は保たれている。
        """
        reasoning = ct.analyze("犬が男を噛んだ").reasoning
        assert reasoning.tags == ["#過去"]
        assert [i.tag for i in reasoning.implications] == ["#過去"]


class Test解釈不能でもタグは取れる:
    def test_3節でもモダリティ由来のタグが付く(self, ct: CogniTag):
        result = ct.analyze("雨が降ったら友人に会いに名古屋へ行きたい")

        assert not result.parsed
        assert "#願望" in result.reasoning.tags

    def test_空入力でも落ちない(self, ct: CogniTag):
        assert ct.analyze("").reasoning.tags == []


class Test挨拶と意志の判定:
    @pytest.mark.parametrize("text", ["こんばんは", "こんにちは", "おはよう"])
    def test_挨拶を認識する(self, ct: CogniTag, text: str):
        """感動詞は述語を持たないので構文解析できないが、返せる。"""
        result = ct.analyze(text)
        assert result.modality.modality is Modality.GREETING

    @pytest.mark.parametrize("text", ["どうしよう", "何しようか", "行こう"])
    def test_意志を推量と区別する(self, ct: CogniTag, text: str):
        """活用形は同じ「意志推量形」だが、品詞で分かれる。"""
        assert ct.analyze(text).modality.modality is Modality.VOLITION

    @pytest.mark.parametrize("text", ["雨が降るだろう", "雨が降るらしい"])
    def test_推量は推量のまま(self, ct: CogniTag, text: str):
        assert ct.analyze(text).modality.modality is Modality.SPECULATION


class Test表層形が壊れない:
    def test_ているを畳む(self, ct: CogniTag):
        """畳まないと「悩ん」で切れ、言い直しが壊れた文になる。"""
        tokens = ct.tokenize("悩んでる")
        assert tokens[0].surface == "悩んでる"
        assert tokens[0].lemma == "悩む"

    def test_サ変動詞を畳む(self, ct: CogniTag):
        tokens = ct.tokenize("確認しました")
        assert len(tokens) == 1
        assert tokens[0].lemma == "確認する"


class Test決定性:
    def test_同じ入力で10回一致する(self, ct: CogniTag):
        results = {
            tuple(tags_of(ct, "疲れて悩んでる")) for _ in range(10)
        }
        assert len(results) == 1
