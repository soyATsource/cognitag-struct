"""
返答に慣用句を添える経路の検証。

カテゴリから候補を取り、前提条件で絞り、1 件選んで文に足す。
選択そのものは test_select.py と idiom.py 側で検証済みなので、
ここで見るのは「どういうときに出て、どういうときに出ないか」。
"""

from __future__ import annotations

import pytest

from cognitag_struct.chat import Responder


@pytest.fixture
def bot() -> Responder:
    """会話ごとに新しくする。使用済みの記録が残ると干渉する。"""
    return Responder()


class Test話題からカテゴリを引く:
    def test_失敗の話にことわざが付く(self, bot: Responder):
        reply = bot.respond("失敗しました")
        assert "二度あることは三度ある" in reply.text

    def test_成功の話には別のカテゴリから引く(self, bot: Responder):
        reply = bot.respond("成功しました")
        assert "終わりよければすべてよし" in reply.text

    def test_根拠が追える(self, bot: Responder):
        reply = bot.respond("失敗しました")
        lines = [t for t in reply.trace if t.startswith("慣用句")]
        assert len(lines) == 1
        # どのタグからどのカテゴリを引いたかが出ること
        assert "#失敗" in lines[0]
        assert "ことわざ×失敗" in lines[0]


class Test前提条件を満たさない句は出ない:
    def test_過去でなければ出ない(self, bot: Responder):
        """「二度あることは三度ある」は既に起きたことが前提。"""
        reply = bot.respond("失敗する")
        assert "二度あることは三度ある" not in reply.text

    def test_熟練を前提とする句は出ない(self, bot: Responder):
        """相手が熟練者だと判断する手立てが無い以上、出ないのが正しい。"""
        reply = bot.respond("失敗しました")
        for surface in ("河童の川流れ", "弘法にも筆の誤り", "猿も木から落ちる"):
            assert surface not in reply.text


class Test引き口を絞ってある:
    @pytest.mark.parametrize("text", ["迷いました", "悩んでいます", "壊れました"])
    def test_困難というだけでは引かない(self, bot: Responder, text: str):
        """#困難 は迷う・悩む・壊れるまで含むので慣用句の照会には使わない。"""
        reply = bot.respond(text)
        assert "二度あることは三度ある" not in reply.text

    @pytest.mark.parametrize("text", ["終わりました", "治りました"])
    def test_達成というだけでは引かない(self, bot: Responder, text: str):
        reply = bot.respond(text)
        assert "終わりよければすべてよし" not in reply.text


class Test使いすぎない:
    def test_同じ句は二度使わない(self, bot: Responder):
        first = bot.respond("失敗しました")
        assert "二度あることは三度ある" in first.text

        bot.respond("そうですか")   # 間に 1 ターン挟む
        third = bot.respond("また失敗しました")
        assert "二度あることは三度ある" not in third.text

    def test_二連続では使わない(self, bot: Responder):
        bot.respond("失敗しました")
        second = bot.respond("成功しました")
        assert "終わりよければすべてよし" not in second.text

    def test_会話が変われば使える(self):
        """使用済みの記録は会話ごと。別の相手には同じ句を使ってよい。"""
        assert "二度あることは三度ある" in Responder().respond("失敗しました").text
        assert "二度あることは三度ある" in Responder().respond("失敗しました").text


class Test決定的である:
    def test_同じ入力に同じ返答(self):
        first = Responder().respond("失敗しました").text
        for _ in range(3):
            assert Responder().respond("失敗しました").text == first


class Test言い回しが繰り返さない:
    """同じ型を続けて使ったときに、同じ文を返し続けないこと。

    乱数は使わない。「その型を何回目に使ったか」で選ぶので、
    会話の中では変わり、会話をやり直せば同じ結果になる。
    """

    def test_同じ入力でも二度目は言い方が変わる(self, bot: Responder):
        first = bot.respond("うるさい").text
        second = bot.respond("うるさい").text
        assert first != second

    def test_会話をやり直せば同じ結果になる(self):
        def 会話() -> list[str]:
            bot = Responder()
            return [bot.respond("うるさい").text for _ in range(4)]

        assert 会話() == 会話()

    def test_相手の語を織り込む(self, bot: Responder):
        """「かわいいね」の「かわいい」を返答に拾えること。"""
        texts = [bot.respond("かわいいね").text for _ in range(3)]
        assert any("かわいい" in t for t in texts)

    def test_言い回しは一巡して戻る(self, bot: Responder):
        """有限の一覧を順に使う。尽きたら先頭へ戻る。"""
        texts = [bot.respond("なるほど").text for _ in range(9)]
        assert texts[0] == texts[4]
        assert len(set(texts)) > 1
