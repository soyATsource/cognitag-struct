"""
外の世界を触る差し込み口の検証。

【本物のネットワークは叩かない】
天気 API を試験で呼ぶと、繋がるかどうかで結果が変わる。差し込み口が
正しく働くかと、外が正しく答えるかは別の問題なので、ここでは
偽の能力を差し込んで前者だけを見る。

時計だけは固定した時刻を渡せるので、実物の Clock で検証する。
"""

from __future__ import annotations

from datetime import datetime

import pytest

from cognitag_struct.capability import Registry, Request, Result
from cognitag_struct.chat import Responder
from cognitag_struct.providers.clock import Clock


class 偽の能力:
    """呼ばれたことを記録するだけの能力。"""

    name = "fake"

    def __init__(self, answer: str | None = "偽の答え", claims: bool = True):
        self.answer = answer
        self.claims = claims
        self.asked: list[str] = []

    def wants(self, analysis):
        if not self.claims:
            return None
        return Request(kind="fake", text=analysis.text)

    def handle(self, request):
        self.asked.append(request.text)
        if self.answer is None:
            return None
        return Result(text=self.answer, source="偽の出どころ", detail="detail")


class Test既定では何も繋がっていない:
    def test_能力を渡さなければ空(self):
        assert not Responder().capabilities

    def test_答えられないままである(self):
        """差し込まないかぎり、外の話には答えない。

        文言ではなく方針で見る。言い回しは調整で変わるが、
        「答えられない経路に入ったこと」は変わらない。
        """
        reply = Responder().respond("今何時ですか")
        assert reply.policy == "q_open"

    def test_決定性が保たれる(self):
        first = Responder().respond("今何時ですか").text
        assert Responder().respond("今何時ですか").text == first


class Test差し込むと呼ばれる:
    def test_返答が能力のものに変わる(self):
        fake = 偽の能力()
        reply = Responder(capabilities=[fake]).respond("今何時ですか")

        assert reply.text == "偽の答え"
        assert reply.policy == "external:fake"
        assert fake.asked == ["今何時ですか"]

    def test_出どころが根拠に残る(self):
        """外から来た値を無印で返さない。"""
        reply = Responder(capabilities=[偽の能力()]).respond("今何時ですか")
        lines = [t for t in reply.trace if t.startswith("外部")]
        assert len(lines) == 1
        assert "偽の出どころ" in lines[0]

    def test_担当しなければ通常の経路(self):
        fake = 偽の能力(claims=False)
        reply = Responder(capabilities=[fake]).respond("今何時ですか")
        assert reply.policy == "q_open"
        assert fake.asked == []

    def test_答えられなければ通常の経路(self):
        """外が落ちている場合。例外ではなく通常の分岐として扱う。"""
        fake = 偽の能力(answer=None)
        reply = Responder(capabilities=[fake]).respond("今何時ですか")
        assert reply.policy == "q_open"
        assert fake.asked == ["今何時ですか"]

    def test_先に登録した方が優先する(self):
        first, second = 偽の能力("先"), 偽の能力("後")
        reply = Responder(capabilities=[first, second]).respond("今何時ですか")
        assert reply.text == "先"
        assert second.asked == []


class Test時計:
    @pytest.fixture
    def bot(self) -> Responder:
        固定 = datetime(2026, 8, 12, 9, 5)
        return Responder(capabilities=[Clock(now=lambda: 固定)])

    def test_時刻を答える(self, bot: Responder):
        assert bot.respond("今何時ですか").text == "9 時 5 分だ。"

    def test_日付を答える(self, bot: Responder):
        assert "2026 年 8 月 12 日" in bot.respond("今日は何日ですか").text

    def test_問いでなければ答えない(self, bot: Responder):
        """「時間がない」は時刻を聞かれてはいない。"""
        reply = bot.respond("時間がない")
        assert not reply.policy.startswith("external")

    def test_無関係な問いには出てこない(self, bot: Responder):
        reply = bot.respond("なぜ空が青いのか")
        assert not reply.policy.startswith("external")


class Test登録簿:
    def test_名前が並ぶ(self):
        registry = Registry([偽の能力()])
        assert registry.names() == ["fake"]

    def test_空なら偽(self):
        assert not Registry()


class Test自分について答える:
    """世界のことは知らないが、自分が何であるかは知っている。

    知識を持たないことと、自己記述を持たないことは別である。
    「君は誰」に「答えられない」と返していたのは、答えを持ちながら
    参照していなかっただけだった。
    """

    def test_名前を答える(self):
        reply = Responder().respond("君は誰")
        assert "CogniTag" in reply.text
        assert reply.policy == "self:identity"

    def test_疑問符が無くても問いとして扱う(self):
        """「君は誰」には終助詞も疑問符も無いが、問いである。"""
        assert Responder().respond("君は誰").policy == "self:identity"

    def test_できることを答える(self):
        reply = Responder().respond("何ができるの")
        assert reply.policy == "self:ability"
        # できないことも一緒に言う。並べるだけだと期待がずれる。
        assert "できない" in reply.text

    def test_状態を聞かれたら性質を答える(self):
        reply = Responder().respond("疲れてない？")
        assert reply.text == "私は疲れないよ。"

    def test_具体的な性質が優先される(self):
        """「疲れてない？」は #つらさ と #疲労 の両方が立つ。

        どちらを答えるかは self.toml の並び順で決まる。
        """
        assert "疲れない" in Responder().respond("疲れてない？").text

    def test_自分以外の問いには出てこない(self):
        reply = Responder().respond("なぜ空が青いのか")
        assert not reply.policy.startswith("self:")


class Test記憶:
    """発話そのものを覚えておく。

    構造（IR）は意味の骨格であって、言われた文ではない。
    生成し直すと言い直しになり、「さっきこう言った」の証拠にならない。
    """

    def test_直前の発話を返せる(self):
        bot = Responder()
        first = bot.respond("疲れた").text
        recalled = bot.respond("さっき何て言った")

        assert first in recalled.text
        assert recalled.policy == "recall"

    def test_何も言っていなければそう言う(self):
        """作り直さない。覚えていないことは覚えていないと言う。"""
        reply = Responder().respond("さっき何て言った")
        assert reply.policy == "recall_empty"

    def test_会話ごとに独立している(self):
        bot = Responder()
        bot.respond("疲れた")
        assert Responder().respond("さっき何て言った").policy == "recall_empty"

    def test_古い発話は捨てる(self):
        """上限を超えたら古いものから消える。全履歴は持たない。"""
        from cognitag_struct.context import TURN_LIMIT

        bot = Responder()
        for _ in range(TURN_LIMIT * 2):
            bot.respond("疲れた")
        assert len(bot.context.turns) <= TURN_LIMIT


class Test事実:
    """百科事典ではなく、答えを用意したものだけ。

    載っていないことは今までどおり分からないと返す。
    境界が動いていないことを確かめるのが、このクラスの主目的。
    """

    def test_用意した事実には答える(self):
        reply = Responder().respond("富士山の高さは")
        assert "3776" in reply.text
        assert reply.policy.startswith("fact:")

    def test_出どころが根拠に残る(self):
        reply = Responder().respond("富士山の高さは")
        assert any("事実" in line for line in reply.trace)

    def test_用意していない事実には答えない(self):
        """「海が青い理由」は表に無い。境界は動かさない。"""
        reply = Responder().respond("なぜ海は青いの")
        assert not reply.policy.startswith("fact:")

    def test_問いでなければ答えない(self):
        """「富士山に行きたい」は高さを尋ねていない。"""
        reply = Responder().respond("富士山に行きたい")
        assert not reply.policy.startswith("fact:")


class Test問いの形:
    """終助詞も疑問符も無い問いを拾えること。"""

    def test_はで言い終わる問い(self):
        assert Responder().respond("日本の首都は").policy.startswith("fact:")

    def test_のと疑問詞の組(self):
        assert Responder().respond("なぜ空は青いの").policy.startswith("fact:")

    def test_平叙のままのもの(self):
        """「これは本です」は問いではない。"""
        reply = Responder().respond("これは本です")
        assert not reply.policy.startswith("fact:")
        assert not reply.policy.startswith("q_")
