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
        """差し込まないかぎり、外の話には答えない。"""
        reply = Responder().respond("今何時ですか")
        assert "答えられない" in reply.text

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
        assert "答えられない" in reply.text
        assert fake.asked == []

    def test_答えられなければ通常の経路(self):
        """外が落ちている場合。例外ではなく通常の分岐として扱う。"""
        fake = 偽の能力(answer=None)
        reply = Responder(capabilities=[fake]).respond("今何時ですか")
        assert "答えられない" in reply.text
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
