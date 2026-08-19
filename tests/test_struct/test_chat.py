"""
対話としての振る舞いの検証。

単発の解析ではなく、ターンをまたいで成り立つかを見る。
「どこに？」と尋ねた次に「名古屋」と言われて受け取れるかが本題。
"""

from __future__ import annotations

import pytest

from cognitag_struct.chat import Responder
from cognitag_struct.modality import Modality


@pytest.fixture
def bot() -> Responder:
    """会話ごとに新しい文脈を持たせる。文脈が残ると試験が干渉する。"""
    return Responder()


class Test会話が繋がる:
    def test_質問の答えを単語で受け取れる(self, bot: Responder):
        first = bot.respond("行きます")
        assert "どこに" in first.text

        second = bot.respond("名古屋")

        assert second.policy.startswith("answered")
        assert "名古屋" in second.text
        assert "取れなかった" not in second.text

    def test_埋め戻した構造を言い直す(self, bot: Responder):
        bot.respond("行きます")
        reply = bot.respond("名古屋")
        assert "名古屋に行きます" in reply.text

    def test_使った質問は消える(self, bot: Responder):
        """残すと無関係な単語まで前の質問への答えにしてしまう。"""
        bot.respond("行きます")
        bot.respond("名古屋")

        assert bot.context.pending is None
        # 次の単語は答えとして扱われない
        third = bot.respond("東京")
        assert not third.policy.startswith("answered")

    def test_述語があれば独立した発話として扱う(self, bot: Responder):
        """「名古屋」は答えだが「名古屋は遠い」は新しい話。"""
        bot.respond("行きます")
        reply = bot.respond("名古屋が好き")
        assert not reply.policy.startswith("answered")

    def test_尋ねていなければ答え扱いしない(self, bot: Responder):
        reply = bot.respond("名古屋")
        assert not reply.policy.startswith("answered")


class Test指示語:
    def test_直近の話題に置き換える(self, bot: Responder):
        bot.respond("名古屋に行きます")
        reply = bot.respond("それはどう")

        assert reply.policy == "resolved"
        assert "名古屋" in reply.text

    def test_話題が無ければ解決しない(self, bot: Responder):
        reply = bot.respond("それはどう")
        assert reply.policy != "resolved"

    def test_指示語そのものは話題にしない(self, bot: Responder):
        bot.respond("名古屋に行きます")
        bot.respond("それはどう")
        # 「それ」で話題が上書きされていないこと
        assert bot.context.latest_topic.lemma != "それ"


class Test挨拶と断片:
    @pytest.mark.parametrize("text", ["こんばんは", "おはよう", "こんにちは"])
    def test_挨拶に返せる(self, bot: Responder, text: str):
        reply = bot.respond(text)
        assert reply.policy == "greeting"
        assert "取れなかった" not in reply.text

    def test_疑問詞だけの断片も問いとして扱う(self, bot: Responder):
        reply = bot.respond("なぜ")
        assert reply.policy == "q_open"

    def test_空入力(self, bot: Responder):
        assert bot.respond("").policy == "empty"

    def test_空白だけ(self, bot: Responder):
        assert bot.respond("   ").policy == "empty"


class Test快と不快を分ける:
    def test_つらさには共感で返す(self, bot: Responder):
        """説明ではなく受け止めの一句を返す。

        以前は含意（「正論より先に、そう感じたことを認めたい」）を
        返していたが、望まれていたのは短い受け止めだった。
        含意は :trace に残る。
        """
        reply = bot.respond("つらい")
        # 受け止めの一句で始まること。後ろに問い返しが付くかは
        # 述語によって変わるので、先頭だけを見る。
        assert reply.text.startswith("大丈夫？")
        assert any("認めたい" in line for line in reply.trace)
        assert "喜んで" not in reply.text

    def test_喜びには喜びで返す(self, bot: Responder):
        reply = bot.respond("嬉しい")
        assert reply.text.startswith("それはいいね。")
        assert any("喜んでいい" in line for line in reply.trace)

    @pytest.mark.parametrize("text", ["苦しい", "寂しい", "怖い", "落ち込む"])
    def test_不快側の語を拾う(self, bot: Responder, text: str):
        assert "#つらさ" in bot.ct.analyze(text).reasoning.tags

    @pytest.mark.parametrize("text", ["面白い", "安心する", "助かる"])
    def test_快側の語を拾う(self, bot: Responder, text: str):
        assert "#喜び" in bot.ct.analyze(text).reasoning.tags


class Test重複を出さない:
    def test_粗いタグの含意は具体的な方に譲る(self, bot: Responder):
        """「気持ちの話」と「しんどい話」を両方言わない。"""
        reply = bot.respond("つらい")
        assert "気持ちの話" not in reply.text

    def test_タグ自体は両方残る(self, bot: Responder):
        """含意は落とすが分類の記録は消さない。"""
        tags = bot.ct.analyze("つらい").reasoning.tags
        assert "#感情" in tags
        assert "#つらさ" in tags

    def test_モダリティで言い切ったことを繰り返さない(self, bot: Responder):
        """応答の型で言ったことを、含意で言い直さない。

        含意そのものを返答に出さなくなったので、重複は起こりえない。
        代わりに「返答に説明が混ざっていないこと」を見る。
        """
        reply = bot.respond("どうしよう")
        assert "これからの話" not in reply.text
        assert "決めるのは本人だ" not in reply.text


class Test補助動詞:
    def test_てくださいは1節(self, bot: Responder):
        ir = bot.ct.analyze("手伝ってください").ir
        assert len(ir.clauses) == 1
        assert ir.clauses[0].predicate.lemma == "手伝う"

    def test_依頼として判定される(self, bot: Responder):
        assert bot.ct.analyze("手伝ってください").modality.modality is Modality.REQUEST

    def test_目的の節分割は保たれる(self, bot: Responder):
        """「会いに行く」は 2 節のまま。「に」は「て」ではない。"""
        ir = bot.ct.analyze("友人に会いに名古屋へ行きます").ir
        assert len(ir.clauses) == 2


class Test名詞のタグ:
    def test_地名を場所として拾う(self, bot: Responder):
        assert "#場所" in bot.ct.analyze("名古屋に行きます").reasoning.tags

    def test_時を表す名詞を拾う(self, bot: Responder):
        assert "#時間" in bot.ct.analyze("明日行きます").reasoning.tags


class Test決定性:
    def test_同じやり取りを10回繰り返して一致する(self):
        def run() -> list[str]:
            bot = Responder()
            return [
                bot.respond("行きます").text,
                bot.respond("名古屋").text,
                bot.respond("つらい").text,
            ]

        first = run()
        for _ in range(9):
            assert run() == first


class Test例外を出さない:
    @pytest.mark.parametrize(
        "text",
        [
            "。。。", "???", "aaa", "1234", "😀",
            "雨が降ったら友人に会いに名古屋へ行きますか",
            "あ" * 200,
        ],
    )
    def test_どんな入力でも返答を返す(self, bot: Responder, text: str):
        reply = bot.respond(text)
        assert isinstance(reply.text, str)
        assert reply.text
