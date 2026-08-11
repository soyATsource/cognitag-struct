"""
条件形の扱いと、品詞細分類・活用形の保持の検証。

「たら」と「た」は見出し語がどちらも「た」で、活用形だけが違う。
見出し語だけで機能素を引くと、条件が過去として述語に吸収され、
「雨が降ったら」と「雨が降った」が区別できなくなる。
その回帰を防ぐのがこのファイルの目的。
"""

from __future__ import annotations

from cognitag_struct.ir import NEGATIVE, PAST, POLITE
from cognitag_struct.tokenizer import Tokenizer


def by_surface(tokens, surface):
    return next(t for t in tokens if t.surface == surface)


class Test品詞細分類:
    def test_格助詞と係助詞を区別できる(self, tokenizer: Tokenizer):
        """pos はどちらも「助詞」なので、subpos が無いと振り分けられない。

        「は」は格ではなく主題なので Clause.topic 側へ回す必要がある。
        """
        tokens = tokenizer.tokenize("明日は名古屋に行きません")

        assert by_surface(tokens, "は").pos == "助詞"
        assert by_surface(tokens, "は").subpos == "係助詞"
        assert by_surface(tokens, "に").pos == "助詞"
        assert by_surface(tokens, "に").subpos == "格助詞"

    def test_がは格助詞(self, tokenizer: Tokenizer):
        tokens = tokenizer.tokenize("犬が男を噛んだ")

        assert by_surface(tokens, "が").subpos == "格助詞"
        assert by_surface(tokens, "を").subpos == "格助詞"

    def test_接続助詞のばも区別できる(self, tokenizer: Tokenizer):
        tokens = tokenizer.tokenize("雨が降れば行く")
        assert by_surface(tokens, "ば").subpos == "接続助詞"

    def test_句由来のトークンは空文字のまま(self, tokenizer: Tokenizer):
        tokens = tokenizer.tokenize("河童の川流れ")

        assert tokens[0].is_phrase is True
        assert tokens[0].subpos == ""
        assert tokens[0].inflection == ""


class Test条件形を吸収しない:
    def test_たらは独立トークンとして残る(self, tokenizer: Tokenizer):
        tokens = tokenizer.tokenize("雨が降ったら")

        predicate = by_surface(tokens, "降っ")
        assert predicate.lemma == "降る"
        assert predicate.features == [], "条件が PAST として吸収されている"

        conditional = by_surface(tokens, "たら")
        assert conditional.pos == "助動詞"
        assert conditional.inflection.startswith("仮定形")
        assert conditional.features == []

    def test_たらとたでトークン列が変わる(self, tokenizer: Tokenizer):
        """これが区別できないと条件節を検出できない。"""
        conditional = tokenizer.tokenize("雨が降ったら")
        past = tokenizer.tokenize("雨が降った")

        assert [t.surface for t in conditional] != [t.surface for t in past]
        # 過去は従来どおり述語へ吸収される
        assert by_surface(past, "降った").features == [PAST]
        # 条件は分かれたまま
        assert [t.surface for t in conditional] == ["雨", "が", "降っ", "たら"]

    def test_ならは独立トークンとして残る(self, tokenizer: Tokenizer):
        tokens = tokenizer.tokenize("名古屋なら行きます")

        nara = by_surface(tokens, "なら")
        assert nara.pos == "助動詞"
        assert nara.inflection.startswith("仮定形")
        assert nara.features == []

    def test_ばは従来どおり独立トークン(self, tokenizer: Tokenizer):
        """接続助詞なので、そもそも吸収の対象外。仮定形の助動詞と扱いが揃う。"""
        tokens = tokenizer.tokenize("雨が降れば行く")
        assert by_surface(tokens, "ば").features == []


class Test既存の挙動が保たれている:
    def test_行きませんは変わらない(self, tokenizer: Tokenizer):
        tokens = tokenizer.tokenize("行きません")

        assert len(tokens) == 1
        assert tokens[0].lemma == "行く"
        assert sorted(tokens[0].features) == sorted([POLITE, NEGATIVE])

    def test_噛んだのトークン数が変わらない(self, tokenizer: Tokenizer):
        """回帰確認。

        なお「噛んだ」の「だ」は見出し語が「だ」であって「た」ではないため、
        修正前から PAST として吸収されておらず独立トークンのままである
        （AUXILIARY_FEATURES に「だ」を入れていない）。
        「これは本だ」の「だ」も同じ見出し語・同じ活用形なので、
        ここで PAST を付けるとコピュラを過去と誤ることになる。
        """
        tokens = tokenizer.tokenize("犬が男を噛んだ")

        assert len(tokens) == 6
        assert [t.surface for t in tokens] == ["犬", "が", "男", "を", "噛ん", "だ"]
        da = by_surface(tokens, "だ")
        assert da.pos == "助動詞"
        assert da.inflection == "終止形-一般"
        assert da.features == []

    def test_過去は従来どおり吸収される(self, tokenizer: Tokenizer):
        tokens = tokenizer.tokenize("行きました")
        assert len(tokens) == 1
        assert sorted(tokens[0].features) == sorted([POLITE, PAST])

    def test_機能素は3つのまま(self):
        from cognitag_struct.ir import FEATURES

        assert set(FEATURES) == {POLITE, NEGATIVE, PAST}
        assert "CONDITIONAL" not in FEATURES
