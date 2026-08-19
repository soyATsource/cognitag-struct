"""
トークナイズの検証。

句照合を先に行うこと、助動詞を用言へ吸収すること、
可能形を別見出しとして扱うことの 3 点が中心。
"""

from __future__ import annotations

from cognitag_struct.ir import NEGATIVE, PAST, POLITE
from cognitag_struct.tokenizer import Tokenizer


def surfaces(tokens) -> list[str]:
    return [t.surface for t in tokens]


def lemmas(tokens) -> list[str]:
    return [t.lemma for t in tokens]


class Test句の照合:
    def test_河童の川流れは1トークン(self, tokenizer: Tokenizer):
        """Sudachi 単独だと 河童 / の / 川流れ に割れる。

        句照合を先に行うからこそ 1 トークンで取れる。
        """
        tokens = tokenizer.tokenize("河童の川流れ")

        assert len(tokens) == 1
        token = tokens[0]
        assert token.surface == "河童の川流れ"
        assert token.is_phrase is True
        assert token.entry_id == "kappa_1"
        assert token.form == "ことわざ"
        assert token.content == ["失敗"]
        assert token.span == (0, 6)

    def test_河童が川を流れるは句にならない(self, tokenizer: Tokenizer):
        """語が同じでも並びが違えば句ではない。通常分割される。"""
        tokens = tokenizer.tokenize("河童が川を流れる")

        assert all(t.is_phrase is False for t in tokens)
        assert len(tokens) > 1
        assert "河童" in surfaces(tokens)
        assert "が" in surfaces(tokens)

    def test_句と形態素が混在する(self, tokenizer: Tokenizer):
        tokens = tokenizer.tokenize("豚に真珠だと言われた")

        assert tokens[0].surface == "豚に真珠"
        assert tokens[0].is_phrase is True
        assert tokens[0].form == "ことわざ"

        assert "言う" in [t.lemma for t in tokens]

    def test_1形態素の語は句照合に載せない(self, tokenizer: Tokenizer):
        """「東京ディズニーランド」は Sudachi が 1 形態素で返す。

        句照合に載せると「会社」が「会社員」の途中で切れるのと同じ事故が
        起きるので、載せない。カテゴリの情報は形態素分割の後に付ける。
        """
        tokens = tokenizer.tokenize("東京ディズニーランドに行きます")

        assert tokens[0].surface == "東京ディズニーランド"
        assert tokens[0].is_phrase is False      # 句としては取っていない
        assert tokens[0].form == "固有名詞"       # カテゴリの情報は付いている
        assert tokens[0].entry_id == "tdl_1"

        assert tokens[1].surface == "に"
        assert tokens[1].pos == "助詞"

        assert tokens[2].lemma == "行く"
        assert tokens[2].has(POLITE)

    def test_句のspanは元文字列上の位置(self, tokenizer: Tokenizer):
        text = "今日は豚に真珠だった"
        tokens = tokenizer.tokenize(text)

        phrase = next(t for t in tokens if t.is_phrase)
        assert text[phrase.span[0] : phrase.span[1]] == "豚に真珠"

    def test_句に無い語の既定値(self, tokenizer: Tokenizer):
        tokens = tokenizer.tokenize("犬が走る")

        for token in tokens:
            assert token.form == "一般語"
            assert token.content == []
            assert token.entry_id is None


class Test機能素:
    def test_行きませんは行くでPOLITEとNEGATIVE(self, tokenizer: Tokenizer):
        tokens = tokenizer.tokenize("行きません")

        assert len(tokens) == 1
        token = tokens[0]
        assert token.lemma == "行く"
        assert sorted(token.features) == sorted([POLITE, NEGATIVE])

    def test_行けませんは行けるのまま(self, tokenizer: Tokenizer):
        """可能形は意味を変えるので「行く」に寄せない。

        Facet を別に持つ必要があるため、別見出しとして扱う。
        """
        tokens = tokenizer.tokenize("行けません")

        assert len(tokens) == 1
        assert tokens[0].lemma == "行ける"
        assert sorted(tokens[0].features) == sorted([POLITE, NEGATIVE])

    def test_行かないはNEGATIVEのみ(self, tokenizer: Tokenizer):
        tokens = tokenizer.tokenize("行かない")

        assert tokens[0].lemma == "行く"
        assert tokens[0].features == [NEGATIVE]

    def test_行きましたはPOLITEとPAST(self, tokenizer: Tokenizer):
        tokens = tokenizer.tokenize("行きました")

        assert tokens[0].lemma == "行く"
        assert sorted(tokens[0].features) == sorted([POLITE, PAST])

    def test_吸収したspanは全体を指す(self, tokenizer: Tokenizer):
        text = "行きません"
        tokens = tokenizer.tokenize(text)

        assert tokens[0].span == (0, len(text))
        assert text[tokens[0].span[0] : tokens[0].span[1]] == text

    def test_助詞はトークンとして残る(self, tokenizer: Tokenizer):
        """格解析に必要なので助動詞と違い吸収しない。"""
        tokens = tokenizer.tokenize("犬が棒に当たる")

        particles = [t.surface for t in tokens if t.pos == "助詞"]
        assert "が" in particles
        assert "に" in particles


class Test境界:
    def test_空文字列(self, tokenizer: Tokenizer):
        assert tokenizer.tokenize("") == []

    def test_spanが入力を覆う(self, tokenizer: Tokenizer):
        text = "東京ディズニーランドに行きます"
        tokens = tokenizer.tokenize(text)

        assert tokens[0].span[0] == 0
        assert tokens[-1].span[1] == len(text)
        # 隣り合うトークンの間に隙間が無い
        for previous, current in zip(tokens, tokens[1:]):
            assert previous.span[1] == current.span[0]

    def test_解析器が使える(self, tokenizer: Tokenizer):
        assert tokenizer.available, "SudachiPy が必要"


class Testタグの導出:
    def test_句トークンからタグを導出できる(self, tokenizer: Tokenizer):
        tokens = tokenizer.tokenize("河童の川流れ")
        assert tokens[0].tags() == ["#ことわざ", "#失敗"]
