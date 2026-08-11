"""句照合の検証。"""

from __future__ import annotations

from cognitag_struct.phrase_dict import PhraseDict


class Test最長一致:
    def test_長い候補が優先される(self, phrase_dict: PhraseDict):
        """「東京ディズニーランド」は「東京」より長いので後者を選ぶ。

        辞書に「東京」単独は無いが、開始位置が同じ複数長を
        長い順に試すという走査順そのものを検査する。
        """
        match = phrase_dict.match_at("東京ディズニーランドに行く", 0)

        assert match is not None
        assert match.surface == "東京ディズニーランド"
        assert (match.start, match.end) == (0, 10)

    def test_途中から始まる句も拾う(self, phrase_dict: PhraseDict):
        matches = phrase_dict.find_all("今日は豚に真珠だった")

        assert [m.surface for m in matches] == ["豚に真珠"]
        assert matches[0].start == 3

    def test_照合区間は重ならない(self, phrase_dict: PhraseDict):
        matches = phrase_dict.find_all("豚に真珠と猫に小判")

        assert [m.surface for m in matches] == ["豚に真珠", "猫に小判"]
        assert matches[0].end <= matches[1].start


class Test完全一致のみ:
    def test_活用したら照合しない(self, phrase_dict: PhraseDict):
        """この段階では活用に対応しない（将来の拡張点）。"""
        assert phrase_dict.find_all("猿も木から落ちた") == []

    def test_語が挿入されたら照合しない(self, phrase_dict: PhraseDict):
        assert phrase_dict.find_all("豚にまさに真珠") == []

    def test_一致する部分だけを拾う(self, phrase_dict: PhraseDict):
        matches = phrase_dict.find_all("豚に真珠を与える")
        assert [m.surface for m in matches] == ["豚に真珠"]


class Test曖昧性:
    def test_同一表層形の候補を全件返す(self, phrase_dict: PhraseDict):
        """「犬も歩けば棒に当たる」は相反する 2 語義を持つ。

        この段階では絞り込まない。presupposition による選択は次の段階。
        """
        match = phrase_dict.match_at("犬も歩けば棒に当たる", 0)

        assert match is not None
        assert match.is_ambiguous
        assert sorted(e.id for e in match.entries) == ["inumo_1", "inumo_2"]

    def test_一意な句は曖昧でない(self, phrase_dict: PhraseDict):
        match = phrase_dict.match_at("河童の川流れ", 0)
        assert not match.is_ambiguous


class Test照合しない場合:
    def test_辞書に無ければNone(self, phrase_dict: PhraseDict):
        assert phrase_dict.match_at("こんにちは", 0) is None

    def test_空文字列は空リスト(self, phrase_dict: PhraseDict):
        assert phrase_dict.find_all("") == []

    def test_candidatesは表層形で引ける(self, phrase_dict: PhraseDict):
        assert len(phrase_dict.candidates("犬も歩けば棒に当たる")) == 2
        assert phrase_dict.candidates("存在しない句") == []
