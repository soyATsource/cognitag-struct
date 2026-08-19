"""
外向きの窓口の検証。

利用者（AIAssistant / LOGOS）が触るのはここだけになるので、
「パスを渡さなくても動く」「解釈不能でも例外を出さない」を重点的に見る。
"""

from __future__ import annotations

import pytest

from cognitag_struct import Analysis, CogniTag, Modality
from cognitag_struct.ir import IR
from cognitag_struct.parse import Unparsable


@pytest.fixture(scope="module")
def ct() -> CogniTag:
    """同梱データだけで組み立てられること。パスを渡さない。"""
    return CogniTag()


class Test引数なしで使える:
    def test_パスを渡さず初期化できる(self, ct: CogniTag):
        # 句の件数は直書きしない（カテゴリ辞書は育てる前提のデータ）
        assert ct.phrase_count > 0
        assert ct.analyzer_available is True

    def test_同梱データの場所を持っている(self, ct: CogniTag):
        assert (ct.data_dir / "axes.jsonl").exists()
        assert (ct.data_dir / "generation_style.toml").exists()

    def test_データを差し替えられる(self, tmp_path, ct: CogniTag):
        """テストや語彙入れ替えのために data_dir を上書きできる。"""
        import shutil

        for name in (
            "axes.jsonl", "phrases.jsonl", "frames.jsonl",
            "questions.toml", "generation_style.toml", "reasoning.toml",
            "conjugation.toml", "patterns.jsonl",
        ):
            shutil.copy(ct.data_dir / name, tmp_path / name)

        other = CogniTag(data_dir=tmp_path)
        assert other.data_dir == tmp_path
        assert other.phrase_count == ct.phrase_count

    def test_起動ログ用の要約が出る(self, ct: CogniTag):
        note = ct.describe()
        assert f"句{ct.phrase_count}件" in note
        assert "Sudachi=有" in note


class Test一度に解析できる:
    def test_解析結果が揃う(self, ct: CogniTag):
        result = ct.analyze("明日は名古屋に行きません")

        assert isinstance(result, Analysis)
        assert result.text == "明日は名古屋に行きません"
        assert result.tokens
        assert result.modality.modality is Modality.STATEMENT
        assert result.parsed
        assert result.ir.clauses[0].slots["NI"].surface == "名古屋"
        assert result.questions() == []

    def test_空スロットが質問になる(self, ct: CogniTag):
        result = ct.analyze("行きます")
        assert result.questions() == ["どこに？"]

    def test_一行要約が読める(self, ct: CogniTag):
        note = ct.analyze("行きます").coverage_note()
        assert "STATEMENT" in note
        assert "解析済" in note


class Test解釈不能でも壊れない:
    def test_3節以上でも例外を出さない(self, ct: CogniTag):
        result = ct.analyze("雨が降ったら友人に会いに名古屋へ行きますか")

        assert not result.parsed
        assert isinstance(result.ir, Unparsable)
        # 構文解析に失敗してもモダリティは必ず入る
        assert result.modality.modality is Modality.Q_YESNO
        assert result.questions() == []

    def test_述語が無くても例外を出さない(self, ct: CogniTag):
        result = ct.analyze("名古屋")
        assert not result.parsed
        assert result.modality is not None

    def test_空文字でも例外を出さない(self, ct: CogniTag):
        result = ct.analyze("")
        assert not result.parsed
        assert result.modality.modality is Modality.STATEMENT

    def test_辞書に無い語だけでも動く(self, ct: CogniTag):
        result = ct.analyze("にゃんにゃんしますか")
        assert result.modality.modality is Modality.Q_YESNO


class Testルーティングに使える:
    def test_知識が要る問いを見分ける(self, ct: CogniTag):
        assert ct.analyze("どこに行きますか").needs_knowledge is True
        assert ct.analyze("名古屋に行きますか").needs_knowledge is False
        assert ct.analyze("作ってください").needs_knowledge is False

    def test_モダリティだけ取れる(self, ct: CogniTag):
        """構文解析を走らせない近道。"""
        assert ct.modality_of("作ってください").modality is Modality.REQUEST


class Test生成:
    def test_解析して言い直せる(self, ct: CogniTag):
        assert ct.say("名古屋に行きます", verbosity=0) == "名古屋に行きます"

    def test_解釈不能ならNone(self, ct: CogniTag):
        assert ct.say("雨が降ったら友人に会いに名古屋へ行きます") is None

    def test_IRから直接生成できる(self, ct: CogniTag):
        result = ct.analyze("名古屋に行きます")
        assert "名古屋" in ct.generate(result.ir, verbosity=3).text


class Test慣用句:
    def test_前提条件を満たせば選ばれる(self, ct: CogniTag):
        chosen = ct.idiom("ことわざ", "失敗", ["#熟練", "#常時成功"])
        assert chosen is not None

    def test_満たさなければNone(self, ct: CogniTag):
        assert ct.idiom("ことわざ", "失敗", []) is None


class Test決定性:
    def test_同じ入力で10回一致する(self, ct: CogniTag):
        texts = {
            ct.analyze("明日は名古屋に行きません").coverage_note()
            for _ in range(10)
        }
        assert len(texts) == 1
