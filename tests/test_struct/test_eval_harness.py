"""
評価セットが読める状態に保たれているかの検証。

【点数はテストしない】
「126/129 通ること」を assert すると、実装を改善するたびにテストが落ちる。
点数は baseline.json と `--compare` で追うものであって、回帰テストの
対象ではない。ここで守るのはケースファイルの記述の正しさだけ。

【なぜ記述を検証するか】
期待値のキーを綴り間違えると、その観点は黙って検証されなくなる。
「検証しているつもりで検証されていない」状態は、測定の土台としては
点数が下がることより悪い。読み込み時に失敗させ、ここで見張る。
"""

from __future__ import annotations

import pytest

from cognitag_struct.eval.run_eval import (
    CASE_KEYS,
    CHECK_KEYS,
    CaseError,
    load_cases,
)


class Test評価セットの記述:
    def test_同梱のケースが読める(self):
        cases = load_cases()
        assert len(cases) > 100

    def test_idが一意(self):
        ids = [case.id for case in load_cases()]
        assert len(ids) == len(set(ids))

    def test_全ケースに期待値がある(self):
        for case in load_cases():
            assert case.expect, f"{case.id} の expect が空"
            assert case.turns, f"{case.id} の turns が空"

    def test_既知の弱点には理由が書いてある(self):
        """known_gap は「なぜ今落ちるか」が分からないと判断できない。"""
        for case in load_cases():
            if case.known_gap:
                assert case.note, f"{case.id} に note が無い"


class Test記述の誤りを弾く:
    def _write(self, tmp_path, line: str):
        path = tmp_path / "cases.jsonl"
        path.write_text(line + "\n", encoding="utf-8")
        return path

    def test_未知の期待値キーで失敗する(self, tmp_path):
        path = self._write(
            tmp_path, '{"id": "x", "turns": ["行く"], "expect": {"modarity": "STATEMENT"}}'
        )
        with pytest.raises(CaseError, match="未知の期待値"):
            load_cases(path)

    def test_未知のケースキーで失敗する(self, tmp_path):
        path = self._write(
            tmp_path, '{"id": "x", "turns": ["行く"], "expect": {"parsed": true}, "kown_gap": true}'
        )
        with pytest.raises(CaseError, match="未知のキー"):
            load_cases(path)

    def test_存在しないモダリティで失敗する(self, tmp_path):
        path = self._write(
            tmp_path, '{"id": "x", "turns": ["行く"], "expect": {"modality": "QUESTION"}}'
        )
        with pytest.raises(CaseError, match="存在しないモダリティ"):
            load_cases(path)

    def test_idの重複で失敗する(self, tmp_path):
        path = tmp_path / "cases.jsonl"
        line = '{"id": "x", "turns": ["行く"], "expect": {"parsed": true}}'
        path.write_text(line + "\n" + line + "\n", encoding="utf-8")
        with pytest.raises(CaseError, match="id が重複"):
            load_cases(path)

    def test_期待値が空だと失敗する(self, tmp_path):
        """何も検証しないケースは、通ったように見えて何も測っていない。"""
        path = self._write(tmp_path, '{"id": "x", "turns": ["行く"], "expect": {}}')
        with pytest.raises(CaseError, match="expect が空"):
            load_cases(path)

    def test_壊れたJSONで失敗する(self, tmp_path):
        path = self._write(tmp_path, '{"id": "x", "turns": ["行く"')
        with pytest.raises(CaseError, match="JSON が壊れている"):
            load_cases(path)


class Test採点の観点:
    def test_期待値キーとケースキーが重ならない(self):
        """expect の中と外で同名のキーがあると、書き間違えても気づけない。"""
        assert not (CHECK_KEYS & CASE_KEYS)
