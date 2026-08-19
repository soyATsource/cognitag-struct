"""
2 軸交差の検証。

このパッケージで最も重要な性質は「引数の順序が結果に影響しない」こと。
木構造を採らなかった理由がここに集約されるので、重点的に検査する。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cognitag_struct.category_dict import CategoryDict, CategoryDictError

from .conftest import write_phrases

ENTRY = {
    "id": "x1",
    "surface": "テスト",
    "lemma": "テスト",
    "pos": "名詞",
    "form": "ことわざ",
    "content": ["失敗"],
    "presupposition": [],
    "facet": {"physical": 0.0, "psychological": 0.0, "temporal": 0.0, "logical": 0.0},
    "note": "",
}


def line(**overrides) -> str:
    return json.dumps({**ENTRY, **overrides}, ensure_ascii=False)


class Test軸の交差:
    def test_引数の順序が結果を変えない(self, category_dict: CategoryDict):
        """積は可換。木構造にしなかった理由そのもの。"""
        a = category_dict.members(form="ことわざ", content="失敗")
        b = category_dict.members(content="失敗", form="ことわざ")

        assert [e.id for e in a] == [e.id for e in b]
        assert a == b

    def test_ことわざ掛ける無駄は3件(self, category_dict: CategoryDict):
        """意味が微妙に異なっても同一カテゴリに共存できること。

        豚に真珠 / 猫に小判 / 馬の耳に念仏 は厳密には意味がずれるが、
        カテゴリ層の役割は候補を数件に絞るところまでなので同居させる。
        """
        found = category_dict.members(form="ことわざ", content="無駄")

        assert [e.surface for e in found] == ["豚に真珠", "猫に小判", "馬の耳に念仏"]

    def test_ことわざ掛ける失敗(self, category_dict: CategoryDict):
        found = category_dict.members(form="ことわざ", content="失敗")
        surfaces = [e.surface for e in found]

        assert "河童の川流れ" in surfaces
        assert "弘法にも筆の誤り" in surfaces
        assert "猿も木から落ちる" in surfaces

    def test_片方だけの指定(self, category_dict: CategoryDict):
        by_form = category_dict.members(form="固有名詞")
        assert len(by_form) == 5
        assert all(e.form == "固有名詞" for e in by_form)

        by_content = category_dict.members(content="幸運")
        assert "inumo_2" in [e.id for e in by_content]
        assert all(e.content == ["幸運"] for e in by_content)

    def test_両方省略すると全件(self, category_dict: CategoryDict):
        # 件数は直書きしない。カテゴリ辞書は育てる前提のデータなので、
        # 語を足すたびにテストが落ちるのは検証として役に立たない。
        assert len(category_dict.members()) == len(category_dict)
        assert len(category_dict) > 0

    def test_該当なしは空リスト(self, category_dict: CategoryDict):
        # 固有名詞に content を付けていないので交差は空になる
        assert category_dict.members(form="固有名詞", content="失敗") == []

    def test_戻り順は登録順で安定する(self, category_dict: CategoryDict):
        """集合の反復順に任せると実行ごとに変わりうる。"""
        first = [e.id for e in category_dict.members(form="ことわざ")]
        for _ in range(5):
            assert [e.id for e in category_dict.members(form="ことわざ")] == first


class Test相反する語義:
    def test_contentの指定で別idが取れる(self, category_dict: CategoryDict):
        """「犬も歩けば棒に当たる」は本来の意味と転じた意味が相反する。

        同時に成立しないので content 配列にまとめず id を分けてある。
        """
        failure = category_dict.members(content="失敗")
        fortune = category_dict.members(content="幸運")

        inumo_failure = [e for e in failure if e.surface == "犬も歩けば棒に当たる"]
        inumo_fortune = [e for e in fortune if e.surface == "犬も歩けば棒に当たる"]

        assert [e.id for e in inumo_failure] == ["inumo_1"]
        assert [e.id for e in inumo_fortune] == ["inumo_2"]

    def test_前提条件も分かれている(self, category_dict: CategoryDict):
        assert category_dict.get("inumo_1").presupposition == ["#出過ぎた行動"]
        assert category_dict.get("inumo_2").presupposition == ["#行動した"]

    def test_表層形は同じ(self, category_dict: CategoryDict):
        assert (
            category_dict.get("inumo_1").surface
            == category_dict.get("inumo_2").surface
        )


class Test読み込み時の検証:
    def test_3本目の軸を拒否する(self, tmp_path: Path):
        axes = tmp_path / "axes.jsonl"
        axes.write_text(
            '{"axis": "form", "values": ["ことわざ"]}\n'
            '{"axis": "content", "values": ["失敗"]}\n'
            '{"axis": "register", "values": ["文語", "口語"]}\n',
            encoding="utf-8",
        )
        phrases = write_phrases(tmp_path / "p.jsonl", line())

        with pytest.raises(CategoryDictError, match="3 本目"):
            CategoryDict.load(axes, phrases)

    def test_未定義のformを拒否する(self, valid_axes: Path, tmp_path: Path):
        phrases = write_phrases(tmp_path / "p.jsonl", line(form="俳句"))

        with pytest.raises(CategoryDictError, match="未定義の form"):
            CategoryDict.load(valid_axes, phrases)

    def test_未定義のcontentを拒否する(self, valid_axes: Path, tmp_path: Path):
        phrases = write_phrases(tmp_path / "p.jsonl", line(content=["郷愁"]))

        with pytest.raises(CategoryDictError, match="未定義の content"):
            CategoryDict.load(valid_axes, phrases)

    def test_idの重複を拒否する(self, valid_axes: Path, tmp_path: Path):
        phrases = write_phrases(
            tmp_path / "p.jsonl", line(), line(surface="別の句")
        )

        with pytest.raises(CategoryDictError, match="重複"):
            CategoryDict.load(valid_axes, phrases)

    def test_軸が足りなければ拒否する(self, tmp_path: Path):
        axes = tmp_path / "axes.jsonl"
        axes.write_text('{"axis": "form", "values": ["ことわざ"]}\n', encoding="utf-8")
        phrases = write_phrases(tmp_path / "p.jsonl", line())

        with pytest.raises(CategoryDictError, match="足りない"):
            CategoryDict.load(axes, phrases)


class Testその他の検索:
    def test_getでidから引ける(self, category_dict: CategoryDict):
        entry = category_dict.get("kappa_1")
        assert entry.surface == "河童の川流れ"
        assert entry.presupposition == ["#熟練", "#常時成功"]

    def test_存在しないidはNone(self, category_dict: CategoryDict):
        assert category_dict.get("no_such_id") is None

    def test_axis_valuesが定義を返す(self, category_dict: CategoryDict):
        assert "ことわざ" in category_dict.axis_values("form")
        assert "失敗" in category_dict.axis_values("content")

    def test_未定義の軸名は例外(self, category_dict: CategoryDict):
        with pytest.raises(CategoryDictError):
            category_dict.axis_values("register")


class Testタグの導出:
    def test_formとcontentから導出する(self, category_dict: CategoryDict):
        """tags をフィールドで持たないので、軸の値とずれようがない。"""
        assert category_dict.get("kappa_1").tags() == ["#ことわざ", "#失敗"]

    def test_contentが空なら形式のみ(self, category_dict: CategoryDict):
        assert category_dict.get("tdl_1").tags() == ["#固有名詞"]


class Testfacetに依存しない:
    def test_facetの値に依存しない(self, category_dict: CategoryDict):
        """facet は注釈で埋まる。このモジュールは値を一切見ない。

        キーが 4 軸揃っていることと、値が 0.0-1.0 の範囲にあることだけを検査する。

        空の facet は許す。カテゴリ辞書に後から足した語はまだ注釈が無く、
        埋まるまでの間も辞書として使えなければならない。適当な値を
        入れて 4 軸を揃えるのは、注釈していないものを注釈済みに見せる
        ことになるのでしない。select() は欠けた軸を既定値で扱う。
        """
        for entry in category_dict.members():
            if not entry.facet:
                continue
            assert set(entry.facet) == {
                "physical",
                "psychological",
                "temporal",
                "logical",
            }
            for axis, v in entry.facet.items():
                assert isinstance(v, float), f"{entry.id}/{axis} が float でない"
                assert 0.0 <= v <= 1.0, f"{entry.id}/{axis} が範囲外: {v}"
