"""
Facet による 1 件選択と、慣用句の前提条件照合の検証。
"""

from __future__ import annotations

import random

from cognitag_struct.category_dict import Entry
from cognitag_struct.idiom import (
    choose_idiom,
    eligible,
    maybe_choose_idiom,
    should_use_idiom,
)
from cognitag_struct.select import distance, select

NEUTRAL = {"physical": 0.0, "psychological": 0.0, "temporal": 0.0, "logical": 0.0}


def entry(entry_id: str, **facet) -> Entry:
    values = dict(NEUTRAL)
    values.update(facet)
    return Entry(
        id=entry_id,
        surface=entry_id,
        lemma=entry_id,
        pos="名詞",
        form="ことわざ",
        content=["失敗"],
        presupposition=[],
        facet=values,
    )


class Test距離で選ぶ:
    def test_targetに最も近いものを返す(self):
        """プレースホルダ 0.0 に依存しないよう Facet を注入する。"""
        candidates = [
            entry("far", physical=1.0),
            entry("near", logical=0.9),
            entry("mid", logical=0.5),
        ]
        target = {"logical": 1.0}

        assert select(candidates, target).id == "near"

    def test_別のtargetなら別の答え(self):
        candidates = [
            entry("phys", physical=1.0),
            entry("logi", logical=1.0),
        ]

        assert select(candidates, {"physical": 1.0}).id == "phys"
        assert select(candidates, {"logical": 1.0}).id == "logi"

    def test_距離の計算(self):
        assert distance({"physical": 1.0}, {"physical": 1.0}) == 0.0
        assert distance({"physical": 1.0}, NEUTRAL) == 1.0

    def test_valenceは距離に含めない(self):
        """双極スケールなので他の 4 軸と同じ空間で測らない。"""
        a = entry("a")
        a.facet["valence"] = 1.0
        b = entry("b")
        b.facet["valence"] = 0.0

        assert distance(a.facet, NEUTRAL) == distance(b.facet, NEUTRAL)


class Test決定性:
    def test_全部0なら辞書順(self):
        """現在の辞書はすべて 0.0 のプレースホルダで全候補が同距離になる。

        それでも出力は再現可能でなければならない。
        """
        candidates = [entry("zebra"), entry("alpha"), entry("mike")]
        assert select(candidates, NEUTRAL).id == "alpha"

    def test_10回実行して一致する(self):
        candidates = [entry("zebra"), entry("alpha"), entry("mike")]
        results = {select(candidates, NEUTRAL).id for _ in range(10)}
        assert results == {"alpha"}

    def test_同距離なら辞書順(self):
        candidates = [
            entry("b", logical=0.5),
            entry("a", logical=0.5),
        ]
        assert select(candidates, {"logical": 1.0}).id == "a"


class Test候補が少ない:
    def test_1件ならFacetを見ずに返す(self):
        only = entry("only", physical=1.0)
        # target からどれだけ遠くても返る
        assert select([only], {"logical": 1.0}) is only

    def test_0件はNone(self):
        assert select([], NEUTRAL) is None

    def test_0件でも例外を投げない(self):
        """条件に合う言い回しが無いのは異常ではない。"""
        assert select([], {"logical": 1.0}) is None


class Test前提条件の照合:
    def test_熟練が無ければ河童は選ばれない(self, category_dict):
        """普段から失敗している相手に使えば嫌味になる。"""
        chosen = choose_idiom(
            category_dict, "ことわざ", "失敗", situation=[], target=NEUTRAL
        )
        assert chosen is None

    def test_熟練があれば候補になる(self, category_dict):
        candidates = eligible(
            category_dict.members(form="ことわざ", content="失敗"),
            ["#熟練", "#常時成功"],
        )
        assert len(candidates) == 3
        assert {e.surface for e in candidates} == {
            "河童の川流れ",
            "弘法にも筆の誤り",
            "猿も木から落ちる",
        }

    def test_条件が揃えば1件選ばれる(self, category_dict):
        chosen = choose_idiom(
            category_dict,
            "ことわざ",
            "失敗",
            situation=["#熟練", "#常時成功"],
            target=NEUTRAL,
        )
        assert chosen is not None
        assert chosen.form == "ことわざ"

    def test_一部しか満たさなければ選ばれない(self, category_dict):
        chosen = choose_idiom(
            category_dict, "ことわざ", "失敗", ["#熟練"], NEUTRAL
        )
        assert chosen is None

    def test_前提が空なら常に候補(self, category_dict):
        """固有名詞は presupposition が空配列。空集合は常に部分集合。"""
        candidates = eligible(
            category_dict.members(form="固有名詞"), situation=[]
        )
        assert len(candidates) == 5

    def test_無関係な状況が混ざっていてもよい(self, category_dict):
        chosen = choose_idiom(
            category_dict,
            "ことわざ",
            "無駄",
            ["#価値を解さない", "#無関係な条件"],
            NEUTRAL,
        )
        assert chosen is not None
        assert chosen.content == ["無駄"]


class Test使用頻度:
    def test_頻度0なら一切使わない(self, category_dict):
        rng = random.Random(0)
        for _ in range(50):
            assert (
                maybe_choose_idiom(
                    category_dict,
                    "ことわざ",
                    "失敗",
                    ["#熟練", "#常時成功"],
                    NEUTRAL,
                    rate=0.0,
                    rng=rng,
                )
                is None
            )

    def test_頻度1なら必ず使う(self, category_dict):
        rng = random.Random(0)
        for _ in range(10):
            assert (
                maybe_choose_idiom(
                    category_dict,
                    "ことわざ",
                    "失敗",
                    ["#熟練", "#常時成功"],
                    NEUTRAL,
                    rate=1.0,
                    rng=rng,
                )
                is not None
            )

    def test_判定は乱数を注入できる(self):
        assert should_use_idiom(0.0) is False
        assert should_use_idiom(1.0) is True
        assert should_use_idiom(0.5, random.Random(1)) in (True, False)

    def test_同じ種なら同じ結果(self, category_dict):
        def run():
            rng = random.Random(42)
            return [
                should_use_idiom(0.3, rng) for _ in range(20)
            ]

        assert run() == run()
