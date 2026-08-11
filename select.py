"""
候補 Entry から Facet 4 軸で 1 件を選ぶ。

【この関数がカテゴリ層と Facet 層の境目になる】
粗いカテゴリで候補を数件に絞り、連続値の Facet で 1 件に決める。
候補の取り出しは CategoryDict.members() が担い、絞り込みはここが担う。
格スロットの充填でも慣用句の選択でも操作は同じなので、
専用の関数を別に作らず、この 1 本に集約する。

【valence を距離に入れない理由】
valence は 0 が「強い不快」、0.5 が中立という双極スケールで、
「その側面をどれだけ持つか」を測る 4 軸とは意味が違う。
同じ空間の座標として扱うとユークリッド距離が意味を失う。
距離は physical / psychological / temporal / logical の 4 軸だけで測る。
"""

from __future__ import annotations

from math import sqrt

from .category_dict import Entry

# 距離を測る軸。valence は含めない（上の説明を参照）。
DISTANCE_AXES: tuple[str, ...] = (
    "physical",
    "psychological",
    "temporal",
    "logical",
)


def distance(facet: dict[str, float], target: dict[str, float]) -> float:
    """4 軸のユークリッド距離。欠けている軸は 0.0 とみなす。"""
    total = 0.0
    for axis in DISTANCE_AXES:
        delta = float(facet.get(axis, 0.0)) - float(target.get(axis, 0.0))
        total += delta * delta
    return sqrt(total)


def select(candidates: list[Entry], target: dict[str, float]) -> Entry | None:
    """target に最も近い Entry を 1 件返す。候補が無ければ None。

    【同値のときに id の辞書順で決める理由】
    現在 Entry.facet はすべて 0.0 のプレースホルダで、
    どの候補も target から等距離になる。辞書が埋まるまでの間も
    出力が再現可能でなければ、生成結果が実行ごとに変わってしまう。
    距離が同じなら id の小さい方を選ぶ、と決めておけば決定的になる。

    例外は投げない。候補が無いのは異常ではなく、
    「その条件に合う言い回しが辞書に無かった」という通常の結果である。
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        # 候補が 1 件なら Facet を見るまでもない
        return candidates[0]

    return min(candidates, key=lambda e: (distance(e.facet, target), e.id))
