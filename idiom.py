"""
状況に適合する慣用句の選択。

3 段構成:
    1. カテゴリで候補を取る       CategoryDict.members()
    2. 前提条件で絞る             presupposition ⊆ situation
    3. Facet で 1 件に決める      select()

1 と 3 は格スロットの充填とまったく同じ機構である。慣用句だからといって
専用の探索を書かない。違うのは 2 の前提条件の照合だけ。

【前提条件を満たさない句は候補にしない】
「河童の川流れ」は presupposition が ["#熟練", "#常時成功"] である。
普段から失敗している相手に使えば嫌味になる。状況に "#熟練" が無いなら
候補から外す。規則ベースの利点はここで、前提条件を明示的に照合するので
誤用が起こらない。

presupposition が空配列の Entry（固有名詞など）は、
空集合がどんな集合の部分集合でもあるため常に条件を満たす。

【頻度を絞る理由】
誤用しないことと、使ってよいことは別である。
毎回ことわざで返す相手は、正しく使っていても定型的に見える。
使用頻度は style 側（generation_style.toml の idiom_rate）で制御し、
既定は 0.1 とする。
"""

from __future__ import annotations

import random

from .category_dict import CategoryDict, Entry
from .select import select


def eligible(candidates: list[Entry], situation: list[str]) -> list[Entry]:
    """前提条件が状況に満たされている Entry だけを残す。"""
    known = set(situation)
    return [e for e in candidates if set(e.presupposition) <= known]


def choose_idiom(
    cd: CategoryDict,
    form: str,
    content: str,
    situation: list[str],
    target: dict[str, float],
) -> Entry | None:
    """状況に合う慣用句を 1 件選ぶ。無ければ None。"""
    candidates = cd.members(form=form, content=content)
    return select(eligible(candidates, situation), target)


def should_use_idiom(rate: float, rng: random.Random | None = None) -> bool:
    """今回の発話で慣用句を使うか。

    rate=0.0 なら決して使わない。rate=1.0 なら必ず使う。
    乱数を注入できるようにしてあるのは、テストと再現のため。
    """
    if rate <= 0.0:
        return False
    if rate >= 1.0:
        return True
    return (rng or random.Random()).random() < rate


def maybe_choose_idiom(
    cd: CategoryDict,
    form: str,
    content: str,
    situation: list[str],
    target: dict[str, float],
    rate: float,
    rng: random.Random | None = None,
) -> Entry | None:
    """頻度の判定込みで慣用句を選ぶ。使わないと決まったら None。"""
    if not should_use_idiom(rate, rng):
        return None
    return choose_idiom(cd, form, content, situation, target)
