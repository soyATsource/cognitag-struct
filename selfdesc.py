"""
自分について答える。

【知識と自己記述は別のもの】
「富士山の高さ」は世界についての知識で、持とうとすると際限がない。
「自分が何であるか」は有限で、しかも書いた本人が正しい。
世界を知らないまま自分については答えられる、というのがこの層の役割。

    君は誰            → 私は CogniTag。
    何ができるの       → できるのは、話を受け止めること…
    疲れてない？       → 私は疲れないよ。

【相手が誰について尋ねているかを見分ける】
    2 人称の語がある      「君は誰」「あなたは何ができるの」
    感情・感覚の述語で主語が無い問い   「疲れてない？」

2 つ目が要るのは、日本語では主語を言わないためである。感情や感覚は
本人にしか分からないので、それを問う形になっていれば、聞かれているのは
聞き手（＝こちら）の状態だと決まる。

【答えを持っていないときは黙る】
data/self.toml に無いことは答えない。ここで推測すると、
自分について嘘をつくことになる。
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .modality import Modality

# 2 人称の語。これがあれば、こちらについて聞かれている。
SECOND_PERSON: frozenset[str] = frozenset(
    {"君", "きみ", "あなた", "お前", "そちら", "CogniTag", "コグニタグ"}
)

# 何を尋ねられたか。
IDENTITY = "identity"    # 誰か / 何者か
ABILITY = "ability"      # 何ができるか
NATURE = "nature"        # どういう状態か

# 「何ができるの」を見分ける見出し語。
ABILITY_LEMMAS: frozenset[str] = frozenset({"できる", "出来る"})
# 「君は誰」を見分ける疑問詞。
IDENTITY_LEMMAS: frozenset[str] = frozenset({"誰", "だれ", "何", "なに", "何者"})


class SelfError(ValueError):
    """自己記述が読めないときに送出する。"""


@dataclass
class SelfDescription:
    """自分について書いたもの。"""

    identity: str = ""
    ability: str = ""
    limit: str = ""
    nature: dict[str, str] = field(default_factory=dict)
    refuse: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "SelfDescription":
        target = Path(path)
        if not target.exists():
            raise SelfError(f"ファイルが無い: {target}")
        with target.open("rb") as handle:
            data = tomllib.load(handle)
        ability = data.get("ability") or {}
        return cls(
            identity=str((data.get("identity") or {}).get("answer", "")),
            ability=str(ability.get("answer", "")),
            limit=str(ability.get("limit", "")),
            nature={str(k): str(v) for k, v in (data.get("nature") or {}).items()},
            refuse={str(k): str(v) for k, v in (data.get("refuse") or {}).items()},
        )

    def answer(self, kind: str, tags: list[str] | None = None) -> str:
        """尋ねられた種別に対する答え。持っていなければ空。"""
        if kind == IDENTITY:
            return self.identity
        if kind == ABILITY:
            return (self.ability + self.limit) if self.ability else ""
        if kind == NATURE:
            present = set(tags or [])
            for tag, answer in self.nature.items():
                if tag in present:
                    return answer
        return ""


def asked_about_self(analysis, frames) -> str | None:
    """こちらについて尋ねられているか。種別を返す。違えば None。"""
    if analysis.modality is None:
        return None

    lemmas = {t.lemma for t in analysis.tokens}
    addressed = bool(lemmas & SECOND_PERSON)

    # 「君は誰」には終助詞も疑問符も無い。形の上では平叙だが、
    # 2 人称と疑問詞が揃っていれば問いとして扱ってよい。
    interrogative = bool(lemmas & IDENTITY_LEMMAS)
    if analysis.modality.modality not in (Modality.Q_OPEN, Modality.Q_YESNO):
        if not (addressed and interrogative):
            return None

    if lemmas & ABILITY_LEMMAS:
        return ABILITY
    if addressed and (lemmas & IDENTITY_LEMMAS):
        return IDENTITY

    # 主語の無い、感情・感覚の問い。「疲れてない？」
    #
    # 本人にしか分からないことを問う形なので、聞かれているのは
    # 聞き手の状態だと決まる。2 人称の語は要らない。
    from .ir import IR

    if isinstance(analysis.ir, IR):
        for clause in analysis.ir.clauses:
            predicate = clause.predicate
            if predicate is None or clause.slots:
                continue
            frame = frames.get(predicate.lemma)
            if frame is not None and frame.experiencer:
                return NATURE
    return NATURE if addressed else None
