"""
よく聞かれる事実に答える。

【百科事典を持たない】
世界の知識を網羅しようとすると終わりが来ないし、中途半端に持つと
「知っている顔で間違える」ようになる。この方式の取り柄は
知らないと言えることなので、そこを崩さない。

ここに置くのは「何度も聞かれるので答えを用意しておく」ものだけ。
載っていないことは今までどおり分からないと返す。境界は動かさない。

【条件を狭くする】
    about の語がすべて出ている  「富士山」と「高さ」の両方
    問いの形である              「富士山に行きたい」には答えない

部分一致で答えると、無関係な問いに引っかかる。答えられる場面を
狭く決めておく方が、間違えないことに寄与する。

【出どころを残す】
辞書から出た答えと、構造から組み立てた返答を混ぜたまま見せない。
どこから来た値かは :trace に出す。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .modality import Modality


class FactError(ValueError):
    """事実の表が読めないときに送出する。"""


@dataclass
class Fact:
    """答えを用意してある事実 1 件。"""

    id: str
    about: list[str]
    answer: str
    source: str = ""


class FactTable:
    """事実の一覧。data/facts.jsonl から読む。"""

    def __init__(self, facts: list[Fact]) -> None:
        self.facts = facts

    @classmethod
    def load(cls, path: str | Path) -> "FactTable":
        target = Path(path)
        if not target.exists():
            raise FactError(f"ファイルが無い: {target}")

        facts: list[Fact] = []
        seen: set[str] = set()
        for line_no, line in enumerate(
            target.read_text(encoding="utf-8").splitlines(), 1
        ):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise FactError(f"{target}:{line_no} JSON が壊れている: {exc}")
            identifier = str(raw.get("id", ""))
            about = raw.get("about")
            answer = str(raw.get("answer", ""))
            if not identifier or not isinstance(about, list) or not answer:
                raise FactError(f"{target}:{line_no} id / about / answer が要る")
            if identifier in seen:
                raise FactError(f"{target}:{line_no} id が重複: {identifier}")
            seen.add(identifier)
            facts.append(
                Fact(
                    id=identifier,
                    about=[str(a) for a in about],
                    answer=answer,
                    source=str(raw.get("source", "")),
                )
            )
        return cls(facts)

    def __len__(self) -> int:
        return len(self.facts)

    def find(self, analysis) -> Fact | None:
        """問いに答えられる事実。無ければ None。

        問いの形に限る。「富士山に行きたい」は高さを尋ねていない。
        """
        if analysis.modality is None:
            return None
        if analysis.modality.modality not in (Modality.Q_OPEN, Modality.Q_YESNO):
            return None

        words = {t.surface for t in analysis.tokens}
        words |= {t.lemma for t in analysis.tokens}
        for fact in self.facts:
            if all(word in words for word in fact.about):
                return fact
        return None
