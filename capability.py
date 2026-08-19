"""
外の世界に問い合わせる差し込み口。

【なぜ核に API を書かないか】
この層の取り柄は「単体で動く」「同じ入力に同じ返答」「根拠が全部見える」
の 3 つである。天気 API を core に埋め込むと 3 つとも壊れる。

    単体で動く   → ネットワークが要る
    決定性       → 返答が日によって変わる
    根拠が見える → 外から来た値は追えない

そこで core は「この発話は外の何かが要る」と判断するところまでを担い、
実際の問い合わせは差し込まれた能力（Capability）に渡す。
既定では何も差し込まれていないので、上の 3 つはそのまま保たれる。

【差し込み口をここに置く理由】
モダリティの判定は既に「外部知識が要る問いか」を出している
（modality.needs_knowledge）。判断はできていて、渡す先が無かっただけである。
新しい判断を足すのではなく、既にある判断の出口を用意する。

【使い方】

    from cognitag_struct.chat import Responder
    from cognitag_struct.providers.clock import Clock

    bot = Responder(capabilities=[Clock()])
    bot.respond("今何時ですか")

【外から来た値は必ず印を付ける】
Result.source に「どこから来たか」を書く。返答の根拠を追えることが
この方式の主張なので、辞書と構文から出た文と、外から来た値を
混ぜたまま出さない。:trace には必ず出る。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class Request:
    """能力に渡す依頼。

    kind は「何を求めているか」の種別。能力の側が名乗る文字列で、
    core は中身を知らない。core が知っているのは「渡した」ことだけ。
    """

    kind: str
    text: str
    # 能力が使う手がかり。格スロットの中身やタグを入れる。
    slots: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


@dataclass
class Result:
    """能力からの答え。

    text がそのまま返答になる。source は出どころで、:trace に出す。
    失敗したときは None を返すこと（例外を投げない）。外が落ちている
    ことは異常ではなく、日常的に起きる通常の結果である。
    """

    text: str
    source: str
    detail: str = ""


@runtime_checkable
class Capability(Protocol):
    """外の世界を触る部品。

    2 つのことをする。
        wants()  この発話が自分の担当かを判断し、依頼を作る
        handle() 実際に問い合わせて答えを作る

    判断と実行を同じ部品に置くのは、担当範囲を知っているのが
    その部品自身だからである。core 側に振り分け表を持たせると、
    能力を足すたびに core を触ることになる。
    """

    name: str

    def wants(self, analysis) -> Request | None:
        """自分の担当なら Request を返す。違えば None。"""
        ...

    def handle(self, request: Request) -> Result | None:
        """問い合わせて答えを作る。できなければ None。"""
        ...


class Registry:
    """差し込まれた能力の一覧。

    先に登録されたものが優先する。同じ発話を 2 つの能力が
    担当しうる場合（「今日の天気」は時刻とも天気とも取れる）に、
    順序で決められるようにしておく。集合にすると実行ごとに変わる。
    """

    def __init__(self, capabilities: list[Capability] | None = None) -> None:
        self.capabilities: list[Capability] = list(capabilities or [])

    def add(self, capability: Capability) -> None:
        self.capabilities.append(capability)

    def __len__(self) -> int:
        return len(self.capabilities)

    def __bool__(self) -> bool:
        return bool(self.capabilities)

    def names(self) -> list[str]:
        return [c.name for c in self.capabilities]

    def consult(self, analysis) -> tuple[Result, str] | None:
        """担当する能力を探して答えを得る。

        戻り値は (答え, 能力の名前)。誰も担当しない、または
        担当したが答えられなかった場合は None。
        呼び出し側は None なら通常の「答えられない」経路へ進む。
        """
        for capability in self.capabilities:
            request = capability.wants(analysis)
            if request is None:
                continue
            result = capability.handle(request)
            if result is not None:
                return result, capability.name
        return None
