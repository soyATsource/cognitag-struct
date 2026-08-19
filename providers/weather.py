"""
天気。Open-Meteo を使う。

【なぜ Open-Meteo か】
API キーが要らない。キーが要ると、鍵の置き場所と秘匿の話が始まり、
「単体で動く」から遠ざかる。使う側が何も用意せずに試せることを優先した。

【場所は引数で受け取る】
既定を東京にしてあるが、これは動かすための既定値であって、
「この人はここにいる」という主張ではない。位置を推定しない。
発話から場所を読み取ることもしない（「名古屋の天気は」に答えるには
地名から座標を引く辞書が要る。持っていないものは持っていないと言う）。

【失敗しても例外を投げない】
ネットワークが落ちているのは異常ではなく通常の結果である。
None を返して、呼び出し側の「答えられない」経路に戻す。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from ..capability import Request, Result
from ..modality import Modality

ENDPOINT = "https://api.open-meteo.com/v1/forecast"

# 天気を尋ねていると判断する見出し語。
WEATHER_LEMMAS: frozenset[str] = frozenset(
    {"天気", "雨", "晴れ", "曇り", "雪", "気温", "降る"}
)

KIND = "weather.current"
KIND_TOMORROW = "weather.tomorrow"

# 「いつの天気か」を表す語。ここに無い時（明後日・来週）は答えない。
#
# 現在の天気を返しておきながら「明日は？」に答えた顔をするのが一番悪い。
# 扱える範囲を狭く決めて、外れたら黙って渡し返す。
TODAY_LEMMAS: frozenset[str] = frozenset({"今", "今日", "現在", "いま"})
TOMORROW_LEMMAS: frozenset[str] = frozenset({"明日"})
OTHER_TIME_LEMMAS: frozenset[str] = frozenset(
    {"明後日", "来週", "来月", "昨日", "一昨日", "先週", "週末"}
)

# WMO の天気コード。必要な範囲だけ。
CODES: dict[int, str] = {
    0: "快晴", 1: "晴れ", 2: "薄曇り", 3: "曇り",
    45: "霧", 48: "霧",
    51: "霧雨", 53: "霧雨", 55: "霧雨",
    61: "雨", 63: "雨", 65: "強い雨",
    71: "雪", 73: "雪", 75: "強い雪",
    80: "にわか雨", 81: "にわか雨", 82: "激しいにわか雨",
    95: "雷雨", 96: "雷雨", 99: "雷雨",
}


class Weather:
    """現在の天気。問いのときだけ答える。"""

    name = "weather"

    def __init__(
        self,
        latitude: float = 35.68,
        longitude: float = 139.77,
        place: str = "東京",
        timeout: int = 8,
    ) -> None:
        self.latitude = latitude
        self.longitude = longitude
        self.place = place
        self.timeout = timeout

    def wants(self, analysis) -> Request | None:
        if analysis.modality is None:
            return None
        if analysis.modality.modality not in (Modality.Q_OPEN, Modality.Q_YESNO):
            return None
        lemmas = {t.lemma for t in analysis.tokens}
        if not (lemmas & WEATHER_LEMMAS):
            return None
        if lemmas & OTHER_TIME_LEMMAS:
            # 扱えない時のことを聞かれた。現在の天気で代用しない。
            return None
        if lemmas & TOMORROW_LEMMAS:
            return Request(kind=KIND_TOMORROW, text=analysis.text)
        return Request(kind=KIND, text=analysis.text)

    def _fetch(self, query: str) -> dict | None:
        url = (
            f"{ENDPOINT}?latitude={self.latitude}&longitude={self.longitude}"
            f"&{query}&timezone=Asia%2FTokyo"
        )
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            # 繋がらないことは通常の結果。黙って渡し返す。
            return None

    def handle(self, request: Request) -> Result | None:
        if request.kind == KIND_TOMORROW:
            return self._tomorrow()

        data = self._fetch("current=temperature_2m,precipitation,weather_code")
        if data is None:
            return None
        current = data.get("current") or {}
        if "temperature_2m" not in current:
            return None

        sky = CODES.get(int(current.get("weather_code", -1)), "不明")
        temperature = current["temperature_2m"]
        rain = float(current.get("precipitation", 0.0))
        wet = "降っている" if rain > 0.0 else "降っていない"

        return Result(
            text=f"{self.place}はいま{sky}、{temperature} 度。雨は{wet}。",
            source="Open-Meteo（現在）",
            detail=json.dumps(current, ensure_ascii=False),
        )

    def _tomorrow(self) -> Result | None:
        """明日の予報。予報であることを言葉に残す。

        現在の観測と予報を同じ口調で返すと、どちらを聞いたのか
        分からなくなる。「予報では」と付けて区別する。
        """
        data = self._fetch(
            "daily=weather_code,temperature_2m_max,temperature_2m_min"
            "&forecast_days=2"
        )
        if data is None:
            return None
        daily = data.get("daily") or {}
        try:
            code = int(daily["weather_code"][1])
            high = daily["temperature_2m_max"][1]
            low = daily["temperature_2m_min"][1]
            date = daily["time"][1]
        except (KeyError, IndexError, TypeError, ValueError):
            return None

        sky = CODES.get(code, "不明")
        return Result(
            text=f"予報では、明日の{self.place}は{sky}。最高 {high} 度、最低 {low} 度だ。",
            source="Open-Meteo（予報）",
            detail=f"{date} code={code}",
        )
