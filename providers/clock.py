"""
時刻と日付。

ネットワークは要らない。時計は手元にある。
「今何時ですか」に「観測する手立てを持っていない」と返していたのは、
単に見ていなかっただけである。天気と違って、これは外部 API の話ですらない。

【なぜ既定で有効にしないか】
時刻を見ると同じ入力に同じ返答を返さなくなる。決定性はこの実装の
性質として明示しているものなので、既定は無効にして、
使う側が「時計を差し込む」と決めたときだけ動くようにする。
"""

from __future__ import annotations

from datetime import datetime

from ..capability import Request, Result
from ..modality import Modality

# 時刻を尋ねていると判断する見出し語。
TIME_LEMMAS: frozenset[str] = frozenset({"何時", "時刻", "時間"})
DATE_LEMMAS: frozenset[str] = frozenset({"何日", "何曜日", "日付", "今日"})

KIND_TIME = "clock.time"
KIND_DATE = "clock.date"

WEEKDAYS = ("月", "火", "水", "木", "金", "土", "日")


class Clock:
    """時計。問いのときだけ答える。"""

    name = "clock"

    def __init__(self, now=None) -> None:
        # 現在時刻の取り方を差し替えられるようにする。
        # 固定した時計を渡せばテストが決定的になる。
        self._now = now or datetime.now

    def wants(self, analysis) -> Request | None:
        """時刻・日付を尋ねる問いか。

        問いに限る。「時間がない」は平叙で、時刻を聞かれてはいない。
        """
        if analysis.modality is None:
            return None
        if analysis.modality.modality not in (Modality.Q_OPEN, Modality.Q_YESNO):
            return None

        lemmas = {t.lemma for t in analysis.tokens}
        if lemmas & TIME_LEMMAS:
            return Request(kind=KIND_TIME, text=analysis.text)
        if lemmas & DATE_LEMMAS:
            return Request(kind=KIND_DATE, text=analysis.text)
        return None

    def handle(self, request: Request) -> Result | None:
        now = self._now()
        if request.kind == KIND_TIME:
            return Result(
                text=f"{now.hour} 時 {now.minute} 分だ。",
                source="手元の時計",
                detail=now.isoformat(timespec="minutes"),
            )
        if request.kind == KIND_DATE:
            weekday = WEEKDAYS[now.weekday()]
            return Result(
                text=f"{now.year} 年 {now.month} 月 {now.day} 日、{weekday}曜日だ。",
                source="手元の時計",
                detail=now.date().isoformat(),
            )
        return None
