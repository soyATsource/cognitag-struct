"""
述語ごとの必須スロット定義。

「行く」には行き先が要る、「熱い」には主体が要る、といった知識をここに置く。
gap.py がこれと IR を突き合わせて「何が空いているか」を判定する。

【content_constraints は制約ではない】
候補を絞るための優先度であって、違反しても解析を失敗させない。
現時点では句辞書に無い語の content が空配列なので、この照合はほとんど
機能しない。CogniTag 本体の辞書と対応付けたときに効く前提の枠組みである。

【volitional / motion を持つ理由】
volitional は主語補完に使う。「行きません」から主語が発話者だと分かるのは、
丁寧語・主語なし・意志動詞という条件がそろったときだけである（gap.py 参照）。

motion は PURPOSE の判定に使う。「会いに行く」が目的を表すのは、
後半が移動動詞だからである。content_constraints で代用しようとすると
現時点では content が空で判定できない。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


class FrameError(ValueError):
    """フレーム定義が不正なときに送出する。"""


@dataclass
class Frame:
    """1 つの述語が取るスロットの定義。"""

    lemma: str
    required: list[str] = field(default_factory=list)
    optional: list[str] = field(default_factory=list)
    # スロット -> 望ましい content 値。優先度であって制約ではない。
    content_constraints: dict[str, list[str]] = field(default_factory=dict)
    # 意志動詞か。主語補完の条件のひとつ。
    volitional: bool = False
    # 移動動詞か。PURPOSE の判定に使う。
    motion: bool = False
    # 感情・感覚の述語か。主語が明示されなければ話者とみなす。
    #
    # 「疲れました」「眠いです」「痛いです」の主語は話者である。
    # これは意志動詞かどうかとは別の性質で、疲れるのは意志ではない。
    # volitional だけで判定していると、これらに「誰が？」と
    # 聞き返してしまう。
    #
    # 意志動詞の補完は丁寧語を条件にするが、こちらは要らない。
    # 「疲れた」も主語は話者である（感情・感覚は本人しか分からない）。
    experiencer: bool = False
    # 意味タグ。#移動 #感情 #疲労 のような粗い括り。
    #
    # これが返答の内容を決める。「疲れた」に #疲労 が付いていれば、
    # reasoning.toml から「疲労なら休んだ方がいい」を引いて
    # 「これは疲労だから休んだ方がいいよね」の形で返せる。
    #
    # 粗くてよい。細かくすると管理できなくなるし、含意も書けなくなる。
    tags: list[str] = field(default_factory=list)

    def slots(self) -> list[str]:
        return list(self.required) + list(self.optional)


class FrameDict:
    """lemma からフレームを引く。"""

    def __init__(self, frames: list[Frame]) -> None:
        self.frames: dict[str, Frame] = {}
        for frame in frames:
            if frame.lemma in self.frames:
                raise FrameError(f"lemma が重複している: {frame.lemma}")
            self.frames[frame.lemma] = frame

    @classmethod
    def load(cls, path: str | Path) -> "FrameDict":
        """パスは引数で受け取る（ハードコードしない）。"""
        target = Path(path)
        if not target.exists():
            raise FrameError(f"ファイルが無い: {target}")

        frames: list[Frame] = []
        with target.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                try:
                    raw = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise FrameError(f"{target}:{line_no} JSON が壊れている: {exc}")
                lemma = raw.get("lemma")
                if not lemma:
                    raise FrameError(f"{target}:{line_no} lemma が無い")
                frames.append(
                    Frame(
                        lemma=str(lemma),
                        required=list(raw.get("required") or []),
                        optional=list(raw.get("optional") or []),
                        content_constraints=dict(
                            raw.get("content_constraints") or {}
                        ),
                        volitional=bool(raw.get("volitional", False)),
                        motion=bool(raw.get("motion", False)),
                        experiencer=bool(raw.get("experiencer", False)),
                        tags=list(raw.get("tags") or []),
                    )
                )
        return cls(frames)

    def get(self, lemma: str) -> Frame | None:
        return self.frames.get(lemma)

    def is_motion(self, lemma: str) -> bool:
        frame = self.frames.get(lemma)
        return bool(frame and frame.motion)

    def __len__(self) -> int:
        return len(self.frames)
