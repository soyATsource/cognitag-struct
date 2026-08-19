"""
Blender に基本形状を作らせる。試験的な能力。

【これは「外に問う」ではなく「外にやらせる」例】
天気や時刻は外から値をもらうだけだが、こちらは外の道具を動かす。
差し込み口が両方を扱えることを確かめるために入れてある。

【入力から Python を組み立てない】
利用者の言葉をそのまま Blender に渡して実行させる作りにはしない。
「立方体を作って」から `bpy.ops.mesh.primitive_cube_add()` を選ぶのであって、
文からコードを生成するのではない。実行するのは下の SHAPES に
書いてある固定の呼び出しだけで、利用者の言葉が入る余地は
形状の選択（辞書の鍵）と、こちらが組み立てた出力パスしかない。

規則ベースでできることの範囲を狭く保つのは不便に見えるが、
「入力次第で何が起きるか分からない」状態を作らないための線引きである。

【既定では無効】
Blender の起動には数秒かかり、ファイルも書く。使う側が明示的に
差し込んだときだけ動く。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..capability import Request, Result
from ..modality import Modality

# 作れる形。鍵が見出し語、値が Blender の呼び出し。
#
# ここに書いていないものは作らない。「何か格好いいものを作って」には
# 応じない。何を作ったか説明できないものは作らない方がよい。
SHAPES: dict[str, tuple[str, str]] = {
    "立方体": ("cube", "bpy.ops.mesh.primitive_cube_add()"),
    "キューブ": ("cube", "bpy.ops.mesh.primitive_cube_add()"),
    "球": ("sphere", "bpy.ops.mesh.primitive_uv_sphere_add()"),
    "球体": ("sphere", "bpy.ops.mesh.primitive_uv_sphere_add()"),
    "円柱": ("cylinder", "bpy.ops.mesh.primitive_cylinder_add()"),
    "円錐": ("cone", "bpy.ops.mesh.primitive_cone_add()"),
    "平面": ("plane", "bpy.ops.mesh.primitive_plane_add()"),
    "トーラス": ("torus", "bpy.ops.mesh.primitive_torus_add()"),
}

# 作れと言われたと判断する述語。
MAKE_LEMMAS: frozenset[str] = frozenset({"作る", "追加する", "置く", "出す"})

KIND = "blender.primitive"

# 既定の探索先。新しい版を優先する。
INSTALL_ROOT = Path(r"C:\Program Files\Blender Foundation")


def find_blender(root: Path = INSTALL_ROOT) -> Path | None:
    """blender.exe を探す。見つからなければ None。"""
    if not root.exists():
        return None
    found = sorted(root.glob("Blender */blender.exe"))
    return found[-1] if found else None


class Blender:
    """基本形状を 1 つ作って .blend に保存する。"""

    name = "blender"

    def __init__(
        self,
        executable: str | Path | None = None,
        output_dir: str | Path = "blender_out",
        timeout: int = 120,
    ) -> None:
        self.executable = Path(executable) if executable else find_blender()
        self.output_dir = Path(output_dir)
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return self.executable is not None and self.executable.exists()

    def wants(self, analysis) -> Request | None:
        """形を作れと言われているか。

        依頼・意志・願望に限る。「立方体を作りました」は報告であって
        こちらへの指示ではない。
        """
        if not self.available or analysis.modality is None:
            return None
        if analysis.modality.modality not in (
            Modality.REQUEST, Modality.VOLITION, Modality.DESIRE
        ):
            return None

        lemmas = [t.lemma for t in analysis.tokens]
        if not (set(lemmas) & MAKE_LEMMAS):
            return None
        for lemma in lemmas:
            if lemma in SHAPES:
                return Request(
                    kind=KIND, text=analysis.text, slots={"shape": lemma}
                )
        return None

    def handle(self, request: Request) -> Result | None:
        shape = request.slots.get("shape")
        operation = SHAPES.get(shape or "")
        if operation is None:
            return None
        stem, call = operation

        self.output_dir.mkdir(parents=True, exist_ok=True)
        target = (self.output_dir / f"{stem}.blend").resolve()

        # 実行するのは固定の呼び出しと、こちらが組み立てたパスだけ。
        # 利用者の言葉は 1 文字も入らない。
        script = (
            "import bpy;"
            "bpy.ops.wm.read_factory_settings(use_empty=True);"
            f"{call};"
            f"bpy.ops.wm.save_as_mainfile(filepath=r'{target}')"
        )
        try:
            completed = subprocess.run(
                [str(self.executable), "--background", "--python-expr", script],
                capture_output=True, text=True, timeout=self.timeout,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None

        if completed.returncode != 0 or not target.exists():
            return None

        return Result(
            text=f"{shape}を作って {target.name} に保存した。",
            source=f"Blender（{self.executable.parent.name}）",
            detail=str(target),
        )
