#!/usr/bin/env python3
"""
配信に映すための画面。

    cd D:\\AIModel
    python -X utf8 cognitag_struct/stream_gui.py
    python -X utf8 cognitag_struct/stream_gui.py --with=clock,weather

【何を見せる画面か】
会話そのものではなく、なぜその返答になったかを見せる。返答だけなら
他にいくらでもあるが、判断の過程が全部出る画面は珍しい。
左に会話、右に内訳を並べ、1 発話ごとに右が丸ごと入れ替わる。

【Tkinter を使う理由】
依存を増やさないため。このパッケージの依存は sudachipy だけ、という
状態を崩さずに画面を付けられるのは標準ライブラリだけである。

【配信での使い方】
OBS の「ウィンドウキャプチャ」で取り込む。背景を暗くしてあるので
そのまま重ねられる。手元を映さずに切り替えられるよう、
表示の操作はキー 1 つに割り当ててある。

    F1  内訳の表示を切り替える
    F2  字を大きく    F3  字を小さく
    F5  会話を消す
    Esc 終了
"""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitag_struct.chat import Responder, _load_capabilities  # noqa: E402
from cognitag_struct.facade import CogniTag  # noqa: E402

# 配色。暗い背景に落ち着いた前景。配信画面に重ねる前提。
BG = "#0e0e10"
PANEL = "#16161a"
LINE = "#2a2a30"
TEXT = "#e8e6e3"
DIM = "#8a8782"
YOU = "#7fb3ff"
BOT = "#8fd4c4"

# 内訳の行につける色。何の行かが一目で分かるようにする。
# 鍵は trace の行頭（「分割:」「タグ:」）と同じ文字列にしてある。
TRACE_COLORS: dict[str, str] = {
    "分割": "#9b8fd4",
    "カテゴリ辞書": "#8fd4c4",
    "種類": "#7fb3ff",
    "節0": "#c9a227",
    "節1": "#c9a227",
    "節の関係": "#c9a227",
    "補完": "#c9a227",
    "構文": "#e08f7f",
    "タグ": "#e08f7f",
    "含意": "#e08f7f",
    "文型": "#8fd4c4",
    "慣用句": "#d4a5c9",
    "外部": "#f0b76a",
    "文脈": "#8a8782",
    "一般規則": "#8fd4c4",
    "空きスロット": "#c9a227",
    "方針": DIM,
    "被覆": "#7fb3ff",
    "未知": "#f0b76a",
}


class StreamWindow:
    """会話と判断の内訳を並べて出す画面。"""

    def __init__(self, root: tk.Tk, responder: Responder) -> None:
        self.root = root
        self.responder = responder
        self.show_trace = True
        self.size = 15

        root.title("CogniTag — 配信用")
        root.configure(bg=BG)
        root.geometry("1280x720")

        self.body = tkfont.Font(family="Meiryo", size=self.size)
        self.small = tkfont.Font(family="Meiryo", size=self.size - 3)
        self.head = tkfont.Font(family="Meiryo", size=self.size - 3, weight="bold")

        self._build()
        self._bind()
        self._banner()

    # -- 画面の組み立て ---------------------------------------------------

    def _build(self) -> None:
        # 下端の 2 つを先に確保する。
        #
        # 会話のテキスト欄は既定で 24 行ぶんの高さを要求するので、
        # 先に pack すると入力欄がウィンドウの外へ押し出される。
        # place ではなく pack の順序で解くのは、窓の大きさを変えても
        # 崩れないようにするため。
        self.status = tk.Label(
            self.root, text="", bg=BG, fg=DIM, font=self.small, anchor="w"
        )
        self.status.pack(side="bottom", fill="x", padx=14, pady=(0, 8))
        self._build_input()

        outer = tk.Frame(self.root, bg=BG)
        outer.pack(side="top", fill="both", expand=True, padx=14, pady=10)

        left = tk.Frame(outer, bg=BG)
        left.pack(side="left", fill="both", expand=True)
        tk.Label(left, text="会話", bg=BG, fg=DIM, font=self.head,
                 anchor="w").pack(fill="x", pady=(0, 4))
        self.log = self._panel(left, self.body)
        self.log.tag_configure("you", foreground=YOU)
        self.log.tag_configure("bot", foreground=BOT)
        self.log.tag_configure("dim", foreground=DIM)

        self.right = tk.Frame(outer, bg=BG, width=520)
        self.right.pack(side="left", fill="both", padx=(12, 0))
        self.right.pack_propagate(False)
        tk.Label(self.right, text="判断の内訳", bg=BG, fg=DIM, font=self.head,
                 anchor="w").pack(fill="x", pady=(0, 4))
        self.trace = self._panel(self.right, self.small)
        for key, color in TRACE_COLORS.items():
            self.trace.tag_configure(key, foreground=color)
        self.trace.tag_configure("plain", foreground=TEXT)

    def _build_input(self) -> None:
        bottom = tk.Frame(self.root, bg=BG)
        bottom.pack(side="bottom", fill="x", padx=14, pady=(0, 10))
        tk.Label(bottom, text="入力", bg=BG, fg=DIM, font=self.small).pack(
            side="left", padx=(0, 10)
        )
        self.entry = tk.Entry(
            bottom, bg=PANEL, fg=TEXT, font=self.body, relief="flat",
            insertbackground=TEXT, highlightthickness=1,
            highlightbackground=LINE, highlightcolor=YOU,
        )
        self.entry.pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 8))
        self.entry.focus_set()
        tk.Button(
            bottom, text="送る", command=self._send, bg=LINE, fg=TEXT,
            font=self.small, relief="flat", padx=18, pady=6,
            activebackground=YOU, activeforeground=BG, cursor="hand2",
        ).pack(side="left")

    def _panel(self, parent: tk.Widget, font: tkfont.Font) -> tk.Text:
        widget = tk.Text(
            parent, bg=PANEL, fg=TEXT, font=font, wrap="word", relief="flat",
            # 既定の 24 行だと窓の高さを超えて要求してしまう。
            # 実際の高さは expand で決まるので、要求は小さくてよい。
            height=8, width=20,
            padx=14, pady=12, spacing1=2, spacing3=6, highlightthickness=1,
            highlightbackground=LINE, state="disabled",
        )
        widget.pack(fill="both", expand=True)
        return widget

    def _bind(self) -> None:
        self.root.bind("<Return>", lambda _e: self._send())
        self.root.bind("<F1>", lambda _e: self._toggle_trace())
        self.root.bind("<F2>", lambda _e: self._resize(2))
        self.root.bind("<F3>", lambda _e: self._resize(-2))
        self.root.bind("<F5>", lambda _e: self._clear())
        self.root.bind("<Escape>", lambda _e: self.root.destroy())

    def _banner(self) -> None:
        describe = self.responder.ct.describe().split(" / data=")[0]
        names = self.responder.capabilities.names()
        self.status.config(
            text=describe
            + (" / 外部: " + "、".join(names) if names else "")
            + "    [F1] 内訳  [F2/F3] 字の大きさ  [F5] 消す  [Esc] 終了"
        )
        self._write(self.log, "LLM を使わずに、辞書と構文だけで返します。\n\n", "dim")

    # -- 操作 -------------------------------------------------------------

    def _toggle_trace(self) -> None:
        """内訳の表示。配信中に手元を映さず切り替えるためのもの。"""
        self.show_trace = not self.show_trace
        if self.show_trace:
            self.right.pack(side="left", fill="both", padx=(12, 0))
        else:
            self.right.pack_forget()

    def _resize(self, delta: int) -> None:
        self.size = max(9, min(40, self.size + delta))
        self.body.configure(size=self.size)
        self.small.configure(size=max(7, self.size - 3))
        self.head.configure(size=max(7, self.size - 3))

    def _clear(self) -> None:
        """会話だけ消す。文脈は残す（消したら話が繋がらなくなる）。"""
        for widget in (self.log, self.trace):
            widget.configure(state="normal")
            widget.delete("1.0", "end")
            widget.configure(state="disabled")

    def _send(self) -> None:
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, "end")
        self._write(self.log, "あなた  ", "dim")
        self._write(self.log, text + "\n", "you")

        reply = self.responder.respond(text)
        self._write(self.log, "CogniTag  ", "dim")
        self._write(self.log, reply.text + "\n\n", "bot")
        self.log.see("end")
        self._show_trace(text, reply)

    def _show_trace(self, text: str, reply) -> None:
        """右側を丸ごと入れ替える。

        被覆率と未知語を先頭に置く。知らない語を隠さないことが
        この方式の主張なので、いちばん目に入る場所に出す。
        """
        analysis = self.responder.ct.analyze(text)
        self.trace.configure(state="normal")
        self.trace.delete("1.0", "end")

        self._write(self.trace, f"被覆 {analysis.coverage:.0%}\n", "被覆")
        unknown = analysis.unknown_words()
        if unknown:
            self._write(self.trace, "未知の語  " + "、".join(unknown) + "\n", "未知")
        self._write(self.trace, "\n", "plain")

        for line in reply.trace:
            key = line.split(":", 1)[0].strip()
            tag = key if key in TRACE_COLORS else "plain"
            self._write(self.trace, line + "\n", tag)
        self._write(self.trace, "\n方針: " + reply.policy + "\n", "方針")
        self.trace.configure(state="disabled")
        self.trace.see("1.0")

    @staticmethod
    def _write(widget: tk.Text, text: str, tag: str) -> None:
        widget.configure(state="normal")
        widget.insert("end", text, tag)
        widget.configure(state="disabled")


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    capabilities: list = []
    if argv and argv[0].startswith("--with="):
        capabilities = _load_capabilities(argv[0][len("--with="):])

    root = tk.Tk()
    StreamWindow(root, Responder(CogniTag(), capabilities=capabilities))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
