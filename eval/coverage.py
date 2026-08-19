#!/usr/bin/env python3
"""
辞書がどれだけの語を説明できるかを測る。

    cd D:\\AIModel
    python -X utf8 cognitag_struct/eval/coverage.py --corpus <テキストの入った場所>
    python -X utf8 cognitag_struct/eval/coverage.py --cases

【なぜ語数ではなく被覆率か】
「あと何語あれば足りるか」は答えの出ない問いである。辞書に何語あるかは
性能ではなく在庫の話で、実際の文章で何割を説明できたかとは別物だからだ。
語数を数えるのをやめて、対象の文章に対する被覆率を測る。

【被覆率より大事なのは、未知を未知と言えること】
知らない語を知っているふりをすると、この方式の取り柄（間違えない）が
崩れる。被覆 90% で未知の申告が正確な方が、被覆 98% で取りこぼす
ものより良い。したがって未知語は必ず一覧で出す。目で見て、
「これは知っているはずなのに未知になっている」を見つけるためである。

【合成で得た分類は別に数える】
「会議室」は辞書に無いが、末尾の「室」から場所だと分かる。これは
登録された語と同じ確度ではないので、被覆率とは分けて報告する。
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cognitag_struct.facade import CONTENT_POS, CogniTag  # noqa: E402

HERE = Path(__file__).resolve().parent
# 1 文の長さの上限。長すぎる行は会話の入力として現実的でない。
MAX_SENTENCE = 80


def sentences(text: str) -> list[str]:
    """句点で文に割る。話者印（A:）は落とす。"""
    import re

    text = re.sub(r"^[A-Z]\s*[:：]\s*", "", text, flags=re.MULTILINE)
    found = []
    for line in re.split(r"[。！？\n]", text):
        line = line.strip()
        if 2 <= len(line) <= MAX_SENTENCE:
            found.append(line)
    return found


def load_corpus(args) -> list[str]:
    if args.cases:
        from cognitag_struct.eval.run_eval import load_cases

        return [turn for case in load_cases() for turn in case.turns]
    target = Path(args.corpus)
    if target.is_dir():
        files = sorted(target.glob("*.txt"))[: args.files]
        return [s for f in files for s in sentences(f.read_text(encoding="utf-8"))]
    return sentences(target.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="辞書の被覆率を測る")
    parser.add_argument("--corpus", help="テキストのファイルかディレクトリ")
    parser.add_argument("--cases", action="store_true",
                        help="評価セットの入力を対象にする")
    parser.add_argument("--files", type=int, default=40,
                        help="ディレクトリのとき読むファイル数")
    parser.add_argument("--top", type=int, default=30, help="未知語の表示件数")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not args.cases and not args.corpus:
        parser.error("--corpus か --cases が要る")

    lines = load_corpus(args)
    if not lines:
        print("対象の文が 1 つも無い")
        return 1

    cognitag = CogniTag()
    total = known = composed = 0
    unknown: collections.Counter[str] = collections.Counter()
    parsed = 0

    for line in lines:
        analysis = cognitag.analyze(line)
        parsed += 1 if analysis.parsed else 0
        for token in analysis.content_words():
            total += 1
            if token.entry_id or token.lemma in analysis.known_predicates:
                known += 1
            elif token.content:
                composed += 1
            else:
                unknown[token.lemma] += 1

    print("=" * 62)
    print("辞書の被覆率")
    print("=" * 62)
    print(f"文            {len(lines)}")
    print(f"内容語        {total}")
    print(f"構造が取れた文 {parsed} / {len(lines)}  "
          f"({parsed / len(lines) * 100:.1f}%)")
    print()
    print(f"辞書にある     {known:6d}  {known / total * 100:5.1f}%")
    print(f"合成で分かる   {composed:6d}  {composed / total * 100:5.1f}%"
          "  ← 複合語。末尾の主要部から判定")
    print(f"未知と申告     {sum(unknown.values()):6d}  "
          f"{sum(unknown.values()) / total * 100:5.1f}%")
    print()
    print(f"被覆率（辞書のみ）      {known / total * 100:.1f}%")
    print(f"被覆率（合成を含む）    {(known + composed) / total * 100:.1f}%")
    print()
    print(f"未知語の異なり数  {len(unknown)}")
    print("-" * 62)
    print("よく出る未知語（ここに日常語があれば、辞書に足す価値がある）")
    for word, count in unknown.most_common(args.top):
        print(f"  {count:4d}  {word}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
