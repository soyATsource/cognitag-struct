#!/usr/bin/env python3
"""
評価セットを走らせて採点する。

    cd D:\\AIModel
    python -X utf8 cognitag_struct/eval/run_eval.py
    python -X utf8 cognitag_struct/eval/run_eval.py --compare cognitag_struct/eval/baseline.json

【何のためにあるか】
辞書を足したときに、返答が良くなったのか悪くなったのかを判定するため。
点数そのものより、**変更の前後で下がった項目を見つけること**が主目的である。
辞書を 1 つ足すと、狙った入力は良くなり、狙っていない入力が壊れる。
壊れたことに気づかないのが一番まずい。

【プロダクトコードを変更しない】
判定に使うのは facade / chat が既に公開している値だけ。測定のために
観測点を本体へ足すと、測るための仕組みが測られる対象を変えてしまう。

【決定性】
同じ入力に同じ結果を返すことがこの実装の性質なので、採点も 2 回走らせれば
完全に一致する。baseline.json に時刻を書かないのはそのため（差分が
時刻だけで出ると、退行の検出という主目的が埋もれる）。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# 単体実行（python cognitag_struct/eval/run_eval.py）でも動くようにする。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cognitag_struct.chat import Reply, Responder  # noqa: E402
from cognitag_struct.facade import CogniTag  # noqa: E402
from cognitag_struct.modality import Modality  # noqa: E402

HERE = Path(__file__).resolve().parent
CASES = HERE / "cases.jsonl"
BASELINE = HERE / "baseline.json"

# 期待値として書けるキー。ここに無いキーはケースファイルの誤記として弾く。
#
# 綴りを間違えた期待値を黙って無視すると、「その観点は検証されている」と
# 思い込んだまま検証されない状態になる。測定の土台としては最悪なので、
# 未知のキーは読み込み時に失敗させる。
CHECK_KEYS: frozenset[str] = frozenset(
    {
        "modality",           # 発話の種類（Modality の名前）
        "parsed",             # 構造が取れたか（真偽）
        "tags_all",           # このタグが全部付いていること
        "tags_none",          # このタグが 1 つも付いていないこと
        "question_contains",  # 返答の問い返しにこの語が含まれること
        "no_question",        # 問い返しをしないこと（真偽）
        "policy",             # 応答方針の完全一致
        "policy_prefix",      # 応答方針の前方一致（answered* をまとめる用）
        "text_contains",      # 返答にこの語が含まれること
        "text_none",          # 返答にこの語が含まれないこと
    }
)

CASE_KEYS: frozenset[str] = frozenset(
    {"id", "turns", "expect", "known_gap", "note"}
)

MODALITY_NAMES: frozenset[str] = frozenset(m.value for m in Modality)

# 「問い返した」と言えるのはこの方針を選んだときだけ。
#
# Analysis.questions() は空きスロットがあれば埋まるが、実際に尋ねるかどうかは
# Responder が決める（推量・依頼・願望では問い詰めない。chat.py:73 の NO_PROBE）。
# 「行きますか」の解析には「どこに？」が入るが、返答は問い返していない。
# 測るべきは利用者に届いた返答なので、方針の方を見る。
PROBE_POLICIES: frozenset[str] = frozenset({"statement_gap", "with_reasoning_gap"})


class CaseError(ValueError):
    """ケースファイルの記述が不正なときに送出する。"""


@dataclass
class Case:
    """評価ケース 1 件。expect は最後のターンに対する期待値。"""

    id: str
    turns: list[str]
    expect: dict
    known_gap: bool = False
    note: str = ""


@dataclass
class CheckResult:
    """観点 1 つ分の判定。"""

    key: str
    passed: bool
    detail: str = ""


@dataclass
class CaseResult:
    case: Case
    checks: list[CheckResult] = field(default_factory=list)
    reply_text: str = ""

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def failures(self) -> list[CheckResult]:
        return [check for check in self.checks if not check.passed]


# --- 読み込み ---------------------------------------------------------------


def load_cases(path: str | Path = CASES) -> list[Case]:
    """ケースを読む。記述の誤りはその場で失敗させる（上の説明を参照）。"""
    target = Path(path)
    if not target.exists():
        raise CaseError(f"ファイルが無い: {target}")

    cases: list[Case] = []
    seen: set[str] = set()
    with target.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise CaseError(f"{target}:{line_no} JSON が壊れている: {exc}")

            unknown = set(raw) - CASE_KEYS
            if unknown:
                raise CaseError(
                    f"{target}:{line_no} 未知のキー: {sorted(unknown)}"
                )

            case_id = raw.get("id")
            if not case_id:
                raise CaseError(f"{target}:{line_no} id が無い")
            if case_id in seen:
                raise CaseError(f"{target}:{line_no} id が重複している: {case_id}")
            seen.add(case_id)

            turns = raw.get("turns")
            if not isinstance(turns, list) or not turns:
                raise CaseError(f"{target}:{line_no} turns が空")

            expect = raw.get("expect") or {}
            unknown_checks = set(expect) - CHECK_KEYS
            if unknown_checks:
                raise CaseError(
                    f"{target}:{line_no} 未知の期待値: {sorted(unknown_checks)}"
                )
            if not expect:
                raise CaseError(f"{target}:{line_no} expect が空")

            modality = expect.get("modality")
            if modality is not None and modality not in MODALITY_NAMES:
                raise CaseError(
                    f"{target}:{line_no} 存在しないモダリティ: {modality!r}"
                )

            cases.append(
                Case(
                    id=str(case_id),
                    turns=[str(t) for t in turns],
                    expect=expect,
                    known_gap=bool(raw.get("known_gap", False)),
                    note=str(raw.get("note", "")),
                )
            )
    return cases


# --- 採点 -------------------------------------------------------------------


def _check(key: str, expected, actual, passed: bool) -> CheckResult:
    if passed:
        return CheckResult(key, True)
    return CheckResult(key, False, f"{key}: 期待 {expected!r} / 実際 {actual!r}")


def score_case(case: Case, cognitag: CogniTag) -> CaseResult:
    """1 ケースを走らせて観点ごとに判定する。

    Responder はケースごとに作る。文脈が残ると前のケースの質問に
    次のケースの入力が答えてしまい、単独では再現しない結果になる。
    CogniTag（辞書と Sudachi）は重いので使い回す。
    """
    responder = Responder(cognitag)
    reply: Reply | None = None
    for turn in case.turns:
        reply = responder.respond(turn)
    assert reply is not None  # turns が空でないことは読み込み時に保証済み

    # 最後のターンをもう一度解析する。respond() は Analysis を返さないので、
    # タグやモダリティはここで取り直す。Responder は状態を持つが analyze は
    # 持たないため、同じ入力に同じ結果が返る。
    analysis = cognitag.analyze(case.turns[-1])
    tags = analysis.reasoning.tags if analysis.reasoning else []
    questions = analysis.questions()

    expect = case.expect
    checks: list[CheckResult] = []

    if "modality" in expect:
        actual = analysis.modality.modality.value if analysis.modality else None
        checks.append(
            _check("modality", expect["modality"], actual, actual == expect["modality"])
        )

    if "parsed" in expect:
        checks.append(
            _check("parsed", expect["parsed"], analysis.parsed,
                   analysis.parsed == bool(expect["parsed"]))
        )

    if "tags_all" in expect:
        missing = [t for t in expect["tags_all"] if t not in tags]
        checks.append(_check("tags_all", expect["tags_all"], tags, not missing))

    if "tags_none" in expect:
        present = [t for t in expect["tags_none"] if t in tags]
        checks.append(_check("tags_none", expect["tags_none"], tags, not present))

    asked = reply.policy in PROBE_POLICIES

    if "question_contains" in expect:
        # 返答に出たかを見る。解析が質問を用意していても、返答が
        # それを使わなかったなら利用者には届いていない。
        word = expect["question_contains"]
        hit = asked and word in reply.text
        checks.append(
            _check("question_contains", word, (reply.policy, questions), hit)
        )

    if "no_question" in expect:
        want_none = bool(expect["no_question"])
        checks.append(
            _check("no_question", want_none, (reply.policy, questions),
                   (not asked) == want_none)
        )

    if "policy" in expect:
        checks.append(
            _check("policy", expect["policy"], reply.policy,
                   reply.policy == expect["policy"])
        )

    if "policy_prefix" in expect:
        prefix = expect["policy_prefix"]
        checks.append(
            _check("policy_prefix", prefix, reply.policy,
                   reply.policy.startswith(prefix))
        )

    if "text_contains" in expect:
        word = expect["text_contains"]
        checks.append(_check("text_contains", word, reply.text, word in reply.text))

    if "text_none" in expect:
        word = expect["text_none"]
        checks.append(_check("text_none", word, reply.text, word not in reply.text))

    return CaseResult(case=case, checks=checks, reply_text=reply.text)


def run(cases: list[Case]) -> list[CaseResult]:
    cognitag = CogniTag()
    return [score_case(case, cognitag) for case in cases]


# --- 集計と表示 -------------------------------------------------------------


def data_fingerprint() -> str:
    """データファイルの指紋。

    【辞書を更新したら判定が変わりうる】
    決定的であることと、更新しても壊れないことは別の話である。
    結果と一緒に指紋を残しておけば、差分が出たときに
    「データが変わったのか、コードが変わったのか」を切り分けられる。
    """
    import hashlib

    digest = hashlib.sha256()
    data = HERE.parent / "data"
    for path in sorted(data.iterdir()):
        if path.suffix in (".jsonl", ".toml"):
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def summarize(results: list[CaseResult]) -> dict:
    """観点別・全体の集計。known_gap は本体の点数に混ぜない。

    混ぜると、既知の弱点を評価セットから消すだけで点数が上がってしまう。
    別枠にしておけば「弱点が何件あり、そのうち何件が解消したか」が残る。
    """
    main = [r for r in results if not r.case.known_gap]
    gaps = [r for r in results if r.case.known_gap]

    by_key: dict[str, dict[str, int]] = {}
    for result in main:
        for check in result.checks:
            bucket = by_key.setdefault(check.key, {"passed": 0, "total": 0})
            bucket["total"] += 1
            bucket["passed"] += 1 if check.passed else 0

    return {
        "data": data_fingerprint(),
        "cases": {
            "passed": sum(1 for r in main if r.passed),
            "total": len(main),
        },
        "known_gap": {
            # known_gap は「今は落ちるのが正しい」ケース。
            # resolved が増えたら、その分だけ弱点が消えたということ。
            "resolved": sum(1 for r in gaps if r.passed),
            "total": len(gaps),
        },
        "by_key": {k: by_key[k] for k in sorted(by_key)},
        "failed_ids": sorted(r.case.id for r in main if not r.passed),
        "resolved_gap_ids": sorted(r.case.id for r in gaps if r.passed),
    }


def _rate(passed: int, total: int) -> str:
    if total == 0:
        return "  -  "
    return f"{passed / total * 100:5.1f}%"


def report(results: list[CaseResult], summary: dict) -> None:
    print("=" * 62)
    print("CogniTag 構造層 — 返答の採点")
    print("=" * 62)

    cases = summary["cases"]
    print(f"\nケース  {cases['passed']:3d}/{cases['total']:3d}  "
          f"{_rate(cases['passed'], cases['total'])}")

    gap = summary["known_gap"]
    print(f"既知の弱点  {gap['resolved']:3d}/{gap['total']:3d} 解消"
          "  （点数には含めない）")

    print("\n観点別")
    print("-" * 62)
    for key, bucket in summary["by_key"].items():
        print(f"  {key:<20} {bucket['passed']:3d}/{bucket['total']:3d}  "
              f"{_rate(bucket['passed'], bucket['total'])}")

    failures = [r for r in results if not r.case.known_gap and not r.passed]
    if failures:
        print("\n落ちたケース")
        print("-" * 62)
        for result in failures:
            print(f"  [{result.case.id}] {' → '.join(result.case.turns)}")
            print(f"      返答: {result.reply_text}")
            for check in result.failures():
                print(f"      {check.detail}")

    resolved = [r for r in results if r.case.known_gap and r.passed]
    if resolved:
        print("\n解消した弱点（期待値を known_gap から外してよい）")
        print("-" * 62)
        for result in resolved:
            print(f"  [{result.case.id}] {' → '.join(result.case.turns)}")


def compare(summary: dict, previous_path: Path) -> int:
    """前回との差分を出す。退行があれば 1 を返す。

    改善より退行を目立たせる。辞書を足す作業では、狙った改善は
    確認しなくても分かるが、巻き添えの破損は探さないと見つからない。
    """
    if not previous_path.exists():
        print(f"\n比較対象が無い: {previous_path}")
        return 0

    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    print("\n" + "=" * 62)
    print(f"前回との差分  ({previous_path.name})")
    print("=" * 62)

    regressed = False

    old_data = previous.get("data")
    if old_data and old_data != summary["data"]:
        print(f"\nデータが変わっている: {old_data} → {summary['data']}")
        print("  判定の差はデータ由来かもしれない。コードだけを疑わないこと。")
    elif old_data:
        print(f"\nデータは同一: {summary['data']}")

    before = previous.get("cases", {})
    delta = summary["cases"]["passed"] - before.get("passed", 0)
    print(f"\nケース  {before.get('passed', 0)} → {summary['cases']['passed']}"
          f"  ({delta:+d})")

    print("\n観点別の変化")
    print("-" * 62)
    keys = sorted(set(summary["by_key"]) | set(previous.get("by_key", {})))
    for key in keys:
        now = summary["by_key"].get(key, {"passed": 0, "total": 0})
        old = previous.get("by_key", {}).get(key, {"passed": 0, "total": 0})
        diff = now["passed"] - old["passed"]
        mark = "  " if diff == 0 else ("↑" if diff > 0 else "↓ 退行")
        if diff != 0 or now["total"] != old["total"]:
            print(f"  {key:<20} {old['passed']}/{old['total']} → "
                  f"{now['passed']}/{now['total']}  {mark}")
        if diff < 0:
            regressed = True

    # 前回通っていたのに落ちたケース。これが一番重要な出力。
    newly_failed = sorted(
        set(summary["failed_ids"]) - set(previous.get("failed_ids", []))
    )
    if newly_failed:
        regressed = True
        print("\n新たに落ちたケース（退行）")
        print("-" * 62)
        for case_id in newly_failed:
            print(f"  {case_id}")

    newly_fixed = sorted(
        set(previous.get("failed_ids", [])) - set(summary["failed_ids"])
    )
    if newly_fixed:
        print("\n新たに通ったケース")
        print("-" * 62)
        for case_id in newly_fixed:
            print(f"  {case_id}")

    new_resolved = sorted(
        set(summary["resolved_gap_ids"]) - set(previous.get("resolved_gap_ids", []))
    )
    if new_resolved:
        print("\n解消した既知の弱点")
        print("-" * 62)
        for case_id in new_resolved:
            print(f"  {case_id}")

    if not regressed:
        print("\n退行なし。")
    return 1 if regressed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="返答の採点")
    parser.add_argument("--cases", default=str(CASES), help="評価セット")
    parser.add_argument("--compare", metavar="PATH", help="この結果と比較する")
    parser.add_argument("--save", metavar="PATH", help="結果の保存先")
    parser.add_argument(
        "--no-save", action="store_true", help="保存しない（試し打ち用）"
    )
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        cases = load_cases(args.cases)
    except CaseError as exc:
        print(f"評価セットが読めない: {exc}")
        return 2

    results = run(cases)
    summary = summarize(results)
    report(results, summary)

    status = 0
    if args.compare:
        status = compare(summary, Path(args.compare))

    if not args.no_save:
        destination = Path(args.save) if args.save else BASELINE
        # 時刻は書かない。2 回走らせて完全一致することを保ちたいため。
        destination.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\n保存: {destination}")

    return status


if __name__ == "__main__":
    raise SystemExit(main())
