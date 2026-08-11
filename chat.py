#!/usr/bin/env python3
"""
質問 → 返答 の入口。これ 1 つで動く。

    python -m cognitag_struct.chat

【何ができて、何ができないか】
LLM を一切使わない。辞書と構文だけで返す。したがって知識を要する問いには
答えられない。「なぜ空が青いのか」と聞かれたら、答えられないと返す。

できること:
    依頼を受領する / 願望を受け止める / 推量に保留する
    必須スロットが空なら尋ねる（「行きます」→「どこに？」）
    理解した内容を言い直す（構造を取れたことの確認）

できないこと:
    事実を答える / 説明する / 要約する / 創作する

もっともらしい嘘を作らないのがこの方式の取り柄なので、
答えられないことは答えられないと返す。

【なぜ返答が短いのか】
返答は「理解した構造」から組み立てている。知識ベースが無いので、
構造から言えること以上は言えない。逆に、返した内容の根拠は
すべて追跡できる。:trace を有効にすると、どのトークンが
どう効いて結論に至ったかが全部見える。ここがブラックボックスとの違い。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

# 単体実行（python chat.py）でも動くようにする。
# パッケージとして呼ばれた場合（python -m）は既に import できている。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitag_struct.context import Context  # noqa: E402
from cognitag_struct.facade import Analysis, CogniTag  # noqa: E402
from cognitag_struct.ir import IR, Clause  # noqa: E402
from cognitag_struct.modality import Modality  # noqa: E402
from cognitag_struct.reasoning import MODALITY_TAGS  # noqa: E402

# 応答の型が data に無かった場合の最終手段。
# 通常は generation_style.toml の [reply] が使われる。
FALLBACK = "……"


@dataclass
class Reply:
    """返答 1 件。text だけ使ってもよいし、trace で根拠を辿ってもよい。"""

    text: str
    policy: str = ""
    trace: list[str] = field(default_factory=list)


class Responder:
    """解析結果から返答を決める。

    ここが「応答方針の写像」にあたる。発話の種類（モダリティ）を主、
    欠落スロットを従にして 1 つの型を選ぶ。文言は style が持つ。

    【推量と依頼では欠落を問い詰めない】
    「行くかな」に「どこに？」と返すのは不自然である。gap.py は
    モダリティを見ないので、その制御はこの層で行う。
    欠落そのものは保持されているので、後から尋ね直すこともできる。
    """

    # このモダリティのときは、必須スロットが空でも問い詰めない
    NO_PROBE = (Modality.SPECULATION, Modality.REQUEST, Modality.DESIRE)

    def __init__(self, cognitag: CogniTag | None = None) -> None:
        self.ct = cognitag or CogniTag()
        # 会話の文脈。直前に尋ねたことを覚えておく。
        # これが無いと「どこに？」→「名古屋」が繋がらない。
        self.context = Context()

    def _template(self, key: str) -> str:
        return self.ct.style.reply.get(key, FALLBACK)

    def _reasoning_text(self, analysis, skip: tuple[str, ...] = ()) -> str:
        """タグから引いた含意を 1 つの文にする。

        「これは<X>だから<Y>だよね」の中身。複数の含意が取れても
        並べすぎると読みにくいので、reason() が既に上限をかけている。

        skip には、既に応答の型で言い切っているタグを渡す。
        「どうしよう」に「決めるのは君だ」と返した直後に
        「これからの話だな。決めるのは本人だ」と続けたら重複する。
        """
        if analysis.reasoning is None or not analysis.reasoning.has_content:
            return ""
        kept = [
            i for i in analysis.reasoning.implications if i.tag not in skip
        ]
        return "。".join(i.as_sentence() for i in kept)

    def respond(self, text: str) -> Reply:
        """返答を決め、文脈へ記録する。

        判断は _decide に分けてある。出口が 10 箇所あるので、
        各 return に記録処理を足すと必ずどこかで漏れる。
        包んで 1 箇所で済ませる。
        """
        if not text.strip():
            return Reply(self._template("empty"), policy="empty")

        analysis = self.ct.analyze(text)
        reply = self._decide(analysis)
        self._remember(analysis, reply)
        return reply

    def _decide(self, analysis: Analysis) -> Reply:
        text = analysis.text
        trace = self._trace(analysis)
        modality = analysis.modality.modality

        # -1. 直前に尋ねたことへの答えか。単語だけの返事を受け取る。
        answered = self._try_answer(analysis, trace)
        if answered is not None:
            return answered

        # 0. 挨拶。構造は無いが返せる。
        if modality is Modality.GREETING:
            return Reply(self._template("greeting"), policy="greeting", trace=trace)

        # 1. 知識を要する問い。答えられないと返す。
        if modality is Modality.Q_OPEN:
            words = "・".join(analysis.modality.interrogatives) or "それ"
            return Reply(
                self._template("q_open").format(interrogative=words),
                policy="q_open", trace=trace,
            )

        # 2. 肯否の問い。真偽の材料が無い。
        if modality is Modality.Q_YESNO:
            return Reply(self._template("q_yesno"), policy="q_yesno", trace=trace)

        # 3. 依頼・願望・推量・意志。受け止める。欠落は問い詰めない。
        #    含意が取れていれば添えるが、その型で既に言っていることは省く。
        for target, key in (
            (Modality.REQUEST, "request"),
            (Modality.DESIRE, "desire"),
            (Modality.SPECULATION, "speculation"),
            (Modality.VOLITION, "volition"),
        ):
            if modality is target:
                own = MODALITY_TAGS.get(target, "")
                extra = self._reasoning_text(analysis, skip=(own,))
                base = self._template(key)
                return Reply(
                    f"{base}{extra}" if extra else base, policy=key, trace=trace
                )

        reasoning = self._reasoning_text(analysis)

        # 4. 平叙。構造が取れなくても、タグが取れていれば返せる。
        #    「構造として取れなかった」で突き放すのは最後の手段にする。
        questions = analysis.questions()
        if reasoning:
            if questions and modality not in self.NO_PROBE:
                return Reply(
                    self._template("with_reasoning_gap").format(
                        reasoning=reasoning, question=questions[0]
                    ),
                    policy="with_reasoning_gap", trace=trace,
                )
            return Reply(
                self._template("with_reasoning").format(reasoning=reasoning),
                policy="with_reasoning", trace=trace,
            )

        # 4.5 構造が取れない断片。突き放す前に 2 つ試す。
        if not analysis.parsed:
            # 指示語があるなら直近の話題に置き換えて受ける。
            # 「それはどう」は、何の話かが分かれば受け答えできる。
            topic = self.context.resolve_demonstrative(analysis.tokens)
            if topic is not None:
                trace.append(f"文脈: 指示語を「{topic.surface}」と解決した")
                base = self._template("resolved").format(topic=topic.surface)
                tail = (
                    self._template("q_open").format(
                        interrogative="・".join(analysis.modality.interrogatives)
                        or "それ"
                    )
                    if analysis.modality.interrogatives
                    else ""
                )
                return Reply(f"{base}{tail}", policy="resolved", trace=trace)

            # 疑問詞だけの断片。「どう？」「なぜ？」も問いとして扱う。
            if analysis.modality.interrogatives:
                words = "・".join(analysis.modality.interrogatives)
                return Reply(
                    self._template("q_open").format(interrogative=words),
                    policy="q_open", trace=trace,
                )

            return Reply(
                self._template("unparsable"), policy="unparsable", trace=trace
            )

        # 5. 含意は無いが必須スロットが空。尋ねる。
        if questions and modality not in self.NO_PROBE:
            return Reply(
                self._template("statement_gap").format(question=questions[0]),
                policy="statement_gap", trace=trace,
            )

        # 6. 何も言えることが無い。理解した内容を言い直す。
        understood = self.ct.generate(analysis.ir, verbosity=2).text
        return Reply(
            self._template("statement").format(understood=understood),
            policy="statement", trace=trace,
        )

    def _try_answer(self, analysis: Analysis, trace: list[str]) -> Reply | None:
        """直前の質問への答えとして受け取れるなら、埋め戻して返す。"""
        answer = self.context.answer_fills_pending(analysis.tokens)
        if answer is None:
            return None

        clause = self.context.filled_clause(answer)
        self.context.clear_pending()
        if clause is None:
            return None

        filled = IR(clauses=[clause], relation=None, source_text=analysis.text)
        understood = self.ct.generate(filled, verbosity=2).text
        extra = self._reasoning_text(analysis)
        trace.append(f"文脈: 直前の質問への答えとして {answer.surface} を受け取った")

        key = "answered_with_reasoning" if extra else "answered"
        return Reply(
            self._template(key).format(understood=understood, reasoning=extra),
            policy=key, trace=trace,
        )

    def _remember(self, analysis: Analysis, reply: Reply) -> None:
        """1 ターン分を文脈へ記録する。"""
        self.context.remember(
            analysis.ir if isinstance(analysis.ir, IR) else None, analysis.tokens
        )
        # 尋ねた場合は、次の入力を答えとして受け取れるようにする
        if reply.policy in ("statement_gap", "with_reasoning_gap") and (
            analysis.gaps and analysis.gaps.gaps and isinstance(analysis.ir, IR)
        ):
            gap = analysis.gaps.gaps[0]
            clause = analysis.ir.clauses[gap.clause_index]
            self.context.ask(gap.slot, gap.lemma, gap.question, clause)

    def _trace(self, analysis: Analysis) -> list[str]:
        """どう解析したかを人が読める形で並べる。

        返答の根拠がすべてここに出る。ブラックボックスでないという
        主張は、この一覧が出せることを指している。
        """
        lines: list[str] = []
        lines.append(
            "分割: " + " | ".join(
                f"{t.surface}({t.pos}"
                + (f"/{t.subpos}" if t.subpos and t.subpos != "*" else "")
                + (f"){t.features}" if t.features else ")")
                for t in analysis.tokens
            )
        )
        phrases = [t for t in analysis.tokens if t.is_phrase]
        if phrases:
            lines.append(
                "句辞書: " + "、".join(f"{t.surface}[{t.entry_id}]" for t in phrases)
            )
        modality = analysis.modality
        lines.append(
            f"種類: {modality.modality.value}"
            + (f" ← {'/'.join(modality.evidence)}" if modality.evidence else "")
            + (f" 態度{modality.attitude}" if modality.attitude else "")
        )
        if isinstance(analysis.ir, IR):
            for index, clause in enumerate(analysis.ir.clauses):
                slots = " ".join(
                    f"{k}={v.surface}" for k, v in clause.slots.items()
                )
                lines.append(
                    f"節{index}: 述語={clause.predicate.lemma if clause.predicate else '?'}"
                    f" {slots}"
                    + (f" 主題={clause.topic.surface}" if clause.topic else "")
                    + (f" {clause.modifiers}" if clause.modifiers else "")
                )
            if analysis.ir.relation:
                lines.append(f"節の関係: {analysis.ir.relation}")
        else:
            lines.append(f"構文: 取れなかった（{analysis.ir.reason}）")
        if analysis.gaps and analysis.gaps.filled:
            lines.append(
                "補完: " + "、".join(
                    f"{f.slot}={f.value}（{f.reason}）" for f in analysis.gaps.filled
                )
            )
        if analysis.reasoning is not None:
            lines.append(f"タグ: {analysis.reasoning.summary()}")
            for implication in analysis.reasoning.implications:
                lines.append(
                    f"含意: {implication.tag} → {implication.label} / {implication.so}"
                )
        if analysis.questions():
            lines.append(f"空きスロット: {analysis.questions()}")
        return lines


# --- 対話ループ -------------------------------------------------------------

BANNER = """\
CogniTag 構造層 — 質問と返答
  LLM を使いません。辞書と構文だけで返します。
  知識を要する問いには「答えられない」と返します。

  :trace   解析の内訳を表示する / しない を切り替える
  :say     入力を言い直す（verbosity 0〜3 を並べる）
  :help    この説明
  :quit    終了（Ctrl+C でも可）
"""


def _show_say(responder: Responder, text: str) -> None:
    analysis = responder.ct.analyze(text)
    if not analysis.parsed:
        print("  構造が取れないので言い直せない")
        return
    for verbosity in range(4):
        utterance = responder.ct.generate(analysis.ir, verbosity)
        held = utterance.withheld_surfaces()
        print(f"  {verbosity}: {utterance.text}" + (f"   保留={held}" if held else ""))


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        responder = Responder()
    except Exception as exc:
        print(f"起動できませんでした: {exc}")
        print("SudachiPy と sudachidict_core が入っているか確認してください。")
        return 1

    # 引数があれば 1 回だけ答えて終わる（スクリプトから呼ぶ用）
    if argv:
        reply = responder.respond(" ".join(argv))
        print(reply.text)
        return 0

    print(BANNER)
    print(f"  {responder.ct.describe()}\n")

    show_trace = False
    while True:
        try:
            text = input("あなた> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            return 0

        if not text:
            continue
        if text in (":quit", ":q", ":exit"):
            print("終了します。")
            return 0
        if text == ":help":
            print(BANNER)
            continue
        if text == ":trace":
            show_trace = not show_trace
            print(f"  内訳の表示を{'有効' if show_trace else '無効'}にしました")
            continue
        if text.startswith(":say"):
            target = text[4:].strip()
            if target:
                _show_say(responder, target)
            else:
                print("  使い方: :say 名古屋に行きます")
            continue

        reply = responder.respond(text)
        if show_trace:
            for line in reply.trace:
                print(f"    {line}")
            print(f"    方針: {reply.policy}")
        print(f"CogniTag> {reply.text}\n")


if __name__ == "__main__":
    raise SystemExit(main())
