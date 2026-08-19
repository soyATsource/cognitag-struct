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
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# 単体実行（python chat.py）でも動くようにする。
# パッケージとして呼ばれた場合（python -m）は既に import できている。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitag_struct.capability import Registry  # noqa: E402
from cognitag_struct.context import Context  # noqa: E402
from cognitag_struct.facade import Analysis, CogniTag  # noqa: E402
from cognitag_struct.ir import IR, POLITE, Clause  # noqa: E402
from cognitag_struct.modality import Modality  # noqa: E402
from cognitag_struct.reasoning import MODALITY_TAGS  # noqa: E402

# 応答の型が data に無かった場合の最終手段。
# 通常は generation_style.toml の [reply] が使われる。
FALLBACK = "……"

# 慣用句として引く form 軸の値。
# 慣用句・四字熟語へ広げるときはここを増やすのではなく、
# axes.jsonl の form に沿って data 側で分ければよい。
IDIOM_FORM = "ことわざ"

# 記号だけの入力を見分けるための品詞。
SYMBOL_POS = "補助記号"

# 観測しないと分からない話題。天気・時刻・数量など。
#
# 「明日は雨ですか」に「真偽は判断できない」と返すのは間違いではないが、
# なぜ答えられないかが伝わらない。外界を見ないと出てこない話だと
# 言えた方が、この方式の限界がはっきりする。
OBSERVABLE_TAGS: frozenset[str] = frozenset({"#自然", "#時間", "#数量", "#場所"})

# 内容について何も言っていないタグ。素性と文型だけのもの。
# これしか無いときは、含意を並べるより構造から言う方が中身が出る。
WEAK_TAGS: frozenset[str] = frozenset(
    {"#過去", "#否定", "#質問", "#仮定", "#不確定", "#意志"}
)


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

    def __init__(
        self,
        cognitag: CogniTag | None = None,
        capabilities: list | None = None,
    ) -> None:
        self.ct = cognitag or CogniTag()
        # 外の世界を触る部品。既定では空で、何も繋がっていない。
        # 空である限り、返答は辞書と構文だけから決まる。
        self.capabilities = Registry(capabilities)
        # 会話の文脈。直前に尋ねたことを覚えておく。
        # これが無いと「どこに？」→「名古屋」が繋がらない。
        self.context = Context()
        # 使い終わった慣用句。同じ会話で二度は使わない。
        self._used_idioms: set[str] = set()
        # 直前の発話で慣用句を使ったか。二連続では使わない。
        self._idiom_last_turn = False
        # 応答の型ごとの使用回数。言い回しを選ぶのに使う。
        self._used_templates: Counter[str] = Counter()
        # 文型ごとの使用回数。まだ使っていない言い方を先に使う。
        self._used_patterns: Counter[str] = Counter()
        # 既に言った含意。同じ会話で繰り返さない。
        self._used_implications: set[str] = set()

    def _template(self, key: str) -> str:
        """応答の型から言い回しを 1 つ選ぶ。

        【同じ型を使うたびに言い方を変える】
        「うるさい」と 2 回言われて 2 回とも同じ文を返すのは、
        会話として不自然である。かといって乱数を入れると、
        同じ会話を再現できなくなる（この実装の性質を捨てることになる）。

        そこで「その型を何回目に使ったか」で選ぶ。会話の中では言い方が
        変わり、同じ会話を最初からやり直せば同じ結果になる。
        乱数を使わずに繰り返しを避けられる。
        """
        variants = self.ct.style.reply.get(key) or [FALLBACK]
        index = self._used_templates[key] % len(variants)
        self._used_templates[key] += 1
        return variants[index]

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
        # 記号だけの入力。「。。。」「？？？」。
        # 構造が取れないのは当然なので「短く言ってほしい」とは返さない。
        if analysis.tokens and all(
            t.pos == SYMBOL_POS for t in analysis.tokens
        ):
            return Reply(self._template("empty"), policy="empty")
        reply = self._decide(analysis)
        self._add_idiom(analysis, reply)
        self._remember(analysis, reply)
        return reply

    def _add_idiom(self, analysis: Analysis, reply: Reply) -> None:
        """話題に合う慣用句があれば返答に添える。

        【カテゴリから取ってくる】
        話題のタグ（#困難 など）が指す (form × content) のカテゴリを引き、
        前提条件を満たす句だけに絞り、Facet で 1 件に決める。
        探索は idiom.py が持っている 3 段構成をそのまま使う。
        句ごとに条件を書くのではなく、カテゴリ単位で扱うのが要点で、
        「ことわざ（無駄）」に何件入っていても呼び出し側は変わらない。

        【乱数で頻度を落とさない】
        idiom.py には確率で使う仕組みもあるが、ここでは使わない。
        同じ入力に同じ返答を返すという性質を捨てたくないためである。
        代わりに「同じ句は二度使わない」「二連続では使わない」で抑える。
        条件を満たす句がそもそも少ないので、これで十分に散る。
        """
        situation = analysis.reasoning.tags if analysis.reasoning else []
        if not situation or self._idiom_last_turn:
            self._idiom_last_turn = False
            return

        for tag in situation:
            content = self.ct.style.idiom_trigger.get(tag)
            if content is None:
                continue
            entry = self.ct.idiom(
                form=IDIOM_FORM, content=content, situation=situation
            )
            if entry is None or entry.id in self._used_idioms:
                continue
            self._used_idioms.add(entry.id)
            self._idiom_last_turn = True
            reply.text += "。" + self.ct.style.idiom_template.format(
                surface=entry.surface
            )
            reply.trace.append(
                f"慣用句: {tag} → ({IDIOM_FORM}×{content}) から "
                f"{entry.surface}[{entry.id}]"
                + (f" 前提{entry.presupposition}" if entry.presupposition else "")
            )
            return

    def _decide(self, analysis: Analysis) -> Reply:
        text = analysis.text
        trace = self._trace(analysis)
        modality = analysis.modality.modality

        # -1. 直前に尋ねたことへの答えか。単語だけの返事を受け取る。
        answered = self._try_answer(analysis, trace)
        if answered is not None:
            return answered

        # 0. 会話の口。「草」「なるほど」「おつ」。
        #
        # 構造の解析より先に見る。この手の発話は述語を持たないので
        # 格解析は必ず失敗するが、失敗したことに意味は無い。
        # 会話ではこれが多数を占めるので、突き放さずに受け止める。
        tags = analysis.reasoning.tags if analysis.reasoning else []
        for tag, key in self.ct.style.reaction.items():
            if tag in tags:
                # 相手の言葉をそのまま返せるようにする。
                # 「かわいいね」に「そう思うか」とだけ返すより、
                # 「かわいい、か」と拾った方が聞いている感じになる。
                text = self._template(key).replace(
                    "{focus}", self._focus(analysis, tag)
                )
                return Reply(text, policy=key, trace=trace)

        # 0.5 挨拶。構造は無いが返せる。
        #
        # 会話の口より後ろに置く。「ありがとう」は感動詞なので挨拶にも
        # 当たるが、辞書に「感謝」として載っている方が具体的である。
        # 大きな括りを先に見ると、細かい区別が全部潰れる。
        if modality is Modality.GREETING:
            return Reply(self._template("greeting"), policy="greeting", trace=trace)

        # 0.8 外の世界を触る能力が差し込まれていれば、先に聞く。
        #
        # 「答えられない」と言う各分岐より前に置くこと。後ろに置くと
        # Q_OPEN の分岐で先に返ってしまい、能力が呼ばれない。
        # 何も差し込まれていなければ即座に抜けるので、既定の動きは変わらない。
        if self.capabilities:
            answered = self.capabilities.consult(analysis)
            if answered is not None:
                result, name = answered
                trace.append(
                    f"外部: {name} → {result.source}"
                    + (f" / {result.detail}" if result.detail else "")
                )
                return Reply(result.text, policy=f"external:{name}", trace=trace)

        # 1. 知識を要する問い。答えられないと返す。
        if modality is Modality.Q_OPEN:
            words = "・".join(analysis.modality.interrogatives) or "それ"
            return Reply(
                self._template("q_open").format(interrogative=words),
                policy="q_open", trace=trace,
            )

        # 1.5 知識を差し出せと言われた場合。依頼でも問いでも受けられない。
        #     「教えて」を「承知した」と受けると、実行できない約束をすることになる。
        if self._wants_knowledge(analysis):
            return Reply(
                self._template("no_knowledge"), policy="no_knowledge", trace=trace
            )

        # 2. 肯否の問い。真偽の材料が無いが、無い理由は 3 通りある。
        if modality is Modality.Q_YESNO:
            key = self._yesno_kind(analysis)
            return Reply(self._template(key), policy=key, trace=trace)

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

        # 素性や文型だけのタグは、内容について何も言っていない。
        #
        # 「猫が鳴いた」のタグは #過去 だけで、返せるのは「済んだ話だな」。
        # 文としては正しいが、何の話かに触れていないので中身が無い。
        # こういう場合は構造からの一般規則（6）に回す方が良い返答になる。
        tags = analysis.reasoning.tags if analysis.reasoning else []
        weak_only = bool(tags) and all(t in WEAK_TAGS for t in tags)
        reasoning = "" if weak_only else self._reasoning_text(analysis)

        # 同じ含意を二度言わない。
        #
        # 「作業の話だな。対象が決まらないと動けないなら…」は 1 回目は
        # 中身があるが、作業の話のたびに出ると定型に聞こえる。
        # 既に言った含意しか無いなら、文型から組み立てる側（6）に回す。
        if reasoning and analysis.reasoning is not None:
            fresh = [
                i.tag for i in analysis.reasoning.implications
                if i.tag not in self._used_implications
            ]
            if not fresh:
                reasoning = ""
            else:
                self._used_implications.update(fresh)

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

        # 6. 文型に単語を埋めて返す。
        #
        # ここが返答の中心である。固定文を選ぶのではなく、
        # 「単語」「助詞」「述語」の並びに相手の語を入れ、述語を活用させる。
        # 文型を 1 本足せば、条件を満たす全入力に効く。
        composed = self._compose(analysis, trace)
        if composed is not None:
            return composed

        # 7. 文型が埋まらない場合（活用できない述語など）。
        #    品詞と構造から言えることを探す。
        general = self._from_structure(analysis, trace)
        if general is not None:
            return general

        # 8. 最後の手段。理解した内容を言い直す。
        understood = self.ct.generate(analysis.ir, verbosity=2).text
        return Reply(
            self._template("statement").format(understood=understood),
            policy="statement", trace=trace,
        )

    def _compose(self, analysis: Analysis, trace: list[str]) -> Reply | None:
        """文型から返答を組み立てる。作れなければ None。

        【候補を作ってから 1 つ選ぶ】
        使える文型をすべて適用し、その中から選ぶ。選び方は
        「この会話でまだ使っていない文型を優先する」で、乱数は使わない。
        同じ入力でも会話の状態で変わり、会話をやり直せば同じになる。

        文型が増えるほど候補が増えるので、言い方の数は
        文型 × 埋まっている格 × 活用 の組み合わせで伸びる。
        """
        if not isinstance(analysis.ir, IR) or not analysis.ir.clauses:
            return None
        # 相手が言った語は候補から外す。そのまま返すと言い換えにしかならない。
        avoid = {t.surface for t in analysis.tokens} | {
            t.lemma for t in analysis.tokens
        }
        candidates = self.ct.composer.candidates(
            analysis.ir.clauses[0], avoid=avoid
        )
        if not candidates:
            return None

        # まだ使っていない文型を先に使う。尽きたら使用回数の少ない順。
        chosen = min(candidates, key=lambda c: self._used_patterns[c.pattern_id])
        self._used_patterns[chosen.pattern_id] += 1
        trace.append(
            f"文型: {chosen.pattern_id}（候補 {len(candidates)} 件から選択）"
        )
        return Reply(chosen.text, policy=f"pattern:{chosen.pattern_id}",
                     trace=trace)

    def _focus(self, analysis: Analysis, tag: str) -> str:
        """返答に織り込む相手の言葉を 1 つ選ぶ。

        そのタグを立てた語を返す。「かわいいね」なら「かわいい」。
        辞書由来のタグは語に紐づいているので、どの語が効いたかが分かる。
        見つからなければ空文字（型の側で {focus} が消える）。
        """
        wanted = tag.lstrip("#")
        for token in analysis.tokens:
            if wanted in token.content:
                return token.lemma
        return ""

    def _from_structure(self, analysis: Analysis, trace: list[str]) -> Reply | None:
        """辞書に無い述語でも、品詞と格の埋まり方から返す。

        【なぜ要るか】
        「うるさい」「煩わしい」はフレームに無いので、タグが取れず
        「うるさい、と理解した」で終わっていた。これは何も言っていない。
        語を足せばその語は直るが、次に来る未知語には効かない。

        辞書に載っていなくても、形態素解析器は品詞を教えてくれる。
        形容詞なら「話し手がそう感じている」、動詞なら「何かが起きた」
        までは、語彙を知らなくても言える。そこまでを返す。

        【踏み込まない】
        何の話かは分からないままなので、評価も助言もしない。
        分かったことだけを言い、足りない要素を尋ねる。
        """
        if not isinstance(analysis.ir, IR) or not analysis.ir.clauses:
            return None
        clause = analysis.ir.clauses[0]
        predicate = clause.predicate
        if predicate is None:
            return None
        # フレームを知っている述語は、タグ経由で既に扱われている
        if self.ct.frames.get(predicate.lemma) is not None:
            return None

        subject = clause.slots.get("GA") or clause.topic
        object_ = clause.slots.get("WO")

        # 表層形を使う。見出し語だと「雨が止んだ」が「雨が止む」になり、
        # 過去だったことが消える。活用は作らず、そのまま埋める。
        #
        # ただし丁寧語は見出し語に戻す。「眩しいです」をそのまま埋めると
        # 「眩しいですのか」という壊れた文になる。丁寧さは命題の内容を
        # 変えないので、落としても意味は失われない。
        surface = predicate.lemma if predicate.has(POLITE) else predicate.surface

        if predicate.pos == "形容詞":
            # 感じ方の表明。主語が無ければ何についてかを尋ねる。
            key = "felt" if subject is None else "felt_about"
            text = self._template(key).format(
                predicate=surface,
                subject=subject.surface if subject else "",
            )
            trace.append(f"一般規則: 形容詞（フレーム外）→ {key}")
            return Reply(text, policy=key, trace=trace)

        if predicate.pos in ("名詞", "代名詞"):
            # 名詞述語。「これは本だ」「容量不足だ」。
            key = "identified" if subject is not None else "identified_bare"
            text = self._template(key).format(
                predicate=surface,
                subject=subject.surface if subject else "",
            )
            trace.append(f"一般規則: 名詞述語 → {key}")
            return Reply(text, policy=key, trace=trace)

        if predicate.pos == "動詞":
            if object_ is not None:
                key, filler = "did_to", object_.surface
            elif subject is not None:
                key, filler = "happened_to", subject.surface
            else:
                key, filler = "happened", ""
            text = self._template(key).format(
                predicate=surface, target=filler
            )
            trace.append(f"一般規則: 動詞（フレーム外）→ {key}")
            return Reply(text, policy=key, trace=trace)

        return None

    def _wants_knowledge(self, analysis: Analysis) -> bool:
        """こちらの知識を差し出すことを求められているか。

        「教えて」「説明してください」「教えてほしい」。
        述語の側に印がある（frames.jsonl の knowledge）。

        依頼と願望に限る。平叙の「数学を教えています」は話し手が
        教えている話であって、こちらへの要求ではない。述語だけで
        判定すると、これを断ってしまう。
        """
        if analysis.modality is None:
            return False
        if analysis.modality.modality not in (Modality.REQUEST, Modality.DESIRE):
            return False
        if not isinstance(analysis.ir, IR):
            return False
        return any(
            clause.predicate is not None
            and (frame := self.ct.frames.get(clause.predicate.lemma)) is not None
            and frame.knowledge
            for clause in analysis.ir.clauses
        )

    def _yesno_kind(self, analysis: Analysis) -> str:
        """肯否の問いを 3 つに分ける。

        「その真偽は判断できない」で全部返していたが、答えられない理由は
        同じではない。理由を言い分けた方が、何ができないのかが伝わる。

            行きますか        → 決めるのは相手。意志動詞で見分ける
            明日は雨ですか     → 外界の事実。調べないと出てこない
            これでいいですか   → それ以外。真偽の材料が無い
        """
        tags = analysis.reasoning.tags if analysis.reasoning else []
        if any(tag in OBSERVABLE_TAGS for tag in tags):
            return "q_yesno_fact"
        if isinstance(analysis.ir, IR):
            for clause in analysis.ir.clauses:
                if clause.predicate is None:
                    continue
                frame = self.ct.frames.get(clause.predicate.lemma)
                if frame is not None and frame.volitional:
                    return "q_yesno_volition"
        return "q_yesno"

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
        # カテゴリ辞書に載っていた語。句として取れたものと、
        # 形態素分割の後に見出し語で引いたものの両方を出す。
        known = [t for t in analysis.tokens if t.entry_id]
        if known:
            lines.append(
                "カテゴリ辞書: " + "、".join(
                    f"{t.surface}[{t.entry_id}]"
                    + ("（句）" if t.is_phrase else "")
                    for t in known
                )
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


def _load_capabilities(names: str) -> list:
    """--with で指定された能力を作る。

    既定では何も読まない。ここを既定で有効にすると、このプログラムは
    「ネットワークが要る」「返答が日によって変わる」ものになる。
    差し込むかどうかは使う側が決める。
    """
    made = []
    for name in (n.strip() for n in names.split(",") if n.strip()):
        if name == "clock":
            from cognitag_struct.providers.clock import Clock
            made.append(Clock())
        elif name == "weather":
            from cognitag_struct.providers.weather import Weather
            made.append(Weather())
        elif name == "blender":
            from cognitag_struct.providers.blender import Blender
            blender = Blender()
            if not blender.available:
                print("  Blender が見つからないので差し込みませんでした")
                continue
            made.append(blender)
        else:
            print(f"  知らない能力: {name}（clock / weather / blender）")
    return made


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    capabilities: list = []
    if argv and argv[0].startswith("--with="):
        capabilities = _load_capabilities(argv[0][len("--with="):])
        argv = argv[1:]

    try:
        responder = Responder(capabilities=capabilities)
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
    print(f"  {responder.ct.describe()}")
    if responder.capabilities:
        print(f"  外部の能力: {'、'.join(responder.capabilities.names())}")
    print()

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
