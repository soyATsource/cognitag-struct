"""
会話の文脈。直前のやり取りを覚えておく。

これが無いと会話にならない。

    CogniTag> どこに？
    あなた> 名古屋
    CogniTag> その文は構造として取れなかった      ← 文脈が無い状態

「名古屋」は単独では述語を持たないので構文解析できない。しかし
直前に「どこに？」と尋ねたのなら、これは NI スロットへの答えである。
前の節に埋め戻せば「名古屋に行くんだな」と返せる。

【覚えるものを 4 つに絞る】
    pending   直前に尋ねた空きスロット（何を聞いたか）
    last_ir   直前に解釈できた構造（埋め戻す先）
    topics    最近出た語（指示語の解決と、話題の反復を避けるため）
    turns     発話そのもの（「さっき何て言った」に答えるため）

turns を足したのは、構造だけでは「何と言ったか」に答えられないため。
IR は意味の骨格であって、言われた文そのものではない。復元しようとすると
言い直しになってしまい、「さっきこう言った」の証拠にならない。

ただし全部は持たない。上限を決めて古いものから捨てる。会話の全履歴を
持ち始めると、どこを見ればよいかが決まらなくなる。

【破棄の規則】
pending は 1 回使ったら消す。答えが来たあとも残っていると、
無関係な単語まで前の質問への答えとして取り込んでしまう。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ir import IR, Clause, Token

# 覚えておく話題の数。指示語の解決に使うだけなので少なくてよい。
TOPIC_LIMIT = 8

# 覚えておく発話の数。「さっき何て言った」に答えるためのもので、
# 遡れるのは直近だけでよい。多く持っても、どれを指しているか決まらない。
TURN_LIMIT = 10

# 発話の主。どちらが言ったかを分けて持つ。
YOU = "you"
ME = "me"

# 指示語。直近の話題に置き換える。
DEMONSTRATIVES: frozenset[str] = frozenset(
    {"それ", "これ", "あれ", "そこ", "ここ", "あそこ", "そちら", "こちら"}
)


@dataclass
class Pending:
    """直前に尋ねた空きスロット。"""

    slot: str
    lemma: str
    question: str
    # 尋ねたときの構造。答えを埋め戻す先。
    clause: Clause | None = None


@dataclass
class Context:
    """会話の状態。1 対話につき 1 つ持つ。"""

    pending: Pending | None = None
    last_ir: IR | None = None
    topics: list[Token] = field(default_factory=list)
    # (誰が, 何と言ったか) の並び。古いものから捨てる。
    turns: list[tuple[str, str]] = field(default_factory=list)

    def record(self, speaker: str, text: str) -> None:
        """発話を 1 つ覚える。"""
        if not text:
            return
        self.turns.append((speaker, text))
        if len(self.turns) > TURN_LIMIT:
            self.turns = self.turns[-TURN_LIMIT:]

    def last_said(self, speaker: str) -> str:
        """その人が最後に言ったこと。無ければ空。"""
        for who, text in reversed(self.turns):
            if who == speaker:
                return text
        return ""

    # -- 記録 -------------------------------------------------------------

    def remember(self, ir: IR | None, tokens: list[Token]) -> None:
        """1 ターン分を覚える。"""
        if isinstance(ir, IR):
            self.last_ir = ir
        for token in tokens:
            if self._is_topic(token):
                self._push_topic(token)

    @staticmethod
    def _is_topic(token: Token) -> bool:
        """話題になりうる語か。指示語そのものは覚えない。"""
        if token.lemma in DEMONSTRATIVES:
            return False
        return token.pos in ("名詞", "代名詞") or token.is_phrase

    def _push_topic(self, token: Token) -> None:
        # 同じ語が続いても 1 つに保つ。最新を末尾にする。
        self.topics = [t for t in self.topics if t.lemma != token.lemma]
        self.topics.append(token)
        if len(self.topics) > TOPIC_LIMIT:
            self.topics = self.topics[-TOPIC_LIMIT:]

    def ask(self, slot: str, lemma: str, question: str, clause: Clause) -> None:
        """尋ねたことを記録する。次の入力を答えとして受け取れるようにする。"""
        self.pending = Pending(
            slot=slot, lemma=lemma, question=question, clause=clause
        )

    def clear_pending(self) -> None:
        """使ったら消す。残しておくと無関係な語まで取り込む。"""
        self.pending = None

    # -- 参照 -------------------------------------------------------------

    @property
    def latest_topic(self) -> Token | None:
        return self.topics[-1] if self.topics else None

    def resolve_demonstrative(self, tokens: list[Token]) -> Token | None:
        """指示語が指しているものを直近の話題から返す。

        「それはどう」の「それ」を、直前に出た語に置き換えるために使う。
        分からなければ None。無理に当てない。
        """
        if not any(t.lemma in DEMONSTRATIVES for t in tokens):
            return None
        return self.latest_topic

    def answer_fills_pending(self, tokens: list[Token]) -> Token | None:
        """この入力が直前の質問への答えかを判定する。

        条件は 2 つ。
          1. 直前に何かを尋ねている
          2. 入力が述語を持たない（単語だけの返事）

        述語があるなら独立した発話として扱う。「名古屋」は答えだが、
        「名古屋は遠い」は新しい話である。
        """
        if self.pending is None:
            return None
        if any(t.pos in ("動詞", "形容詞", "形状詞") for t in tokens):
            return None
        for token in tokens:
            if token.pos in ("名詞", "代名詞") or token.is_phrase:
                return token
        return None

    def filled_clause(self, answer: Token) -> Clause | None:
        """答えを直前の節に埋め戻した節を返す。

        元の節は書き換えない。埋め戻した写しを返す。
        破壊的に書き換えると、同じ節を後から別の用途で使えなくなる。
        """
        if self.pending is None or self.pending.clause is None:
            return None
        original = self.pending.clause
        return Clause(
            predicate=original.predicate,
            slots={**original.slots, self.pending.slot: answer},
            modifiers=list(original.modifiers),
            topic=original.topic,
            degree=original.degree,
            supplements=list(original.supplements),
        )

    def reset(self) -> None:
        self.pending = None
        self.last_ir = None
        self.topics.clear()
