"""
外から使うための窓口。

【なぜ窓口が要るか】
各モジュールはパスを引数で受け取る設計にしてある（テストで差し替えられる
ようにするため）。しかし利用者に 5 つのデータファイルのパスを毎回渡させる
のは酷である。同梱データを既定で読む入口をひとつ用意する。

    from cognitag_struct import CogniTag

    ct = CogniTag()
    result = ct.analyze("明日は名古屋に行きません")
    result.modality.modality      # Modality.STATEMENT
    result.ir.clauses[0].slots     # {'NI': Token(名古屋)}
    result.questions()             # []

URL も設定ファイルも要らない。pip install -e で入れておけば、
他のプロジェクトからも同じように import できる。

【HTTP を挟まない理由】
CogniTag v2 の辞書は HTTP API（:8010）を持つが、こちらは持たない。
構造層は 0.1 ミリ秒級で終わる処理なので、HTTP の往復（1 ミリ秒以上）が
支配的になってしまう。同一プロセスで呼ぶ方が速く、依存も少ない。
サーバの起動忘れという運用の失敗要因も消える。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .category_dict import CategoryDict, Entry
from .compose import Composer, PatternSet
from .inflect import Conjugator
from .facts import FactTable
from .frames import FrameDict
from .gap import GapReport, QuestionTemplates, detect
from .generate import Style, Utterance, generate
from .idiom import choose_idiom
from .ir import IR, Token
from .modality import ModalityResult, detect_modality, needs_knowledge
from .parse import Unparsable, parse
from .phrase_dict import PhraseDict
from .reasoning import ImplicationTable, Reasoning, reason
from .selfdesc import SelfDescription
from .tokenizer import Tokenizer

# 同梱データの場所。パッケージと一緒に配られる。
# importlib.resources を使わないのは、editable install でも wheel でも
# この相対パスがそのまま通り、余計な分岐が要らないため。
DATA_DIR = Path(__file__).resolve().parent / "data"

AXES = "axes.jsonl"
PHRASES = "phrases.jsonl"
FRAMES = "frames.jsonl"
QUESTIONS = "questions.toml"
STYLE = "generation_style.toml"
REASONING = "reasoning.toml"

# 内容語の品詞。カバー率の分母になる。
# 助詞・助動詞・記号は辞書に載る種類の語ではないので数えない。
CONTENT_POS: frozenset[str] = frozenset({"名詞", "動詞", "形容詞", "形状詞", "副詞"})
CONJUGATION = "conjugation.toml"
PATTERNS = "patterns.jsonl"
SELF = "self.toml"
FACTS = "facts.jsonl"


@dataclass
class Analysis:
    """1 入力分の解析結果。

    ir は解釈できなかった場合 Unparsable になる。modality は
    どんな入力にも必ず入る（辞書にも構文解析にも依存しないため）。
    """

    text: str
    tokens: list[Token] = field(default_factory=list)
    modality: ModalityResult | None = None
    ir: IR | Unparsable | None = None
    gaps: GapReport | None = None
    # 入力から取れたタグと、そこから言えること。
    # 構文解析に失敗してもモダリティ由来のタグは入る。
    reasoning: Reasoning | None = None
    # 述語として説明できる見出し語。カバー率の判定に使う。
    known_predicates: frozenset[str] = frozenset()

    @property
    def parsed(self) -> bool:
        return isinstance(self.ir, IR)

    def content_words(self) -> list[Token]:
        """内容語だけ。助詞・助動詞・記号は数えない。"""
        return [t for t in self.tokens if t.pos in CONTENT_POS]

    def unknown_words(self) -> list[str]:
        """辞書に無かった内容語。

        【黙れることを可視化する】
        この方式の取り柄は「知らないことを知らないと言える」ことにある。
        ところが今までは、知らない語も知っている語も同じ顔で通っていた。
        既知として扱ったのか、何も付かなかっただけなのかが区別できないと、
        「間違えない」という主張を確かめようがない。

        既知とみなす根拠は 2 つだけ。カテゴリ辞書に載っているか、
        述語のフレームがあるか。どちらも無ければ未知と申告する。
        """
        found: list[str] = []
        for token in self.content_words():
            if token.entry_id or token.content:
                continue
            if token.lemma in self.known_predicates:
                continue
            if token.lemma not in found:
                found.append(token.lemma)
        return found

    @property
    def coverage(self) -> float:
        """内容語のうち、辞書で説明できた割合。内容語が無ければ 1.0。"""
        words = self.content_words()
        if not words:
            return 1.0
        return 1.0 - len(self.unknown_words()) / len(words)

    @property
    def needs_knowledge(self) -> bool:
        """外部知識が要る問いか。ルーティングで LLM へ回す判断に使う。"""
        return self.modality is not None and needs_knowledge(self.modality)

    def questions(self) -> list[str]:
        """埋まっていない必須スロットを尋ねる文。無ければ空。"""
        return self.gaps.questions() if self.gaps else []

    def coverage_note(self) -> str:
        """何がどこまで分かったかの一行要約。ログに出す用。"""
        modality = self.modality.modality.value if self.modality else "?"
        clauses = len(self.ir.clauses) if self.parsed else 0
        state = "解析済" if self.parsed else "解釈不能"
        tags = self.reasoning.summary() if self.reasoning else "-"
        unknown = self.unknown_words()
        return (
            f"{modality} / {state} / 節{clauses} / "
            f"質問{len(self.questions())} / {tags} / "
            f"被覆{self.coverage:.0%}"
            + (f" 未知{unknown}" if unknown else "")
        )


class CogniTag:
    """構造層の窓口。同梱データを既定で読む。

    data_dir を渡せば差し替えられる（テストや、語彙を入れ替えた
    実験のため）。渡さなければパッケージ同梱のものを使う。
    """

    def __init__(self, data_dir: str | Path | None = None) -> None:
        base = Path(data_dir) if data_dir else DATA_DIR
        self.data_dir = base

        self.category_dict = CategoryDict.load(base / AXES, base / PHRASES)
        self.phrase_dict = PhraseDict(self.category_dict)
        self.tokenizer = Tokenizer(self.category_dict, self.phrase_dict)
        self.frames = FrameDict.load(base / FRAMES)
        self.questions = QuestionTemplates.load(base / QUESTIONS)
        self.style = Style.load(base / STYLE)
        self.implications = ImplicationTable.load(base / REASONING)
        # 返答を組み立てる側。活用表と文型。
        # 固定文を並べるのではなく、単語と助詞の並びから作る。
        self.conjugator = Conjugator.load(base / CONJUGATION)
        # 自分について書いたもの。世界の知識とは別物で、有限。
        self.selfdesc = SelfDescription.load(base / SELF)
        # よく聞かれる事実。百科事典ではなく、答えを用意したものだけ。
        self.facts = FactTable.load(base / FACTS)
        # 関連語を引くために辞書とフレームも渡す。
        # 「行く」の NI は場所、という宣言をフレームが持ち、
        # その場所カテゴリの語をカテゴリ辞書が持っている。
        self.composer = Composer(
            PatternSet.load(base / PATTERNS), self.conjugator,
            category_dict=self.category_dict, frames=self.frames,
        )

    # -- 解析 -------------------------------------------------------------

    def analyze(self, text: str) -> Analysis:
        """トークナイズから欠落検出までを一度に行う。

        構文解析に失敗しても modality は必ず入る。呼び出し側は
        parsed を見て分岐すればよく、例外を捕まえる必要はない。
        """
        tokens = self.tokenizer.tokenize(text)
        result = Analysis(
            text=text,
            tokens=tokens,
            modality=detect_modality(tokens, text),
            known_predicates=frozenset(self.frames.frames),
        )
        result.ir = parse(tokens, source_text=text, frames=self.frames)
        if isinstance(result.ir, IR):
            result.gaps = detect(result.ir, self.frames, self.questions)
        # タグは構文解析の成否に関わらず集める。
        # 解釈不能でもモダリティ由来のタグは付く。
        result.reasoning = reason(
            result.ir if isinstance(result.ir, IR) else None,
            result.modality,
            self.frames,
            self.implications,
            tokens=tokens,
        )
        return result

    def tokenize(self, text: str) -> list[Token]:
        return self.tokenizer.tokenize(text)

    def modality_of(self, text: str) -> ModalityResult:
        """モダリティだけ知りたい場合の近道。構文解析を走らせない。"""
        return detect_modality(self.tokenizer.tokenize(text), text)

    # -- 生成 -------------------------------------------------------------

    def generate(self, ir: IR, verbosity: int = 1) -> Utterance:
        """IR を文にする。verbosity は範囲外でも丸められる。"""
        return generate(ir, verbosity, self.style)

    def say(self, text: str, verbosity: int = 1) -> str | None:
        """解析して言い直す。往復の確認や言い換えに使う。

        解釈できなかった場合は None を返す（例外は投げない）。
        """
        analysis = self.analyze(text)
        if not analysis.parsed:
            return None
        return self.generate(analysis.ir, verbosity).text

    # -- 慣用句 -----------------------------------------------------------

    def idiom(
        self,
        form: str,
        content: str,
        situation: list[str],
        target: dict[str, float] | None = None,
    ) -> Entry | None:
        """状況に合う慣用句を 1 件返す。前提条件を満たさないものは選ばない。"""
        return choose_idiom(
            self.category_dict, form, content, situation, target or {}
        )

    # -- 情報 -------------------------------------------------------------

    @property
    def phrase_count(self) -> int:
        return len(self.category_dict)

    @property
    def analyzer_available(self) -> bool:
        """形態素解析器（SudachiPy）が使えるか。"""
        return self.tokenizer.available

    def describe(self) -> str:
        """読み込んだ資源の一行要約。起動ログに出す用。"""
        return (
            f"cognitag_struct: 句{self.phrase_count}件 / "
            f"フレーム{len(self.frames)}件 / "
            f"含意{len(self.implications)}件 / "
            f"Sudachi={'有' if self.analyzer_available else '無'} / "
            f"data={self.data_dir}"
        )
