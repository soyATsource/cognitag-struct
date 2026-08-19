"""
文末表現から発話の種類を判定する。

【辞書に依存しない層である】
判定に使うのは形態素の品詞・活用形・見出し語だけで、Facet も
カテゴリ辞書も参照しない。したがって coverage の影響を受けず、
どんな入力にも必ず答えが出る。

これが重要なのは、辞書ベースの判定が実測で 22.5% の発話しか
担当できなかったのに対し、モダリティは 100% に適用できるからである。
「疲れた」に共感するか、「作って」を受領するかは、physical や
temporal の値を必要としない。

【parse() と独立させた理由】
入力は list[Token] であって IR ではない。parse() は 3 節以上の文を
Unparsable として返すが、そういう文にもモダリティの判定は要る。
「雨が降ったら友人に会いに名古屋へ行きますか」は構文解析を諦めても
質問であることは分かるし、そこで応答方針を決められなければならない。

【判定の順序に意味がある】
先に判定したものが勝つ。順序を変えると結果が変わる。

    REQUEST     依頼が最優先。「手伝ってくれますか」は疑問形だが依頼である
    SPECULATION 「行くかな」の「か」を疑問と取り違えないよう、Q より先に見る
    Q_OPEN      疑問詞つきの問い。答えを知らないと返せない
    Q_YESNO     肯否を問う
    DESIRE      願望の表明
    STATEMENT   上のいずれでもない
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .ir import NEGATIVE, PAST, POLITE, Token


class Modality(str, Enum):
    """発話の種類。応答方針を決める主要な手がかり。"""

    GREETING = "GREETING"        # こんにちは / おはよう（感動詞）
    REQUEST = "REQUEST"          # 〜して / 〜してください / お願い
    VOLITION = "VOLITION"        # 〜しよう / どうしよう（意志・勧誘・思案）
    SPECULATION = "SPECULATION"  # 〜だろう / 〜かな / 〜と思う / 〜らしい
    Q_OPEN = "Q_OPEN"            # 疑問詞つきの問い
    Q_YESNO = "Q_YESNO"          # 肯否を問う
    DESIRE = "DESIRE"            # 〜たい / 〜ほしい
    STATEMENT = "STATEMENT"      # 平叙


# 対人態度。モダリティとは独立に付く。
AGREEMENT = "#同意要求"   # ね
INFORMING = "#情報提供"   # よ

# --- 判定に使う手がかり ---------------------------------------------------

# 依頼の中核。命令形か、この見出し語が現れたら依頼とみなす。
#   ください → くださる（命令形）
#   てくれる / てもらえる / ていただける
REQUEST_LEMMAS: frozenset[str] = frozenset(
    {"くださる", "くれる", "もらえる", "いただける", "願う"}
)
IMPERATIVE = "命令形"

# 願望。「たい」は助動詞、「ほしい」は形容詞として現れる。
DESIRE_LEMMAS: frozenset[str] = frozenset({"たい", "ほしい"})

# 「意志推量形」は名前のとおり意志と推量の両方に使われる活用形で、
# これだけでは区別がつかない。品詞で分ける。
#
#     だろう  助動詞 + 意志推量形  → 推量（雨が降るだろう）
#     しよう  動詞   + 意志推量形  → 意志（どうしよう / 何しようか）
#
# 「どうしよう」を推量と判定すると「かもしれないな」と返してしまう。
# あれは思案や困惑であって、推量ではない。
VOLITIONAL_PRESUMPTIVE = "意志推量形"

# 伝聞・推定。「らしい」「そうだ」「ようだ」。
# 話者自身の判断ではなく、外から得た情報であることを示す。
HEARSAY_LEMMAS: frozenset[str] = frozenset({"らしい", "そうだ", "ようだ", "みたいだ"})

# 挨拶。感動詞は述語を持たないので構文解析できない。
# 「こんばんは」に「構造として取れなかった」と返すのは論外なので、
# モダリティの段階で拾う。
INTERJECTION_POS = "感動詞"
# 「と思う」の並び
THINK_LEMMAS: frozenset[str] = frozenset({"思う", "考える"})

# 疑問詞。これがあるかどうかで Q_OPEN と Q_YESNO を分ける。
INTERROGATIVE_LEMMAS: frozenset[str] = frozenset(
    {
        "どこ", "誰", "だれ", "何", "なに", "いつ", "どちら", "どれ",
        "どの", "どんな", "いくつ", "いくら", "なぜ", "どう", "どうして",
        # 話し言葉の形。「なんで」「なん」「どっち」「だれ」。
        "なんで", "なん", "どっち", "どこら", "いつごろ",
    }
)

# 終助詞
FINAL_QUESTION = "か"
FINAL_AGREEMENT = "ね"
FINAL_INFORMING = "よ"
FINAL_SOFTENER = "な"
FINAL_EXPLANATORY = "の"

FINAL_PARTICLE_SUBPOS = "終助詞"
# 「〜は」で言い終わる問い。述語を相手に埋めさせる形。
TOPIC_PARTICLE = "は"
QUESTION_MARKS = ("?", "？")

# 「教えて」「手伝って」の依頼を見るための手がかり。
CONJUNCTIVE_PARTICLE = "接続助詞"
TE_SURFACES: frozenset[str] = frozenset({"て", "で"})

# 「何時」「何人」「何回」のように、疑問詞が名詞に取り込まれる形。
#
# Sudachi は「何人です」を 1 語の名詞（見出し語 何人）として返すので、
# 見出し語の一致では疑問詞「何」が見えない。結果として
# 「東京の人口は何人ですか」が肯否の問いと判定されていた。
INTERROGATIVE_PREFIX = "何"


@dataclass
class ModalityResult:
    """判定結果。"""

    modality: Modality
    attitude: list[str] = field(default_factory=list)
    # 出現した疑問詞。Q_OPEN のとき「何を尋ねられたか」が分かる。
    interrogatives: list[str] = field(default_factory=list)
    # 述語に付いていた素性のまとめ。応答の語調に効く。
    negative: bool = False
    past: bool = False
    polite: bool = False
    # 判定根拠。誤判定を追えるようにしておく。
    evidence: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        tags = "".join(self.attitude)
        return f"{self.modality.value}{tags}"


# --- 個々の手がかりの検出 -------------------------------------------------


def _final_particles(tokens: list[Token]) -> list[Token]:
    """末尾に連続する終助詞。「かな」のように 2 つ続くことがある。"""
    found: list[Token] = []
    for token in reversed(tokens):
        if token.subpos == FINAL_PARTICLE_SUBPOS:
            found.insert(0, token)
            continue
        break
    return found


def _has_question_mark(text: str) -> bool:
    return any(mark in text for mark in QUESTION_MARKS)


def _is_request(tokens: list[Token]) -> str | None:
    for token in tokens:
        if token.inflection.startswith(IMPERATIVE):
            return f"命令形: {token.surface}"
        if token.lemma in REQUEST_LEMMAS:
            return f"依頼の語: {token.surface}"
    return _is_te_request(tokens)


def _is_te_request(tokens: list[Token]) -> str | None:
    """「教えて」「手伝って」のように接続助詞「て」で言い切る依頼。

    日本語の依頼で最も普通の形だが、命令形でも依頼の語でもないので
    上の判定に掛からず、平叙文として扱われていた。

    見るのは文末だけにする。
        教えて          → 末尾が接続助詞「て」。依頼
        教えています     → 「て」の後に用言が続く。依頼ではない
        走って帰る       → 同上。連用の「て」
    「ね」「よ」が付く場合（教えてね）があるので終助詞は除いて見る。
    """
    body = [t for t in tokens if t.subpos != FINAL_PARTICLE_SUBPOS]
    if len(body) < 2:
        return None
    last, previous = body[-1], body[-2]
    if (
        last.subpos == CONJUNCTIVE_PARTICLE
        and last.surface in TE_SURFACES
        and previous.pos == "動詞"
    ):
        return f"て形の依頼: {previous.surface}{last.surface}"
    return None


def _is_desire(tokens: list[Token]) -> str | None:
    for token in tokens:
        if token.lemma in DESIRE_LEMMAS:
            return f"願望の語: {token.surface}"
    return None


def _is_greeting(tokens: list[Token]) -> str | None:
    """感動詞だけで成り立つ発話か。"""
    content = [t for t in tokens if t.subpos != FINAL_PARTICLE_SUBPOS]
    if content and all(t.pos == INTERJECTION_POS for t in content):
        return f"感動詞: {content[0].surface}"
    return None


def _is_volition(tokens: list[Token]) -> str | None:
    """意志・勧誘・思案か。「しよう」「行こう」「どうしよう」。"""
    for token in tokens:
        if token.pos == "動詞" and token.inflection.startswith(
            VOLITIONAL_PRESUMPTIVE
        ):
            return f"動詞の意志推量形: {token.surface}"
    return None


def _is_speculation(tokens: list[Token], finals: list[Token]) -> str | None:
    for token in tokens:
        # 助動詞の意志推量形（だろう）だけを推量とする。
        # 動詞の意志推量形（しよう）は意志なので _is_volition が先に拾う。
        if token.pos == "助動詞" and token.inflection.startswith(
            VOLITIONAL_PRESUMPTIVE
        ):
            return f"助動詞の意志推量形: {token.surface}"
        if token.lemma in HEARSAY_LEMMAS:
            return f"伝聞: {token.surface}"
    # 「か」+「な」で終わるのは疑問ではなく推量。
    # これを Q_YESNO より先に見ないと「行くかな」を質問と取り違える。
    surfaces = [t.surface for t in finals]
    if FINAL_QUESTION in surfaces and FINAL_SOFTENER in surfaces:
        return "終助詞: かな"
    # 「〜と思う」
    for index in range(len(tokens) - 1):
        if tokens[index].surface == "と" and tokens[index + 1].lemma in THINK_LEMMAS:
            return f"と{tokens[index + 1].surface}"
    # 「かもしれない」。parse.py の #不確定 と同じ並びを見る。
    for index in range(len(tokens) - 2):
        if (
            tokens[index].surface == "か"
            and tokens[index + 1].surface == "も"
            and tokens[index + 2].lemma == "しれる"
        ):
            return "かもしれない"
    return None


def _interrogatives(tokens: list[Token]) -> list[Token]:
    return [
        t for t in tokens
        if t.lemma in INTERROGATIVE_LEMMAS
        or (
            t.pos == "名詞"
            and len(t.lemma) > 1
            and t.lemma.startswith(INTERROGATIVE_PREFIX)
        )
    ]


def _is_question(tokens: list[Token], finals: list[Token], text: str) -> str | None:
    if any(t.surface == FINAL_QUESTION for t in finals):
        return "終助詞: か"
    if _has_question_mark(text):
        return "疑問符"
    # 終助詞「の」で終わる形。「なんでそうなるの」「何時までやるの」。
    #
    # 「の」には説明の平叙（「行くの」＝行くのだ）もあるが、話し言葉では
    # 問いの方が多い。疑問詞が同席していればほぼ問いなので、
    # 疑問詞がある場合に限って問いとみなす。
    #     なんでそうなるの → 問い（疑問詞あり）
    #     もう帰るの       → 問いとみなさない（疑問詞なし。断定もありうる）
    # 「の」は終助詞として付くこともあれば、準体助詞として付くこともある。
    # 「なぜ空は青いの」の「の」は準体助詞なので finals に入らない。
    # 末尾の語そのものを見る。
    body = [t for t in tokens if t.pos != "補助記号"]
    ends_with_no = bool(body) and body[-1].surface == FINAL_EXPLANATORY
    if ends_with_no and _interrogatives(tokens):
        return "「の」＋疑問詞"

    # 「富士山の高さは」「日本の首都は」。
    #
    # 係助詞「は」で言い終わる形。述語を相手に埋めさせる言い方で、
    # 会話では問いとして働く。終助詞も疑問符も無いのでここで拾う。
    if body and body[-1].surface == TOPIC_PARTICLE and body[-1].subpos == "係助詞":
        return "「は」で言い終わる"
    return None


def _attitude(finals: list[Token]) -> list[str]:
    tags: list[str] = []
    for token in finals:
        if token.surface == FINAL_AGREEMENT and AGREEMENT not in tags:
            tags.append(AGREEMENT)
        if token.surface == FINAL_INFORMING and INFORMING not in tags:
            tags.append(INFORMING)
    return tags


# --- 本体 -----------------------------------------------------------------


def detect_modality(tokens: list[Token], text: str = "") -> ModalityResult:
    """Token 列から発話の種類を判定する。

    text は疑問符の検出にだけ使う。省略しても動く（終助詞で判定する）。
    """
    if not tokens:
        return ModalityResult(
            modality=Modality.STATEMENT, evidence=["入力が空"]
        )

    finals = _final_particles(tokens)
    interrogatives = _interrogatives(tokens)
    predicate_features = {f for t in tokens for f in t.features}

    result = ModalityResult(
        modality=Modality.STATEMENT,
        attitude=_attitude(finals),
        # 表層形ではなく見出し語を持つ。「どうですか」の表層形は
        # 「どうです」で、そのまま返答に埋めると「どうですを知る必要がある」
        # という壊れた文になる。
        interrogatives=[t.lemma for t in interrogatives],
        negative=NEGATIVE in predicate_features,
        past=PAST in predicate_features,
        polite=POLITE in predicate_features,
    )

    # 順序に意味がある。冒頭の説明を参照。
    reason = _is_greeting(tokens)
    if reason:
        result.modality = Modality.GREETING
        result.evidence.append(reason)
        return result

    reason = _is_request(tokens)
    if reason:
        result.modality = Modality.REQUEST
        result.evidence.append(reason)
        return result

    # 意志は推量より先。活用形が同じ（意志推量形）なので、
    # 品詞で分けたうえで意志を優先しないと「どうしよう」が推量になる。
    reason = _is_volition(tokens)
    if reason:
        result.modality = Modality.VOLITION
        result.evidence.append(reason)
        return result

    reason = _is_speculation(tokens, finals)
    if reason:
        result.modality = Modality.SPECULATION
        result.evidence.append(reason)
        return result

    reason = _is_question(tokens, finals, text)
    if reason:
        if interrogatives:
            result.modality = Modality.Q_OPEN
            result.evidence.append(f"{reason} + 疑問詞{result.interrogatives}")
        else:
            result.modality = Modality.Q_YESNO
            result.evidence.append(reason)
        return result

    reason = _is_desire(tokens)
    if reason:
        result.modality = Modality.DESIRE
        result.evidence.append(reason)
        return result

    result.evidence.append("該当なし")
    return result


def needs_knowledge(result: ModalityResult) -> bool:
    """外部知識が要る問いか。

    Q_OPEN は「何を / なぜ / どこ」を尋ねるもので、辞書と構文だけでは
    答えを作れない。ルーティングでは LLM へ回す判断に使う。
    """
    return result.modality is Modality.Q_OPEN
