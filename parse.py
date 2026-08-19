"""
Token 列を Clause / IR に組み上げる。

目的は「文の何が埋まっていて、何が空いているか」を判定できる形にすること。
埋まっているかの判定は gap.py が行う。ここは構造を作るだけ。

【格助詞だけを格スロットに入れる】
「は」（係助詞）は格ではない。主題・対比を示すもので、
「名古屋には行きません」の含意（他の場所は違うかもしれない）は
格と独立に保持しないと表現できない。したがって Clause.topic に置く。

格と主題は排他ではない。「名古屋には」は 名古屋 / に / は の 3 トークンで、
名古屋は NI スロットに入ると同時に topic にもなる。

【3 節以上は解釈しない】
規則ベースは節の組合せで破綻する。上限を設けて、超えたら
解釈不能を明示的に返す。例外ではなく戻り値で表すのは、
呼び出し側が「解釈できなかった」を通常の分岐として扱えるようにするため。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .frames import FrameDict
from .ir import IR, CONDITION, NONE, POLITE, PURPOSE, Clause, Token

# 格助詞の表層形 -> スロット名
CASE_PARTICLES: dict[str, str] = {
    "が": "GA",
    "を": "WO",
    "に": "NI",
    "へ": "HE",
    "から": "KARA",
    "まで": "MADE",
}

# 節全体にかかる修飾タグ
UNCERTAIN = "#不確定"
HYPOTHETICAL = "#仮定"
QUESTION = "#質問"

# 形状詞（な形容詞）も述語になる。「好き」「簡単」「静か」。
# 含めないと「名古屋が好き」が「述語が無い」になる。
PREDICATE_POS = ("動詞", "形容詞", "形状詞")
MAX_CLAUSES = 2

# 名詞述語の判定に使う。「これは本だ」「不安です」。
AUXILIARY_POS = "助動詞"
COPULA = "だ"

# 「かもしれない」を構成する並び。実測では以下の 3 形態素に割れる。
#     か(助詞/副助詞) + も(助詞/係助詞) + しれる(動詞/一般)
# 「しれる」を用言として数えると「明日名古屋に行くかもしれません」が
# 2 節になってしまう。これは推量の助動詞相当なので、用言から外して
# #不確定 に畳む。
UNCERTAIN_LEMMA = "しれる"


@dataclass
class Unparsable:
    """解釈しないと決めた入力。例外ではなく戻り値で表す。"""

    reason: str
    source_text: str = ""

    def __bool__(self) -> bool:
        return False


# 副詞可能名詞であることの印。Sudachi の品詞 3 番目に入る。
#
# 以前は見出し語を Sudachi へ引き直して調べていたが、
# Token が subpos2 を持つようになったので不要になった。
ADVERBIAL_SUBPOS2 = "副詞可能"


def is_adverbial_noun(token: Token) -> bool:
    """「明日」「今日」「来週」のような副詞可能名詞か。

    これらは助詞を伴わずに時を表す。裸で現れても格ではないので、
    格スロットにも入れないし、助詞の補完対象にもしない。
    「明日名古屋に行く」の「明日」を GA に入れると主語を誤る。
    """
    return token.pos == "名詞" and token.subpos2 == ADVERBIAL_SUBPOS2


# --- 前処理 -----------------------------------------------------------------


def find_uncertain(tokens: list[Token]) -> set[int]:
    """「か + も + しれる」の並びを探し、その位置を返す。

    見つかった「しれる」は用言として数えない。
    """
    found: set[int] = set()
    for index in range(len(tokens) - 2):
        first, second, third = tokens[index : index + 3]
        if (
            first.surface == "か"
            and second.surface == "も"
            and third.lemma == UNCERTAIN_LEMMA
        ):
            found.update({index, index + 1, index + 2})
    return found


def _is_auxiliary_verb(tokens: list[Token], index: int) -> bool:
    """補助動詞か。「作ってください」の「ください」など。

    「て（接続助詞）＋ 非自立可能の動詞」の並びは 1 つの述語をなす。
    別の用言として数えると「作ってください」が 2 節になり、
    「ください」を述語とする節ができてしまう。

    トークン側で畳まないのは、モダリティ判定が「ください」の
    命令形や「くれる」の見出し語を手がかりにしているため。
    畳むとその手がかりが消えて依頼を検出できなくなる。
    節を数えるここだけで除けばよい。

    なお「会いに行きます」の「行きます」は直前が「に」なので
    この条件に当たらない。目的の節分割は保たれる。
    """
    if index == 0:
        return False
    token = tokens[index]
    previous = tokens[index - 1]
    return (
        token.pos == "動詞"
        and token.subpos == "非自立可能"
        and previous.subpos == "接続助詞"
        and previous.surface == "て"
    )


def predicate_indices(tokens: list[Token], skip: set[int]) -> list[int]:
    found = [
        i
        for i, token in enumerate(tokens)
        if token.pos in PREDICATE_POS
        and i not in skip
        and not _is_auxiliary_verb(tokens, i)
    ]
    if found:
        return found
    # 用言が 1 つも無い場合に限り、名詞述語を探す。
    #
    # 「これは本だ」「不安です」「容量不足だ」は日本語の普通の文だが、
    # 述語が名詞なので用言だけを見ていると「述語が無い」になる。
    # 用言がある文では名詞を述語にしないので、既存の解析は変わらない。
    noun = _copula_predicate(tokens, skip)
    return [noun] if noun is not None else []


def _copula_predicate(tokens: list[Token], skip: set[int]) -> int | None:
    """名詞述語の位置。無ければ None。

    条件は 2 つのどちらか。
        丁寧の「です」が畳まれている  「不安です」（POLITE が付く）
        直後が断定の「だ」            「これは本だ」

    末尾の名詞に限る。「本を読む」の「本」を述語にしないため。
    終助詞と記号は末尾判定から除く（「本だよ」「本です。」）。
    """
    body = [
        i for i, t in enumerate(tokens)
        if t.subpos != "終助詞" and t.pos != "補助記号"
    ]
    if not body:
        return None
    last = body[-1]
    token = tokens[last]

    if last in skip or token.pos not in ("名詞", "代名詞"):
        # 「本だ」は 本(名詞) + だ(助動詞) に割れるので、末尾が
        # 断定の助動詞なら 1 つ前の名詞を見る。
        if token.pos == AUXILIARY_POS and token.lemma == COPULA and len(body) >= 2:
            previous = body[-2]
            if tokens[previous].pos in ("名詞", "代名詞") and previous not in skip:
                return previous
        return None

    return last if token.has(POLITE) else None


# --- 本体 -------------------------------------------------------------------


def parse(
    tokens: list[Token], source_text: str = "", frames: FrameDict | None = None
) -> IR | Unparsable:
    """Token 列を IR にする。解釈しない場合は Unparsable を返す。"""
    if not tokens:
        return Unparsable("入力が空", source_text)

    uncertain = find_uncertain(tokens)
    indices = predicate_indices(tokens, uncertain)

    if not indices:
        return Unparsable("述語が無い", source_text)
    if len(indices) > MAX_CLAUSES:
        return Unparsable(
            f"{len(indices)} 節は解釈しない（上限 {MAX_CLAUSES} 節）", source_text
        )

    clauses = _build_clauses(tokens, indices, uncertain, frames)
    relation = _relation(tokens, clauses, indices, frames)
    return IR(clauses=clauses, relation=relation, source_text=source_text)


def _build_clauses(
    tokens: list[Token],
    indices: list[int],
    uncertain: set[int],
    frames: FrameDict | None,
) -> list[Clause]:
    """述語の位置で節を切り、各節に格要素を割り当てる。"""
    clauses: list[Clause] = []
    start = 0
    for position, predicate_index in enumerate(indices):
        end = predicate_index
        # 述語直後の条件表現はその節に含める。
        #   たら / なら … 仮定形の助動詞
        #   ば          … 接続助詞（述語自身が仮定形になり、助動詞は付かない）
        # 含めないと「雨が降れば」の「ば」が次の節に落ち、
        # 条件節に #仮定 が付かなくなる。
        if end + 1 < len(tokens) and _is_conditional_marker(tokens[end + 1]):
            end += 1
        is_last = position == len(indices) - 1
        if is_last:
            end = len(tokens) - 1

        clause = _build_clause(
            tokens, start, end, tokens[predicate_index], uncertain, frames
        )
        clauses.append(clause)
        start = end + 1
    return clauses


def _build_clause(
    tokens: list[Token],
    start: int,
    end: int,
    predicate: Token,
    uncertain: set[int],
    frames: FrameDict | None,
) -> Clause:
    clause = Clause(predicate=predicate)
    span = list(range(start, end + 1))

    # 格助詞・係助詞の直前の語を割り当てる
    assigned: set[int] = set()
    for i in span:
        token = tokens[i]
        if token.pos != "助詞" or i == start:
            continue

        if token.subpos == "格助詞" and token.surface in CASE_PARTICLES:
            head = tokens[i - 1]
            if head.pos in ("名詞", "代名詞"):
                clause.slots[CASE_PARTICLES[token.surface]] = head
                assigned.add(i - 1)
        elif token.subpos == "係助詞" and token.surface == "は":
            # 「は」は格ではない。格と排他でもないので、
            # 「名古屋には」は NI に入りつつ topic にもなる。
            #
            # 「には」では 名古屋 / に / は の 3 トークンになり、
            # 「は」の直前は格助詞である。格助詞を読み飛ばして
            # その前の名詞を主題として取る。
            head_index = i - 1
            if tokens[head_index].subpos == "格助詞":
                head_index -= 1
            if head_index >= 0 and tokens[head_index].pos in ("名詞", "代名詞"):
                clause.topic = tokens[head_index]
                assigned.add(head_index)

    _assign_bare_nouns(tokens, span, clause, predicate, assigned, frames)
    clause.modifiers = _modifiers(tokens, span, uncertain)
    return clause


def _assign_bare_nouns(
    tokens: list[Token],
    span: list[int],
    clause: Clause,
    predicate: Token,
    assigned: set[int],
    frames: FrameDict | None,
) -> None:
    """助詞を伴わない名詞に、適切な格を補って割り当てる。

    「僕行きます」の「僕」のように助詞が落ちている場合、フレームの
    空いている必須スロットへ入れる。

    ただし副詞可能名詞（明日 / 今日 / 来週）は対象外とする。
    これらは助詞なしで時を表すもので、格ではない。
    「明日名古屋に行く」の「明日」を GA に入れてしまうと主語を誤る。

    フレームが無い述語では補完しない。どの格が妥当かを知る根拠が
    無いまま推測すると、黙って間違えるより悪い。
    """
    frame = frames.get(predicate.lemma) if frames else None
    if frame is None:
        return

    for i in span:
        token = tokens[i]
        if i in assigned or token is predicate:
            continue
        if token.pos not in ("名詞", "代名詞"):
            continue
        # 直後が助詞なら、そちらの処理で扱われている
        if i + 1 < len(tokens) and tokens[i + 1].pos == "助詞":
            continue
        if is_adverbial_noun(token):
            continue

        for slot in frame.required + [
            s for s in frame.optional if s != "TIME"
        ]:
            if slot not in clause.slots:
                clause.slots[slot] = token
                break


def _modifiers(tokens: list[Token], span: list[int], uncertain: set[int]) -> list[str]:
    """節全体にかかるタグ。個々の Token には付けない。"""
    found: list[str] = []
    for i in span:
        token = tokens[i]
        if i in uncertain and UNCERTAIN not in found:
            found.append(UNCERTAIN)
        if _is_conditional_aux(token) and HYPOTHETICAL not in found:
            found.append(HYPOTHETICAL)
        if (
            token.subpos == "接続助詞"
            and token.surface == "ば"
            and HYPOTHETICAL not in found
        ):
            found.append(HYPOTHETICAL)
        # 文末の「か」。「かもしれない」の「か」は副助詞なので混ざらない。
        if (
            token.subpos == "終助詞"
            and token.surface == "か"
            and QUESTION not in found
        ):
            found.append(QUESTION)
    return found


def _is_conditional_aux(token: Token) -> bool:
    return token.pos == "助動詞" and token.inflection.startswith("仮定形")


def _is_conditional_ba(token: Token) -> bool:
    return token.subpos == "接続助詞" and token.surface == "ば"


def _is_conditional_marker(token: Token) -> bool:
    return _is_conditional_aux(token) or _is_conditional_ba(token)


def _relation(
    tokens: list[Token],
    clauses: list[Clause],
    indices: list[int],
    frames: FrameDict | None,
) -> str | None:
    if len(clauses) < 2:
        return None
    if any(HYPOTHETICAL in clause.modifiers for clause in clauses):
        return CONDITION
    if _is_purpose(tokens, indices, frames):
        return PURPOSE
    return NONE


def _is_purpose(
    tokens: list[Token], indices: list[int], frames: FrameDict | None
) -> bool:
    """「会いに行く」型の目的を検出する。

        前の節の述語が連用形
        その直後が格助詞の「に」
        後の節の述語が移動動詞

    連用形だけでは足りない。助動詞を吸収した文末述語も
    「連用形-一般」のままになる（「行きます」は「行き」の活用形を保つ）ため、
    直後が「に」であることを必須にしている。
    """
    if frames is None or len(indices) < 2:
        return False
    first, second = tokens[indices[0]], tokens[indices[1]]
    if not first.inflection.startswith("連用形"):
        return False
    following = indices[0] + 1
    if following >= len(tokens):
        return False
    particle = tokens[following]
    if particle.subpos != "格助詞" or particle.surface != "に":
        return False
    return frames.is_motion(second.lemma)
