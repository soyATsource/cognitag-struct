"""
句照合 → 形態素分割 の順で入力をトークン列にする。

【処理順序を逆にしないこと】
  1. 句辞書との最長一致照合
  2. 照合されなかった残りの区間だけを形態素分割

形態素分割してから句を再結合する実装にしてはいけない。
「河童 / の / 川流れ」から句を復元しようとすると、どの連続を
句とみなすかの候補が組み合わせで増える。先に句を確定させれば
その探索が要らない。

【形態素解析器】
CogniTag v2 が使っているものに合わせ、SudachiPy の SplitMode.C を使う。
CogniTag_V2 のコードを import せず SudachiPy を直接呼ぶのは、
このパッケージが CogniTag_V2 のディレクトリ配置に依存しないようにするため。
設定が同じなので分割結果は一致する。

【丁寧・否定・時制は素性、可能・受身・使役は別見出し】
    行きません → lemma「行く」  features [POLITE, NEGATIVE]
    行けません → lemma「行ける」features [POLITE, NEGATIVE]

「行ける」を「行く」に寄せない。可能・受身・使役は意味を変えるため、
Facet を別に持たなければならないからである。丁寧・否定・時制は
命題の内容を変えないので素性で足りる。

さいわい Sudachi は可能動詞「行ける」を独立した見出しとして持っており、
dictionary_form() をそのまま使えばこの区別が得られる。追加の活用解析は要らない。

なお一段動詞の可能形「食べられる」は 食べる + られる と分割される。
この「られる」は素性にせず、独立したトークンとして残す。
素性でないものを素性にしないための措置であり、
可能・受身の判別は次の段階で扱う。
"""

from __future__ import annotations

from .category_dict import CategoryDict
from .ir import DEFAULT_FORM, NEGATIVE, PAST, POLITE, Token
from .phrase_dict import PhraseDict

# 助動詞の見出し語 -> 機能素。
#
# ここに無い助動詞（られる / せる / させる / だ / う など）は
# 吸収せず独立したトークンとして残す。意味を変えるものを
# 黙って捨てないための方針である。
AUXILIARY_FEATURES: dict[str, str] = {
    "ます": POLITE,
    "です": POLITE,
    "ない": NEGATIVE,
    # 「ん」の見出し語は「ぬ」。「行きません」は ませ + ん の 2 形態素で、
    # 「ません」という単位は存在しない。結果として POLITE と NEGATIVE の
    # 両方が付く。
    "ぬ": NEGATIVE,
    # 「た」は過去のほか完了・状態にも使う（「困った人」など）。
    # この段階では区別せず PAST を付ける。
    "た": PAST,
}

AUXILIARY_POS = "助動詞"

# 機能素を付けずに述語へ畳むだけの助動詞。
#
# 「悩んでる」は 悩ん(動詞) + でる(助動詞) に割れる。畳まないと表層形が
# 「悩ん」で切れ、言い直しが「悩ん、と理解した」という壊れた文になる。
#
# 進行相（〜ている / 〜てる）は命題の内容を変えないが、機能素は
# POLITE / NEGATIVE / PAST の 3 つに限る決まりなので素性は付けない。
# 表層形と span を正しく保つためだけに畳む。
SURFACE_ONLY_LEMMAS: frozenset[str] = frozenset({"でる", "てる", "いる"})

# 活用形がこれで始まる助動詞は吸収しない。
#
# 「たら」と「た」は見出し語がどちらも「た」で、活用形だけが違う。
#     たら  lemma=た  活用形=仮定形-一般   条件
#     た    lemma=た  活用形=終止形-一般   過去
# 見出し語だけで引くと条件の「たら」を PAST として述語に飲み込んでしまい、
# 「雨が降ったら」と「雨が降った」が区別できなくなる。
#
# 条件は意味を変える要素なので、素性に潰すと復元できない。
# 可能・受身・使役を別見出し語として扱うのと同じ方針で、独立トークンとして残す。
# こうすると接続助詞の「ば」と扱いが揃い、後段は「接続助詞」と
# 「仮定形の助動詞」の 2 つを見るだけで条件節を検出できる。
#
# 機能素は POLITE / NEGATIVE / PAST の 3 つのままとする。
# CONDITIONAL を機能素として追加しない（素性は意味を変えないものに限る）。
CONDITIONAL_INFLECTION_PREFIX = "仮定形"

# --- サ変動詞 -------------------------------------------------------------
#
# 「連絡します」は 連絡(名詞/普通名詞/サ変可能) + します(動詞/非自立可能) に
# 割れる。そのままだと述語の見出し語が「する」になり、意味が名詞側に
# 取り残される。「連絡する」「確認する」「移動する」がすべて同じ述語に
# 見えてしまい、フレーム（必須スロットの定義）を持たせようがない。
#
# 日本語では名詞＋するで 1 つの動詞をなすので、1 トークンに畳んで
# 見出し語を「連絡する」にする。サ変可能かどうかは Sudachi の
# 品詞 3 番目が教えてくれるので、語彙表を自前で持つ必要はない。
# 「安心」は「サ変形状詞可能」、「緊張」は「サ変可能」。
# 表記が一通りではないので、完全一致ではなく含むかで見る。
SAHEN_MARKER = "サ変"
SURU_LEMMA = "する"


class Tokenizer:
    """句照合と形態素分割を組み合わせてトークン列を作る。"""

    def __init__(
        self, category_dict: CategoryDict, phrase_dict: PhraseDict | None = None
    ) -> None:
        self.category_dict = category_dict
        self.phrase_dict = phrase_dict or PhraseDict(category_dict)
        self._sudachi = None
        self._split_mode = None
        try:
            from sudachipy import Dictionary, SplitMode

            self._sudachi = Dictionary().create()
            # CogniTag v2 と同じ最長単位。A だと
            # 「東京ディズニーランド」が 東京 + ディズニーランド に割れる。
            self._split_mode = SplitMode.C
        except Exception:
            self._sudachi = None

    @property
    def available(self) -> bool:
        return self._sudachi is not None

    # -- 本体 -------------------------------------------------------------

    def tokenize(self, text: str) -> list[Token]:
        """句を先に確定させ、残りを形態素分割する。"""
        if not text:
            return []

        tokens: list[Token] = []
        cursor = 0
        for match in self.phrase_dict.find_all(text):
            if match.start > cursor:
                tokens.extend(self._morphemes(text, cursor, match.start))
            tokens.append(self._phrase_token(match))
            cursor = match.end

        if cursor < len(text):
            tokens.extend(self._morphemes(text, cursor, len(text)))
        return tokens

    def _phrase_token(self, match) -> Token:
        """句を 1 トークンにする。

        同一 surface に複数候補がある場合（「犬も歩けば棒に当たる」）は、
        Token が 1 件しか持てないため JSONL の登場順で先頭を採る。
        全候補は PhraseDict.match_at() から取れる。
        presupposition による絞り込みは次の段階で実装する。
        """
        entry = match.entries[0]
        return Token(
            surface=match.surface,
            lemma=entry.lemma,
            pos=entry.pos,
            form=entry.form,
            content=list(entry.content),
            features=[],
            is_phrase=True,
            entry_id=entry.id,
            span=(match.start, match.end),
        )

    def _morphemes(self, text: str, start: int, end: int) -> list[Token]:
        """[start, end) の区間を形態素分割する。

        span は元の文字列上の位置にそろえる（区間内の相対位置ではない）。
        """
        segment = text[start:end]
        if not segment:
            return []
        if self._sudachi is None:
            # 解析器が無い環境では区間をそのまま 1 トークンにする。
            # 落とすより、分割されていないことが分かる形で残す方がよい。
            return [
                Token(
                    surface=segment,
                    lemma=segment,
                    pos="未知",
                    span=(start, end),
                )
            ]

        tokens: list[Token] = []
        # 直前の形態素がサ変可能名詞だったか。「する」を見たときに
        # 畳むかどうかの判断に使う。
        previous_is_sahen = False

        for morpheme in self._sudachi.tokenize(segment, self._split_mode):
            surface = morpheme.surface()
            lemma = morpheme.dictionary_form()
            pos_tuple = morpheme.part_of_speech()
            pos = pos_tuple[0]
            subpos = pos_tuple[1] if len(pos_tuple) > 1 else ""
            subpos2 = pos_tuple[2] if len(pos_tuple) > 2 else ""
            inflection = pos_tuple[5] if len(pos_tuple) > 5 else ""
            begin = start + morpheme.begin()
            finish = start + morpheme.end()

            # サ変名詞 + する を 1 つの述語に畳む。
            # 「連絡」＋「します」→ 見出し語「連絡する」の動詞 1 語。
            if lemma == SURU_LEMMA and pos == "動詞" and previous_is_sahen and tokens:
                noun = tokens[-1]
                noun.surface = text[noun.span[0] : finish]
                noun.lemma = f"{noun.lemma}{SURU_LEMMA}"
                noun.pos = pos
                noun.subpos = subpos
                noun.subpos2 = subpos2
                noun.inflection = inflection
                noun.span = (noun.span[0], finish)
                # ます / た などの助動詞はこの後ろに来るので、
                # 既存の吸収処理がこの畳んだトークンへ features を付ける。
                previous_is_sahen = False
                continue

            previous_is_sahen = pos == "名詞" and SAHEN_MARKER in subpos2

            # 素性は付けないが表層形として畳む助動詞（〜てる / 〜ている）
            if (
                pos == AUXILIARY_POS
                and lemma in SURFACE_ONLY_LEMMAS
                and not inflection.startswith(CONDITIONAL_INFLECTION_PREFIX)
                and tokens
            ):
                previous = tokens[-1]
                previous.surface = text[previous.span[0] : finish]
                previous.span = (previous.span[0], finish)
                previous_is_sahen = False
                continue

            feature = AUXILIARY_FEATURES.get(lemma) if pos == AUXILIARY_POS else None
            # 仮定形は吸収しない（条件を素性に潰さないため。上の説明を参照）
            if inflection.startswith(CONDITIONAL_INFLECTION_PREFIX):
                feature = None
            if feature is not None and tokens:
                # 直前のトークンへ吸収する。span は吸収した範囲全体に伸ばす。
                previous = tokens[-1]
                if feature not in previous.features:
                    previous.features.append(feature)
                previous.surface = text[previous.span[0] : finish]
                previous.span = (previous.span[0], finish)
                continue

            tokens.append(
                Token(
                    surface=surface,
                    lemma=lemma,
                    pos=pos,
                    subpos=subpos,
                    subpos2=subpos2,
                    inflection=inflection,
                    # 句辞書に無い語の既定値。既存 CogniTag 辞書の
                    # カテゴリとの対応付けは次の段階で行う。
                    form=DEFAULT_FORM,
                    content=[],
                    features=[],
                    is_phrase=False,
                    entry_id=None,
                    span=(begin, finish),
                )
            )
        return tokens
