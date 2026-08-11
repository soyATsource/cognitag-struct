"""
cognitag_struct — CogniTag の構造層。

CogniTag 本体はテキストを形態素に割り、4 軸 Facet の TF-IDF 加重平均を返す。
話題の検出には機能するが、語順・格関係・否定を捨てるため命題を扱えない。

このパッケージは Facet の手前に構造層を置く。Facet は廃止せず、
構造層が絞り込んだ候補の中からどの語を選ぶかを決める役割へ移る。

    粗い 2 軸カテゴリで候補を数件に絞る  ← cognitag_struct
    連続値の Facet で 1 件を選ぶ          ← CogniTag v2

現段階で実装済みなのは以下の 3 つだけ。格解析と生成はまだ無い。

    ir.py            中間表現の型定義
    category_dict.py 2 軸交差による分類辞書
    phrase_dict.py   分解してはいけない句の照合
    tokenize.py      句照合 -> 形態素分割

CogniTag_V2 のコードは一切変更していない。参照もしていない
（形態素解析器の設定だけを合わせている）。
"""

from .category_dict import CategoryDict, CategoryDictError, Entry
from .facade import Analysis, CogniTag
from .ir import (
    CASES,
    FEATURES,
    NEGATIVE,
    PAST,
    POLITE,
    Clause,
    IR,
    Token,
)
from .modality import Modality, ModalityResult, detect_modality, needs_knowledge
from .phrase_dict import PhraseDict, PhraseMatch
from .tokenizer import Tokenizer

__all__ = [
    "CogniTag",
    "Analysis",
    "Modality",
    "detect_modality",
    "CategoryDict",
    "CategoryDictError",
    "Entry",
    "PhraseDict",
    "PhraseMatch",
    "Tokenizer",
    "Token",
    "Clause",
    "IR",
    "POLITE",
    "NEGATIVE",
    "PAST",
    "FEATURES",
    "CASES",
]
