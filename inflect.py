"""
述語を活用させる。

【なぜ要るか】
返答を「単語＋助詞＋単語」の並びから組み立てるなら、末尾の述語を
場面に合わせて活用できなければならない。活用できないと、
「行く」しか置けず、結局「〜と理解した」のような固定文に戻る。

    行く + 丁寧 + 疑問   → 行きますか
    行く + 過去 + 確認   → 行ったんだな
    行く + 否定          → 行かない
    行く + 意志          → 行こう

【表はデータに置く】
活用型ごとの語尾は data/conjugation.toml にある。ここにあるのは
「どの活用形を使い、何を後ろに付けるか」という組み立ての規則だけ。

【知らない活用型は活用させない】
表に無い活用型（形状詞など）は見出し語をそのまま返す。
推測で活用させると、存在しない語を作ってしまう。作れないものは
作れないままにして、呼び出し側が別の文型を選べるようにする。
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

# 態度。文末の形を決める。モダリティ（相手の発話の種類）とは別で、
# こちらがどう言うかの選択である。
PLAIN = "plain"            # 言い切り        行く
QUESTION = "question"      # 問い            行くか
CONFIRM = "confirm"        # 確認            行くんだな
VOLITIONAL = "volitional"  # 意志・勧誘      行こう
SUPPOSE = "suppose"        # 推量            行くだろう

MOODS: tuple[str, ...] = (PLAIN, QUESTION, CONFIRM, VOLITIONAL, SUPPOSE)

# 例外表の見出し。活用型ではなく見出し語で引く。
EXCEPTION = "exception"


class InflectError(ValueError):
    """活用表が読めないときに送出する。"""


@dataclass
class Spec:
    """どう活用させるか。素性の組み合わせで文末が決まる。"""

    polite: bool = False
    negative: bool = False
    past: bool = False
    mood: str = PLAIN
    # 終助詞。「ね」「よ」「な」。付けないなら空。
    final: str = ""


class Conjugator:
    """活用型ごとの語尾を持ち、述語を活用させる。"""

    def __init__(self, table: dict[str, dict[str, str]]) -> None:
        self.table = table

    @classmethod
    def load(cls, path: str | Path) -> "Conjugator":
        target = Path(path)
        if not target.exists():
            raise InflectError(f"ファイルが無い: {target}")
        with target.open("rb") as handle:
            data = tomllib.load(handle)
        return cls({str(k): dict(v) for k, v in data.items()})

    def _row(self, conjugation: str) -> dict[str, str] | None:
        """活用型に対応する行。

        Sudachi は「下一段-バ行」のように行まで書くが、一段は行によって
        語尾が変わらないので「下一段」の 1 行で足りる。前方一致で拾う。
        """
        row = self.table.get(conjugation)
        if row is not None:
            return row
        for key, value in self.table.items():
            if key != EXCEPTION and conjugation.startswith(key):
                return value
        return None

    def knows(self, conjugation: str) -> bool:
        return self._row(conjugation) is not None

    def stem(self, lemma: str, conjugation: str) -> str | None:
        row = self._row(conjugation)
        if row is None:
            return None
        cut = int(row.get("stem_cut", 1))
        # 「来る」は語幹が空になる。len == cut を許さないと
        # 語幹が見出し語のままになり「来る来る」ができてしまう。
        return lemma[:-cut] if cut and len(lemma) >= cut else lemma

    def base(self, lemma: str, conjugation: str, form: str) -> str | None:
        """活用形 1 つ分。語幹 + 語尾。

        規則から外れる語は例外表を先に見る。「行く」の過去は
        カ行の規則どおりなら「行いた」だが、実際は「行った」である。
        """
        exception = self.table.get(EXCEPTION, {}).get(lemma)
        if isinstance(exception, dict) and form in exception:
            return str(exception[form])

        row = self._row(conjugation)
        stem = self.stem(lemma, conjugation)
        if row is None or stem is None or form not in row:
            return None
        return stem + str(row[form])

    def realize(self, lemma: str, conjugation: str, spec: Spec) -> str | None:
        """述語 1 語を文末の形にする。作れなければ None。

        組み立ての順序は日本語の語順そのままで、
        「活用形 → 否定 → 丁寧 → 態度 → 終助詞」と後ろへ足していく。
        """
        if not self.knows(conjugation):
            return None
        adjective = conjugation == "形容詞"

        body = self._body(lemma, conjugation, spec, adjective)
        if body is None:
            return None
        return body + self._tail(spec, adjective)

    def _body(
        self, lemma: str, conjugation: str, spec: Spec, adjective: bool
    ) -> str | None:
        """終助詞と態度を除いた本体。"""
        # 意志は否定・過去と組み合わせない。「行かなかろう」は作らない。
        if spec.mood == VOLITIONAL:
            return self.base(lemma, conjugation, "volitional")

        # 確認と推量は本体を普通体にする。
        #
        # 「んだな」「だろう」は普通体に付く形なので、丁寧体に足すと
        # 「行きますんだな」という壊れた文になる。丁寧さは語尾の側で
        # 「んですね」「でしょう」として表す（_tail 参照）。
        if spec.mood in (CONFIRM, SUPPOSE):
            return self._plain(lemma, conjugation, spec, adjective)

        if spec.polite:
            if adjective:
                # 形容詞の丁寧は語幹ではなく終止形に「です」を付ける。
                #     高いです / 高くないです / 高かったです
                stem = self._plain(lemma, conjugation, spec, adjective)
                return None if stem is None else stem + "です"
            continuative = self.base(lemma, conjugation, "continuative")
            if continuative is None:
                return None
            if spec.negative and spec.past:
                return continuative + "ませんでした"
            if spec.negative:
                return continuative + "ません"
            if spec.past:
                return continuative + "ました"
            return continuative + "ます"

        return self._plain(lemma, conjugation, spec, adjective)

    def _plain(
        self, lemma: str, conjugation: str, spec: Spec, adjective: bool
    ) -> str | None:
        """普通体の本体。"""
        if spec.negative:
            imperfective = self.base(lemma, conjugation, "imperfective")
            if imperfective is None:
                return None
            # 形容詞の否定は「高くない」。動詞は「行かない」。同じ形になる。
            return imperfective + ("なかった" if spec.past else "ない")
        if spec.past:
            return self.base(lemma, conjugation, "past")
        return self.base(lemma, conjugation, "terminal")

    def _tail(self, spec: Spec, adjective: bool) -> str:
        """態度と終助詞。本体の後ろに足す。"""
        if spec.mood == QUESTION:
            # 「のか」は詰問に聞こえる。会話としては「の？」の方が近い。
            head = "か" if spec.polite else "の？"
        elif spec.mood == CONFIRM:
            # 本体は普通体なので、丁寧さはここで出す。
            head = "んですね" if spec.polite else "んだね"
        elif spec.mood == SUPPOSE:
            head = "でしょう" if spec.polite else "だろうね"
        else:
            head = ""
        return head + spec.final
