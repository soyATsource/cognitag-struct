"""
分解してはいけない句の照合。

「河童の川流れ」を形態素解析器に渡すと 河童 / の / 川流れ に割れる。
割ってから再結合しようとすると、どの連続を句とみなすかの候補が爆発する。
先に句を確定させ、残った区間だけを形態素解析へ回す（tokenizer.py 参照）。

【この段階の割り切り】
完全一致のみを実装する。活用や語の挿入には対応しない。

    「猿も木から落ちた」   → 照合しない（末尾が活用しているため）
    「豚に真珠を与える」   → 「豚に真珠」の部分だけ照合する
    「猫に  小判」         → 照合しない（空白が入っているため）

将来の拡張点:
  - 末尾の用言の活用を許す（lemma で持っているので照合側で正規化する）
  - 句中への副詞などの挿入を許す（例:「豚にまさに真珠」）
  いずれも候補の増加を伴うので、絞り込みの仕組みと同時に入れること。

【曖昧性は解消しない】
同一 surface に複数エントリがある場合（「犬も歩けば棒に当たる」の
本来の意味と転じた意味）は、候補を全件返す。
presupposition を使った絞り込みは次の段階で行う。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .category_dict import CategoryDict, Entry


@dataclass(frozen=True)
class PhraseMatch:
    """入力文字列中で句と照合した 1 区間。"""

    start: int
    end: int
    surface: str
    # 同一 surface の候補を全件持つ。1 件に絞るのは次の段階。
    entries: tuple[Entry, ...] = field(default_factory=tuple)

    @property
    def is_ambiguous(self) -> bool:
        return len(self.entries) > 1


class PhraseDict:
    """句の最長一致照合。

    13 件程度を想定した素直な実装。Trie 等の最適化はしていない。
    語数が数千に増えたら見直すこと。
    """

    def __init__(self, category_dict: CategoryDict) -> None:
        self.category_dict = category_dict
        self._by_surface: dict[str, list[Entry]] = {}
        for entry in category_dict.entries.values():
            self._by_surface.setdefault(entry.surface, []).append(entry)
        # 走査時に「ここから何文字まで見るか」を決めるための上限
        self._max_length = max(
            (len(s) for s in self._by_surface), default=0
        )

    def candidates(self, surface: str) -> list[Entry]:
        """表層形に一致するエントリを全件返す。無ければ空。"""
        return list(self._by_surface.get(surface, []))

    def match_at(self, text: str, index: int) -> PhraseMatch | None:
        """index から始まる最長の句を返す。無ければ None。

        同じ開始位置に複数の長さの候補があれば、長い方を選ぶ。
        「東京」と「東京ディズニーランド」なら後者。
        """
        limit = min(self._max_length, len(text) - index)
        for length in range(limit, 0, -1):
            surface = text[index : index + length]
            entries = self._by_surface.get(surface)
            if entries:
                return PhraseMatch(
                    start=index,
                    end=index + length,
                    surface=surface,
                    entries=tuple(entries),
                )
        return None

    def find_all(self, text: str) -> list[PhraseMatch]:
        """文字列全体を走査し、重ならない句を先頭から順に拾う。

        照合した区間は飛ばして続きを見る。したがって結果は
        位置の昇順に並び、互いに重ならない。
        """
        matches: list[PhraseMatch] = []
        index = 0
        while index < len(text):
            found = self.match_at(text, index)
            if found is None:
                index += 1
                continue
            matches.append(found)
            index = found.end
        return matches
