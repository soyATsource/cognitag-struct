"""
2 軸交差による分類辞書。

【分類の粒度は粗くてよい】
多少意味が異なるものが同じカテゴリに同居することを許容する。
たとえば「豚に真珠」「猫に小判」「馬の耳に念仏」は厳密には意味がずれるが、
すべて (ことわざ × 無駄) に入れる。

カテゴリ層の役割は候補を数件に絞るところまでで、最終的な 1 件の選択は
Facet 4 軸が行う。カテゴリを細かくしようとすると分類体系が管理不能になり、
「この語はどちらのカテゴリか」という判断が無限に増える。
粗いカテゴリ × 連続値の Facet という分担にすることで、
どちらの層も手に負える大きさに収まる。

【木構造にしない理由】
分類を木にすると経路の順序が固定される。

    ことわざ → 失敗 → 河童の川流れ
    失敗 → ことわざ → 河童の川流れ

利用者はどちらの順でも同じ結果に到達できなければならない。
木構造で逆順の経路を作るには同一エントリを二重に登録するしかなく、
片方だけ更新されたときに整合性が壊れる。

2 本の軸の交差なら、集合の積が可換であることによって
両方向が自動的に保証される。members(form=X, content=Y) と
members(content=Y, form=X) は、実装上まったく同じ積集合を計算する。

【軸は 2 本のみ】
3 本目を追加しないこと。3 本目が必要に見える区別は presupposition で表す。
たとえば「熟練者の失敗」は 3 軸目ではなく presupposition ["#熟練"] である。
軸を増やすと交差の組み合わせが指数で増え、粗い分類という利点が消える。
読み込み時に 3 本目を検出したらエラーにする。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# この設計で許す軸はこの 2 本だけ。増やさないこと（冒頭の説明を参照）。
FORM_AXIS = "form"
CONTENT_AXIS = "content"
ALLOWED_AXES: frozenset[str] = frozenset({FORM_AXIS, CONTENT_AXIS})


class CategoryDictError(ValueError):
    """辞書の定義が不正なときに送出する。"""


@dataclass
class Entry:
    """句辞書の 1 エントリ。

    id が主キー。surface は重複しうる（同一表層形で意味が相反する場合、
    id を分けて別エントリにするため）。
    """

    id: str
    surface: str
    lemma: str
    pos: str
    form: str
    content: list[str] = field(default_factory=list)
    presupposition: list[str] = field(default_factory=list)
    # 今はすべて 0.0 のプレースホルダ。既存の注釈パイプラインで後から埋める。
    # このモジュールは facet の具体的な値に一切依存しない。
    facet: dict[str, float] = field(default_factory=dict)
    note: str = ""

    def tags(self) -> list[str]:
        """form / content からタグ表記を導出する。二重管理を避けるため。"""
        return [f"#{self.form}"] + [f"#{value}" for value in self.content]


class CategoryDict:
    """軸の定義と句エントリを保持し、2 軸の交差で検索する。"""

    def __init__(
        self, axes: dict[str, list[str]], entries: list[Entry]
    ) -> None:
        self._axes = axes
        self.entries: dict[str, Entry] = {}
        # JSONL の登場順。members() の戻り順を決定的にするために使う。
        # set をそのまま list 化すると順序が実行ごとに変わりえて、
        # 「引数順に依存しない」ことを list の比較で検証できなくなる。
        self._order: dict[str, int] = {}
        self._by_form: dict[str, set[str]] = {}
        self._by_content: dict[str, set[str]] = {}

        for index, entry in enumerate(entries):
            if entry.id in self.entries:
                raise CategoryDictError(f"id が重複している: {entry.id}")
            self.entries[entry.id] = entry
            self._order[entry.id] = index
            self._by_form.setdefault(entry.form, set()).add(entry.id)
            for value in entry.content:
                self._by_content.setdefault(value, set()).add(entry.id)

    # -- 読み込み ---------------------------------------------------------

    @classmethod
    def load(cls, axes_path: str | Path, phrases_path: str | Path) -> "CategoryDict":
        """軸と句を読み込む。パスは必ず引数で受け取る（ハードコードしない）。"""
        axes = cls._load_axes(Path(axes_path))
        entries = cls._load_entries(Path(phrases_path), axes)
        return cls(axes, entries)

    @staticmethod
    def _load_axes(path: Path) -> dict[str, list[str]]:
        axes: dict[str, list[str]] = {}
        for line_no, raw in _iter_jsonl(path):
            name = raw.get("axis")
            values = raw.get("values")
            if not name or not isinstance(values, list):
                raise CategoryDictError(
                    f"{path}:{line_no} axis と values が必要"
                )
            if name in axes:
                raise CategoryDictError(f"{path}:{line_no} 軸が重複している: {name}")
            axes[name] = [str(v) for v in values]

        unknown = set(axes) - ALLOWED_AXES
        if unknown:
            raise CategoryDictError(
                f"3 本目の軸は許可されていない: {sorted(unknown)}。"
                "増やしたい区別は presupposition で表すこと"
            )
        missing = ALLOWED_AXES - set(axes)
        if missing:
            raise CategoryDictError(f"軸が足りない: {sorted(missing)}")
        return axes

    @staticmethod
    def _load_entries(path: Path, axes: dict[str, list[str]]) -> list[Entry]:
        forms = set(axes[FORM_AXIS])
        contents = set(axes[CONTENT_AXIS])

        entries: list[Entry] = []
        for line_no, raw in _iter_jsonl(path):
            entry_id = raw.get("id")
            if not entry_id:
                raise CategoryDictError(f"{path}:{line_no} id が無い")

            form = raw.get("form")
            if form not in forms:
                raise CategoryDictError(
                    f"{path}:{line_no} 未定義の form: {form!r}。"
                    f"axes.jsonl に無い値は使えない"
                )

            content = list(raw.get("content") or [])
            for value in content:
                if value not in contents:
                    raise CategoryDictError(
                        f"{path}:{line_no} 未定義の content: {value!r}。"
                        f"axes.jsonl に無い値は使えない"
                    )

            entries.append(
                Entry(
                    id=str(entry_id),
                    surface=str(raw.get("surface", "")),
                    lemma=str(raw.get("lemma", raw.get("surface", ""))),
                    pos=str(raw.get("pos", "")),
                    form=str(form),
                    content=content,
                    presupposition=list(raw.get("presupposition") or []),
                    facet=dict(raw.get("facet") or {}),
                    note=str(raw.get("note", "")),
                )
            )
        return entries

    # -- 検索 -------------------------------------------------------------

    def members(
        self, form: str | None = None, content: str | None = None
    ) -> list[Entry]:
        """2 軸の交差でエントリを引く。

        両方指定なら積集合、片方だけならその軸で絞り、両方 None なら全件。
        積は可換なので、引数をどちらの順で考えても結果は同じになる。

        戻り値は JSONL の登場順にそろえる。集合の反復順に任せると
        実行ごとに並びが変わり、結果の同一性を比較できなくなるため。
        """
        if form is None and content is None:
            ids: set[str] = set(self.entries)
        elif content is None:
            ids = set(self._by_form.get(form, set()))
        elif form is None:
            ids = set(self._by_content.get(content, set()))
        else:
            ids = self._by_form.get(form, set()) & self._by_content.get(
                content, set()
            )

        return [self.entries[i] for i in sorted(ids, key=self._order.__getitem__)]

    def get(self, entry_id: str) -> Entry | None:
        return self.entries.get(entry_id)

    def axis_values(self, axis: str) -> list[str]:
        if axis not in self._axes:
            raise CategoryDictError(f"未定義の軸: {axis}")
        return list(self._axes[axis])

    def __len__(self) -> int:
        return len(self.entries)


def _iter_jsonl(path: Path):
    """JSONL を 1 行ずつ読む。空行と # で始まる行は飛ばす。"""
    if not path.exists():
        raise CategoryDictError(f"ファイルが無い: {path}")
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                yield line_no, json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise CategoryDictError(f"{path}:{line_no} JSON が壊れている: {exc}")
