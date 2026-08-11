"""
テスト共通のフィクスチャ。

data/ の実ファイルを読む。ここは 13 件の小さな辞書なので、
モックを作るより実物を使う方が「登録内容と検索結果の対応」を
そのまま検証できる。

不正な定義（3 本目の軸・未定義の値）の検証だけは tmp_path に
その場で書いて読ませる。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# cognitag_struct を import できるようにする。
# tests/test_struct から 2 階層上がパッケージのルート、その親が import パス。
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_PARENT = _PACKAGE_ROOT.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from cognitag_struct.category_dict import CategoryDict  # noqa: E402
from cognitag_struct.phrase_dict import PhraseDict  # noqa: E402
from cognitag_struct.tokenizer import Tokenizer  # noqa: E402

DATA_DIR = _PACKAGE_ROOT / "data"
AXES_PATH = DATA_DIR / "axes.jsonl"
PHRASES_PATH = DATA_DIR / "phrases.jsonl"


@pytest.fixture(scope="session")
def category_dict() -> CategoryDict:
    return CategoryDict.load(AXES_PATH, PHRASES_PATH)


@pytest.fixture(scope="session")
def phrase_dict(category_dict: CategoryDict) -> PhraseDict:
    return PhraseDict(category_dict)


@pytest.fixture(scope="session")
def tokenizer(category_dict: CategoryDict, phrase_dict: PhraseDict) -> Tokenizer:
    """Sudachi 辞書のロードは重いので使い回す。"""
    return Tokenizer(category_dict, phrase_dict)


@pytest.fixture
def valid_axes(tmp_path: Path) -> Path:
    path = tmp_path / "axes.jsonl"
    path.write_text(
        '{"axis": "form", "values": ["ことわざ", "一般語"]}\n'
        '{"axis": "content", "values": ["失敗", "無駄"]}\n',
        encoding="utf-8",
    )
    return path


def write_phrases(path: Path, *lines: str) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def frames():
    from cognitag_struct.frames import FrameDict
    return FrameDict.load(DATA_DIR / "frames.jsonl")


@pytest.fixture(scope="session")
def questions():
    from cognitag_struct.gap import QuestionTemplates
    return QuestionTemplates.load(DATA_DIR / "questions.toml")


@pytest.fixture(scope="session")
def style():
    from cognitag_struct.generate import Style
    return Style.load(DATA_DIR / "generation_style.toml")
