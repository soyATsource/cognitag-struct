"""
生成層の検証。

核から周辺へ足していく方式が、どの段階でも文として成立すること。
出力しなかった要素が捨てられていないこと。
"""

from __future__ import annotations

from cognitag_struct.generate import (
    CASES,
    DEGREE,
    PREDICATE,
    SUBJECT,
    SUPPLEMENT,
    TOPIC,
    Utterance,
    elements_of,
    generate,
)
from cognitag_struct.ir import IR, Clause, Token
from cognitag_struct.parse import Unparsable, parse


def w(surface: str, pos: str, **kw) -> Token:
    return Token(surface=surface, lemma=surface, pos=pos, span=(0, 0), **kw)


def pi_clause() -> Clause:
    """目標 4 段階のもとになる節。"""
    return Clause(
        predicate=w("熱い", "形容詞"),
        slots={
            "GA": w(
                "Pi", "名詞", form="固有名詞", is_phrase=True, entry_id="raspi_1"
            )
        },
        topic=w("今日", "名詞"),
        degree=w("ちょっと", "副詞"),
        supplements=[w("48度", "名詞")],
    )


def pi_ir() -> IR:
    return IR(clauses=[pi_clause()], relation=None, source_text="")


class Test段階生成:
    def test_目標の4段階(self, style):
        expected = {
            0: "Piが熱い",
            1: "Piがちょっと熱い",
            2: "Piが今日はちょっと熱い",
            3: "Pi、今日はちょっと熱い。48度",
        }
        for verbosity, text in expected.items():
            assert generate(pi_ir(), verbosity, style).text == text

    def test_どの段階でも文になる(self, style):
        for verbosity in range(4):
            assert generate(pi_ir(), verbosity, style).text.strip()

    def test_プログラムから組み立てたIRを受け付ける(self, style):
        """Sentinel の観測値から直接作る用途。source_text は空でよい。"""
        ir = pi_ir()
        assert ir.source_text == ""
        assert generate(ir, 0, style).text == "Piが熱い"


class Test下位段階を壊さない:
    def test_要素は増える一方(self, style):
        """文字列の包含ではなく要素単位で確かめる。

        verbosity 3 で主格が「が」から読点に変わるので、
        文字列の包含では検証できない。失われていないのは要素の方である。
        """
        previous: set[str] = set()
        for verbosity in range(4):
            utterance = generate(pi_ir(), verbosity, style)
            emitted = _emitted_roles(utterance)
            assert previous <= emitted, f"verbosity {verbosity} で要素が減った"
            previous = emitted

    def test_主格は全段階で保たれる(self, style):
        """v3 で「Piが」→「Pi、」になるが Pi は消えていない。"""
        for verbosity in range(4):
            utterance = generate(pi_ir(), verbosity, style)
            assert "Pi" in utterance.text
            assert SUBJECT not in utterance.withheld

    def test_述語は全段階で保たれる(self, style):
        for verbosity in range(4):
            assert "熱い" in generate(pi_ir(), verbosity, style).text


def _emitted_roles(utterance: Utterance) -> set[str]:
    """出力された役割 = 全役割 - withheld された役割。"""
    all_roles = {
        role for role, tokens in elements_of(pi_clause()).items() if tokens
    }
    return all_roles - set(utterance.withheld)


class Test保留:
    def test_出力しなかった要素が残る(self, style):
        utterance = generate(pi_ir(), 0, style)

        assert utterance.withheld_surfaces() == {
            TOPIC: ["今日"],
            DEGREE: ["ちょっと"],
            SUPPLEMENT: ["48度"],
        }

    def test_段階が上がると保留が減る(self, style):
        counts = [
            len(generate(pi_ir(), v, style).withheld) for v in range(4)
        ]
        assert counts == sorted(counts, reverse=True)
        assert counts[-1] == 0

    def test_保留はTokenで持つ(self, style):
        """後から問われたときに品詞まで含めて答えられるようにするため。"""
        utterance = generate(pi_ir(), 0, style)
        token = utterance.withheld[SUPPLEMENT][0]

        assert isinstance(token, Token)
        assert token.surface == "48度"
        assert token.pos == "名詞"

    def test_全部出したら保留は空(self, style):
        assert generate(pi_ir(), 3, style).withheld == {}


class Test範囲外のverbosity:
    def test_負の値は0に丸める(self, style):
        utterance = generate(pi_ir(), -1, style)
        assert utterance.verbosity == 0
        assert utterance.text == "Piが熱い"

    def test_大きすぎる値は最大に丸める(self, style):
        utterance = generate(pi_ir(), 5, style)
        assert utterance.verbosity == 3
        assert utterance.text == "Pi、今日はちょっと熱い。48度"

    def test_例外を投げない(self, style):
        for verbosity in (-100, 0, 3, 100):
            generate(pi_ir(), verbosity, style)


class Test慣用句は使わない:
    def test_used_idiomは常にNone(self, style):
        """挿入位置が未定のため generate からは呼ばない。

        choose_idiom() は idiom.py に独立した部品として用意してある。
        """
        for verbosity in range(4):
            assert generate(pi_ir(), verbosity, style).used_idiom is None


class Test往復:
    def test_解析した文を生成に通せる(self, tokenizer, frames, style):
        text = "明日は名古屋に行きません"
        ir = parse(tokenizer.tokenize(text), source_text=text, frames=frames)
        assert not isinstance(ir, Unparsable)

        utterance = generate(ir, 3, style)

        # 完全一致は求めない。主要な要素が失われていないことだけ見る。
        assert "名古屋" in utterance.text
        assert "行きません" in utterance.text
        assert "明日" in utterance.text

    def test_格要素は核に含まれる(self, tokenizer, frames, style):
        """「名古屋に」は装飾ではなく命題の一部。verbosity 0 でも落とさない。"""
        text = "名古屋に行きます"
        ir = parse(tokenizer.tokenize(text), source_text=text, frames=frames)

        assert generate(ir, 0, style).text == "名古屋に行きます"

    def test_主格と目的語が保たれる(self, tokenizer, frames, style):
        text = "犬が男を噛んだ"
        ir = parse(tokenizer.tokenize(text), source_text=text, frames=frames)

        utterance = generate(ir, 3, style)
        assert "犬" in utterance.text
        assert "男" in utterance.text

    def test_2節でも生成できる(self, tokenizer, frames, style):
        text = "雨が降ったら名古屋には行きません"
        ir = parse(tokenizer.tokenize(text), source_text=text, frames=frames)

        utterance = generate(ir, 3, style)
        assert "雨" in utterance.text
        assert "名古屋" in utterance.text


class Testスタイル:
    def test_文言はtomlから読む(self, style):
        """組み立て方をコードに直書きしていないこと。"""
        assert style.levels[0].elements == [SUBJECT, CASES, PREDICATE]
        assert style.levels[3].subject_comma is True
        assert style.levels[0].subject_comma is False

    def test_慣用句頻度の既定(self, style):
        assert style.idiom_rate == 0.1
