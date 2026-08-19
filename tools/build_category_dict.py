#!/usr/bin/env python3
"""
カテゴリ辞書の候補を LLM に列挙させる。

    cd D:\\AIModel
    python -X utf8 cognitag_struct/tools/build_category_dict.py --content 場所
    python -X utf8 cognitag_struct/tools/build_category_dict.py --all
    python -X utf8 cognitag_struct/tools/build_category_dict.py --apply cognitag_struct/tools/out/reviewed.jsonl

【何をするものか】
「場所と言える名詞を挙げよ」と Ollama の gemma3:4b に複数回聞き、
安定して出てきた語だけを候補として書き出す。phrases.jsonl にそのまま
足せる形の JSONL と、人が目で見て削るための TSV を出力する。

【data/phrases.jsonl には直接書かない】
LLM の出力をそのまま辞書にすると、誤りが混ざったまま返答に出る。
このプロジェクトの取り柄は「間違いを人が直せること」なので、
人の検収を工程として強制する。--apply を明示的に叩いたときだけ追記する。

【1 回目の試行を捨てる】
同じ問いを繰り返すと 1 回目だけが系統的にずれることが、姉妹プロジェクトの
実験で分かっている。ここでも 1 回目は捨て、残りの試行で一致した語を採る。

【依存を増やさない】
ollama の Python パッケージは使わず、標準ライブラリだけで HTTP を叩く。
このパッケージの依存は sudachipy だけ、という状態を崩さないため。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cognitag_struct.category_dict import (  # noqa: E402
    CONTENT_AXIS,
    FORM_AXIS,
    CategoryDict,
)

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
DATA = PACKAGE / "data"
OUT = HERE / "out"

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "gemma3:4b"

# 既定の試行回数。1 回目は捨てるので、実際に採決に使うのは -1 回。
TRIALS = 4
# 採決に使う試行のうち、何回出れば採用するか。
AGREE = 2

# 解析器が品詞から既に判別できる名詞の下位分類。
#
# 「名古屋」は地名なので Sudachi だけで #場所 が付く（reasoning.py の
# NOUN_TAGS）。辞書に足しても情報が増えないので候補から外す。
# 二重管理を避けるため、ここは reasoning.py から読む。
from cognitag_struct.reasoning import NOUN_TAGS  # noqa: E402

# 表層形として認めない形。記号・英数字のみ・長すぎるものを弾く。
BAD_SURFACE = re.compile(r"^[\W\d_]+$|^[A-Za-z0-9]+$")
MAX_LEN = 12

# id の接頭辞。content 値は日本語なので ASCII の別名を持たせる。
# ここに無い値は cat<番号> になる（動作はするが読みにくい）。
SLUGS: dict[str, str] = {
    "失敗": "fail", "幸運": "luck", "無駄": "vain", "努力": "effort",
    "移動": "move", "場所": "place", "状態": "state",
    "健康": "health", "食事": "food", "仕事": "work", "学習": "study",
    "金銭": "money", "道具": "tool", "人物": "person", "身体": "body",
    "家族": "family", "自然": "nature", "交通": "transit", "住居": "home",
    "情報": "info", "法制度": "law",
}

# 文を書かせて格から抜く方式（--mode sentence）で使う。
#
# 【なぜ語の列挙ではなく文なのか】
# 「場所を挙げよ」と聞くと gemma3:4b は 10 語で止まる。何度聞いても
# 同じ 10 語が出る。一方、文を書かせれば 1 回で数十語が本文に現れる。
# CogniTag_V2 の辞書 11,980 語も同じ方法で作った（語を発明させず、
# 書かせた文に実際に出てきた語だけを採る）。
#
# 【どの名詞がどのカテゴリかは、フレームが決める】
# 「駅に行った」の NI は場所である、と frames.jsonl が既に宣言している
# （content_constraints）。LLM に分類させず、人が書いた 136 語ぶんの
# 知識で振り分ける。分類の責任を LLM に渡さない。
SENTENCE_PROMPT = """日常のありふれた出来事を、短い文で {count} 個書いてください。

条件:
- 1 行に 1 文。番号も説明も付けない
- かならず「{verb}」を使う
- 20 文字以内にする
- {scene}の場面にする

例（形の見本。この語は使わなくてよい）:
窓を開けた
電話が鳴りました

{count} 文:"""

# 場面。同じ問いを繰り返しても同じ語しか出ないので、場面を変えて振れ幅を作る。
SCENES = [
    "朝の支度", "通勤や通学", "仕事中", "休日", "買い物", "食事",
    "体調が悪いとき", "旅行", "家事", "夜の時間", "友人と会うとき",
    "季節の行事",
]

# 1 つのスロットが複数の content を許すとき、どちらかに決めるための規則。
#
# 「行く」の NI は場所と移動の両方を許す（frames.jsonl）。
#     駅に行く      → 駅は場所
#     買い物に行く  → 買い物は行為であって場所ではない
# 両者は Sudachi の品詞で割れる。サ変可能（＝「する」が付く）なら行為。
# 実測では 散歩/買い物/ドライブ/仕事 が サ変可能、
# 駅/病院/公園/銀行/郵便局/店/家 が 一般 で、きれいに分かれた。
SAHEN_SUBPOS2 = "サ変可能"
SAHEN_PREFERS = "移動"
NON_SAHEN_PREFERS = "場所"

# 逆算に使わないスロット。
#
# GA（主語）の content_constraints は「その語が何であるか」ではなく
# 「どういう状態にあるか」を述べる位置なので、逆算するとカテゴリにならない。
# 実測: GA=状態 を持つ述語から抜くと「信号」「コーヒー」「ボタン」が
# 状態として採れてしまった。信号は状態ではない。
#
# frames.py の冒頭にあるとおり content_constraints は「優先度であって
# 制約ではない」。優先度を逆向きに読んでよいのは、その格に入る語の
# 種類が決まっている場合だけである。「〜に行く」の NI はそれに当たる。
DENY_SLOTS: frozenset[str] = frozenset({"GA"})

PROMPT = """日本語の名詞を挙げてください。

条件:
- 「{content}」と言える具体的な名詞だけを挙げる
- 日常会話に出てくる語に限る。専門用語や固有名詞は挙げない
- 1 行に 1 語だけ書く。説明や記号を付けない
- {count} 語まで

例（「食べ物」の場合）:
ご飯
パン
味噌汁

「{content}」:"""


@dataclass
class Candidate:
    surface: str
    content: str
    hits: int          # 採決に使った試行のうち何回出たか
    trials: int        # 採決に使った試行の回数
    warmup_only: bool = False   # 1 回目にだけ出た語
    reasons: list[str] = field(default_factory=list)  # 落とした理由

    @property
    def accepted(self) -> bool:
        return not self.reasons and self.hits >= AGREE


# --- LLM ---------------------------------------------------------------


def ask(prompt: str, seed: int, timeout: int) -> str:
    """Ollama に 1 回聞く。失敗したら空文字を返す（止めない）。

    temperature=0 と seed を渡すが、完全な決定性は保証されない。
    だからこそ複数回聞いて一致を見る。
    """
    body = json.dumps(
        {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "seed": seed, "num_predict": 400},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read()).get("response", "")
    except urllib.error.URLError as exc:
        print(f"    Ollama に繋がらない: {exc}", file=sys.stderr)
        return ""
    except TimeoutError:
        print("    応答が時間内に返らなかった", file=sys.stderr)
        return ""


def parse_words(text: str) -> list[str]:
    """応答から語を拾う。箇条書きの記号と番号は落とす。"""
    words: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        line = re.sub(r"^[-*・\d.、,)\s]+", "", line)
        line = re.sub(r"[。、,.\s].*$", "", line)
        if line and line not in words:
            words.append(line)
    return words


# --- 検証 --------------------------------------------------------------


class Validator:
    """候補が辞書に足す価値のある名詞かを判定する。

    落とした理由を必ず記録する。黙って捨てると、なぜその語が
    辞書に入らなかったのかを後から追えない。
    """

    def __init__(self, existing: CategoryDict) -> None:
        self.existing = {e.surface for e in existing.members()}
        try:
            from sudachipy import Dictionary, SplitMode

            self.sudachi = Dictionary().create()
            self.mode = SplitMode.C
        except Exception as exc:  # noqa: BLE001
            print(f"SudachiPy が使えない: {exc}", file=sys.stderr)
            self.sudachi = None
            self.mode = None

    def is_action_noun(self, surface: str) -> str | None:
        """動作や性質を指す名詞か。カテゴリ辞書に入れるべきでないもの。

        「予防」「挑戦」「検索」はサ変可能で、指しているのは事物ではなく
        動作である。動作は述語の担当（frames.jsonl）であって、
        カテゴリ辞書に「食事カテゴリの語」として入れると誤る。
        「専門的」「一貫」も形状詞可能で、性質であって事物ではない。

        なお sentence 方式ではサ変可能を「行為」として使い分けているので、
        この判定はそちらには適用しない（「買い物に行く」の買い物は
        移動カテゴリの語として正しい）。
        """
        if self.sudachi is None:
            return None
        morphemes = self.sudachi.tokenize(surface, self.mode)
        if len(morphemes) != 1:
            return None
        subpos2 = morphemes[0].part_of_speech()[2]
        if "サ変" in subpos2:
            return f"動作を指す名詞（{subpos2}）"
        if "形状詞" in subpos2:
            return f"性質を指す名詞（{subpos2}）"
        return None

    def check(self, surface: str) -> list[str]:
        reasons: list[str] = []
        if len(surface) > MAX_LEN:
            reasons.append("長すぎる")
        if BAD_SURFACE.match(surface):
            reasons.append("記号か英数のみ")
        if surface in self.existing:
            reasons.append("既に辞書にある")
        if self.sudachi is None:
            return reasons

        morphemes = self.sudachi.tokenize(surface, self.mode)
        if len(morphemes) != 1:
            # 複数形態素でも句として登録する価値はあるが、
            # カテゴリ辞書の第一段では単語に絞る。
            reasons.append(f"{len(morphemes)} 形態素に割れる")
            return reasons

        morpheme = morphemes[0]
        pos = morpheme.part_of_speech()
        if pos[0] != "名詞":
            reasons.append(f"名詞でない（{pos[0]}）")
        if morpheme.surface() != surface:
            reasons.append("表層形が一致しない")
        # 品詞から既にタグが取れる語は足す意味が無い
        for level in (pos[1], pos[2]):
            if level in NOUN_TAGS:
                reasons.append(f"品詞から取れる（{level} → {NOUN_TAGS[level]}）")
                break
        return reasons


# --- 収集 --------------------------------------------------------------


def collect(content: str, count: int, trials: int, timeout: int,
            validator: Validator) -> list[Candidate]:
    """1 つの content 値について候補を集める。"""
    prompt = PROMPT.format(content=content, count=count)
    runs: list[list[str]] = []
    for index in range(trials):
        label = "捨てる" if index == 0 else "採決"
        print(f"  試行 {index + 1}/{trials}（{label}）… ", end="", flush=True)
        words = parse_words(ask(prompt, seed=index, timeout=timeout))
        print(f"{len(words)} 語")
        runs.append(words)

    if len(runs) < 2:
        return []

    warmup, rest = runs[0], runs[1:]
    counter: Counter[str] = Counter()
    for words in rest:
        for word in set(words):
            counter[word] += 1

    candidates: list[Candidate] = []
    for word, hits in counter.most_common():
        candidate = Candidate(
            surface=word, content=content, hits=hits, trials=len(rest)
        )
        candidate.reasons = validator.check(word)
        candidates.append(candidate)

    # 1 回目にしか出なかった語も記録する。捨てた判断を見えるようにするため。
    only_warmup = set(warmup) - set(counter)
    for word in sorted(only_warmup):
        candidate = Candidate(
            surface=word, content=content, hits=0, trials=len(rest),
            warmup_only=True,
        )
        candidate.reasons = ["1 回目にしか出ていない"] or candidate.reasons
        candidates.append(candidate)

    return candidates


def collect_by_sentence(rounds: int, count: int, timeout: int,
                        validator: Validator, wanted: set[str] | None,
                        verbose: bool) -> list[Candidate]:
    """文を書かせ、格スロットから名詞を抜く。

    対象の述語は frames.jsonl のうち content_constraints を持つものに限る。
    どのスロットがどの content を取るかを宣言しているのがそれだけだからで、
    宣言の無い述語から抜くと、どのカテゴリに入れてよいか分からない。
    """
    from cognitag_struct.facade import CogniTag
    from cognitag_struct.ir import IR

    cognitag = CogniTag()
    targets = [
        (lemma, frame)
        for lemma, frame in cognitag.frames.frames.items()
        if any(slot not in DENY_SLOTS for slot in frame.content_constraints)
    ]
    if wanted:
        targets = [
            (lemma, frame) for lemma, frame in targets
            if any(v in wanted for vs in frame.content_constraints.values()
                   for v in vs)
        ]
    print(f"対象の述語 {len(targets)} 語 × {rounds} 巡")

    # (content, 語) -> 出現回数。文をまたいで数える。
    seen: Counter[tuple[str, str]] = Counter()
    sentences = 0
    parsed = 0

    for round_index in range(rounds):
        scene = SCENES[round_index % len(SCENES)]
        for lemma, frame in targets:
            prompt = SENTENCE_PROMPT.format(count=count, verb=lemma, scene=scene)
            text = ask(prompt, seed=round_index, timeout=timeout)
            for line in text.splitlines():
                line = re.sub(r"^[-*・\d.、,)\s]+", "", line.strip())
                if not (4 <= len(line) <= 40):
                    continue
                sentences += 1
                analysis = cognitag.analyze(line)
                if not isinstance(analysis.ir, IR):
                    continue
                parsed += 1
                for clause in analysis.ir.clauses:
                    if clause.predicate is None:
                        continue
                    got = cognitag.frames.get(clause.predicate.lemma)
                    if got is None:
                        continue
                    for slot, wants in got.content_constraints.items():
                        if slot in DENY_SLOTS:
                            continue
                        token = clause.slots.get(slot)
                        if token is None or token.pos != "名詞":
                            continue
                        content = _pick_content(token, wants)
                        if content and (not wanted or content in wanted):
                            seen[(content, token.lemma)] += 1
            if verbose:
                print(f"  [{scene}] {lemma}: 文 {sentences} / "
                      f"解析成功 {parsed} / 語 {len(seen)}")

    print(f"生成した文 {sentences} / 構造が取れた {parsed} / "
          f"抜き出した語 {len(seen)}")

    candidates: list[Candidate] = []
    for (content, word), hits in seen.most_common():
        candidate = Candidate(
            surface=word, content=content, hits=hits, trials=sentences
        )
        candidate.reasons = validator.check(word)
        candidates.append(candidate)
    return candidates


def collect_from_dictionary(source: Path, mapping: dict[str, list[str]],
                            share: float, votes: int,
                            validator: Validator,
                            wanted: set[str] | None) -> list[Candidate]:
    """CogniTag_V2 の語彙辞書から専門用語を拾う。

    【なぜ多数決なのか】
    V2 の tags は分類ではなく出現記録なので、1 語に平均 10 個の
    カテゴリが付く。「1 つのカテゴリにしか出ない語だけ採る」方式は
    実測すると BNP・COPD・EBITDA のような稀語ばかりが残り、
    使う語がすべて落ちた。カテゴリ数が少ない語＝めったに出ない語だからである。

    そこで写ったカテゴリの多数決を取り、首位の占有率で足切りする。
    日常語（会社 57 カテゴリ / 仕事 63 カテゴリ）は割れて落ちるが、
    出現が偏っている専門用語は残る。日常語は sentence 側の担当。

    LLM は使わない。ここは既にある辞書を読み替えるだけの処理である。
    """
    data = json.loads(source.read_text(encoding="utf-8"))
    entries = data.get("entries") or {}
    print(f"語彙辞書 {len(entries)} 語 / 写像 {len(mapping)} カテゴリ値")

    # カテゴリ名 -> content 値の集合
    all_cats = {c for e in entries.values() for c in e.get("tags", [])}
    cat_to_content: dict[str, set[str]] = {}
    for cat in all_cats:
        name = cat.lstrip("#")
        hit = {
            content for content, keys in mapping.items()
            if any(key in name for key in keys)
        }
        if hit:
            cat_to_content[cat] = hit
    print(f"カテゴリ {len(all_cats)} のうち写ったのは {len(cat_to_content)}")

    candidates: list[Candidate] = []
    for word, entry in entries.items():
        if entry.get("pos") != "名詞":
            continue
        counter: Counter[str] = Counter()
        for cat in entry.get("tags", []):
            for content in cat_to_content.get(cat, ()):
                counter[content] += 1
        if not counter:
            continue
        total = sum(counter.values())
        content, hits = counter.most_common(1)[0]
        if wanted and content not in wanted:
            continue

        candidate = Candidate(
            surface=word, content=content, hits=hits, trials=total
        )
        candidate.reasons = validator.check(word)
        action = validator.is_action_noun(word)
        if action:
            candidate.reasons.append(action)
        if hits < votes:
            candidate.reasons.append(f"票が {hits} 件だけ")
        if hits / total < share:
            candidate.reasons.append(
                f"首位の占有率 {hits / total:.0%} が閾値未満"
            )
        candidates.append(candidate)
    return candidates


def _pick_content(token, wants: list[str]) -> str | None:
    """スロットが複数の content を許すとき、1 つに決める。

    決められないときは None を返して捨てる。曖昧なまま辞書に入れると
    後から誰も直せない。
    """
    if len(wants) == 1:
        return wants[0]
    if set(wants) == {SAHEN_PREFERS, NON_SAHEN_PREFERS}:
        return (SAHEN_PREFERS if token.subpos2 == SAHEN_SUBPOS2
                else NON_SAHEN_PREFERS)
    return None


def load_mapping(path: Path, known: set[str]) -> dict[str, list[str]] | None:
    """カテゴリ名 → content 値 の対応表を読む。

    axes.jsonl に無い content 値が書いてあれば止める。軸の値を
    増やさないまま写像だけ足すと、--apply の段階で全件弾かれる。
    """
    import tomllib

    if not path.exists():
        print(f"対応表が無い: {path}")
        return None
    with path.open("rb") as handle:
        data = tomllib.load(handle)

    mapping = {
        str(content): [str(k) for k in keys]
        for content, keys in data.items()
        if isinstance(keys, list)
    }
    unknown = sorted(set(mapping) - known)
    if unknown:
        print(f"axes.jsonl に無い content 値が対応表にある: {unknown}")
        print("先に axes.jsonl の content へ追加すること（軸は増やさない）")
        return None
    return mapping


# --- 出力 --------------------------------------------------------------


def to_entry(candidate: Candidate, index: int) -> dict:
    """phrases.jsonl の 1 行にする。facet は空でよい（select が既定値で扱う）。"""
    slug = SLUGS.get(candidate.content, "cat")
    return {
        "id": f"{slug}_{index:03d}",
        "surface": candidate.surface,
        "lemma": candidate.surface,
        "pos": "名詞",
        "form": "一般語",
        "content": [candidate.content],
        "presupposition": [],
        "facet": {},
        "note": f"{MODEL} 生成 / {candidate.hits}/{candidate.trials} 一致",
    }


def write_outputs(candidates: list[Candidate], stem: str) -> tuple[Path, Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    accepted = [c for c in candidates if c.accepted]

    jsonl = OUT / f"{stem}.candidates.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for index, candidate in enumerate(accepted, 1):
            handle.write(
                json.dumps(to_entry(candidate, index), ensure_ascii=False) + "\n"
            )

    tsv = OUT / f"{stem}.review.tsv"
    with tsv.open("w", encoding="utf-8") as handle:
        handle.write("採否\t語\tcontent\t一致\t落とした理由\n")
        for candidate in candidates:
            mark = "採用" if candidate.accepted else "除外"
            reasons = list(candidate.reasons)
            # 理由の空欄をなくす。なぜ落ちたか分からない行を残さない。
            if not reasons and candidate.hits < AGREE:
                reasons.append(f"出現が {candidate.hits} 回だけ")
            handle.write(
                f"{mark}\t{candidate.surface}\t{candidate.content}\t"
                f"{candidate.hits}/{candidate.trials}\t"
                f"{'; '.join(reasons)}\n"
            )
    return jsonl, tsv


def apply_entries(path: Path, target: Path, dry_run: bool) -> int:
    """検収済みの JSONL を phrases.jsonl へ追記する。

    id と surface の重複を弾く。既存行には一切触れない（追記のみ）。
    """
    existing_lines = target.read_text(encoding="utf-8").splitlines()
    existing = [json.loads(l) for l in existing_lines if l.strip()
                and not l.strip().startswith("#")]
    known_ids = {e["id"] for e in existing}
    known_surfaces = {e["surface"] for e in existing}

    axes = CategoryDict.load(DATA / "axes.jsonl", DATA / "phrases.jsonl")
    forms = set(axes.axis_values(FORM_AXIS))
    contents = set(axes.axis_values(CONTENT_AXIS))

    additions: list[dict] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.strip().startswith("#"):
            continue
        entry = json.loads(line)
        if entry["id"] in known_ids:
            print(f"  {line_no}: id が重複 {entry['id']} → 飛ばす")
            continue
        if entry["surface"] in known_surfaces:
            print(f"  {line_no}: 語が重複 {entry['surface']} → 飛ばす")
            continue
        if entry["form"] not in forms:
            print(f"  {line_no}: 未定義の form {entry['form']} → 飛ばす")
            continue
        bad = [c for c in entry["content"] if c not in contents]
        if bad:
            print(f"  {line_no}: 未定義の content {bad} → 飛ばす。"
                  f"先に axes.jsonl へ追加すること")
            continue
        additions.append(entry)
        known_ids.add(entry["id"])
        known_surfaces.add(entry["surface"])

    print(f"\n追記できる件数: {len(additions)}")
    if dry_run:
        print("--yes を付けると実際に追記する（既存行は変更しない）")
        return 0

    with target.open("a", encoding="utf-8") as handle:
        for entry in additions:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"追記した: {target}")
    print("この後かならず実行すること:")
    print("  python -m pytest cognitag_struct/tests -q")
    print("  python -X utf8 cognitag_struct/eval/run_eval.py "
          "--compare cognitag_struct/eval/baseline.json")
    return len(additions)


# --- 入口 --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="カテゴリ辞書の候補を作る")
    parser.add_argument("--content", nargs="+", help="対象の content 値")
    parser.add_argument("--all", action="store_true",
                        help="axes.jsonl の content 値を全部")
    parser.add_argument(
        "--mode", choices=("list", "sentence", "dictionary"), default="list",
        help="list=語を列挙させる（少量・確実） / "
             "sentence=文を書かせて格から抜く（日常語・大量） / "
             "dictionary=CogniTag_V2 の語彙辞書から拾う（専門用語・大量・LLM 不要）",
    )
    parser.add_argument(
        "--source", default="D:/AIModel/CogniTag_V2/dictionary_v2.json",
        help="dictionary: 語彙辞書の場所",
    )
    parser.add_argument("--map", default=str(HERE / "category_map.toml"),
                        help="dictionary: カテゴリ名 → content の対応表")
    parser.add_argument("--share", type=float, default=0.5,
                        help="dictionary: 首位タグの占有率の下限")
    parser.add_argument("--votes", type=int, default=3,
                        help="dictionary: 首位タグの票数の下限")
    parser.add_argument("--rounds", type=int, default=3,
                        help="sentence: 場面を変えて何巡するか")
    parser.add_argument("--count", type=int, default=40, help="1 回に挙げさせる語数")
    parser.add_argument("--trials", type=int, default=TRIALS,
                        help="試行回数（1 回目は捨てる）")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--apply", metavar="FILE",
                        help="検収済み JSONL を phrases.jsonl へ追記")
    parser.add_argument("--yes", action="store_true", help="--apply を実行する")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.apply:
        return 0 if apply_entries(
            Path(args.apply), DATA / "phrases.jsonl", not args.yes
        ) >= 0 else 1

    existing = CategoryDict.load(DATA / "axes.jsonl", DATA / "phrases.jsonl")
    if args.all:
        targets = existing.axis_values(CONTENT_AXIS)
    elif args.content:
        targets = args.content
    elif args.mode in ("sentence", "dictionary"):
        targets = []          # 省略可。フレームまたは写像表が決める
    else:
        parser.error("--content か --all のどちらかが要る")

    known = set(existing.axis_values(CONTENT_AXIS))
    unknown = [t for t in targets if t not in known]
    if unknown:
        print(f"axes.jsonl に無い content 値: {unknown}")
        print("先に axes.jsonl の content へ追加すること（軸は増やさない）")
        return 2

    validator = Validator(existing)
    all_candidates: list[Candidate] = []
    if args.mode == "dictionary":
        mapping = load_mapping(Path(args.map), known)
        if mapping is None:
            return 2
        all_candidates = collect_from_dictionary(
            Path(args.source), mapping, args.share, args.votes,
            validator, set(targets) or None,
        )
    elif args.mode == "sentence":
        all_candidates = collect_by_sentence(
            args.rounds, args.count, args.timeout, validator,
            set(targets) or None, args.verbose,
        )
    else:
        for content in targets:
            print(f"\n[{content}]")
            all_candidates += collect(
                content, args.count, args.trials, args.timeout, validator
            )

    if not all_candidates:
        print("\n候補が 1 件も得られなかった。Ollama が動いているか確認すること。")
        return 1

    stem = "_".join(SLUGS.get(t, t) for t in targets)[:40] or "out"
    jsonl, tsv = write_outputs(all_candidates, stem)

    accepted = [c for c in all_candidates if c.accepted]
    rejected = [c for c in all_candidates if not c.accepted]
    print(f"\n候補 {len(all_candidates)} / 採用 {len(accepted)} / 除外 {len(rejected)}")
    reasons: Counter[str] = Counter()
    for candidate in rejected:
        for reason in candidate.reasons:
            reasons[reason.split("（")[0]] += 1
    for reason, count in reasons.most_common():
        print(f"  除外理由 {reason}: {count}")
    print(f"\n候補   {jsonl}")
    print(f"検収用 {tsv}")
    print("\nTSV を目で見て、誤りの行を候補 JSONL から消してから:")
    print(f"  python -X utf8 {Path(__file__).name} --apply {jsonl} --yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
