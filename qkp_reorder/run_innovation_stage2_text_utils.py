from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import ours_utils  # noqa: E402
from qkp_reorder.ours_utils import (  # noqa: E402
    compute_protection_bonus,
    contains_cjk,
    extract_entity_like_terms,
    extract_numbers,
    extract_question_keywords,
    is_jieba_available,
    split_into_segments,
    tokenize_for_bm25,
)


LOG_PATH = PROJECT_ROOT / "logs" / "innovation" / "stage2_text_utils_log.txt"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    log_lines = [
        "Stage 2 text utility verification",
        f"checked_at={datetime.now().astimezone().isoformat(timespec='seconds')}",
        "No DeepSeek API call is made by this script.",
        f"jieba_available={is_jieba_available()}",
        "jieba_fallback_note=If jieba is unavailable, tokenize_for_bm25 falls back to regex English/number tokens and CJK character tokens.",
        "",
    ]

    english_text = "Paris is in France. It is a capital! NASA launched JWST.\nBM25 keeps relevant evidence?"
    english_segments = split_into_segments(english_text)
    _require(len(english_segments) == 4, "English sentence-like split should produce four segments.")
    _require(english_segments[0] == "Paris is in France.", "English split should preserve sentence text.")
    log_lines.append(f"english_segments={english_segments}")

    chinese_text = "北京是中国首都。它有故宫；游客很多，历史悠久。"
    chinese_segments = split_into_segments(chinese_text)
    _require(contains_cjk(chinese_text), "contains_cjk should detect Chinese text.")
    _require("北京是中国首都。" in chinese_segments, "Chinese period split missing.")
    _require("它有故宫；" in chinese_segments, "Chinese semicolon split missing.")
    _require("游客很多，" in chinese_segments, "Chinese comma split missing.")
    log_lines.append(f"chinese_segments={chinese_segments}")

    mixed_text = "Albert Einstein在1905年提出special relativity。NASA在2024年发布AI_report。"
    mixed_segments = split_into_segments(mixed_text)
    mixed_tokens = tokenize_for_bm25(mixed_text)
    _require(len(mixed_segments) == 2, "Mixed Chinese/English text should split on Chinese punctuation.")
    _require("albert" in mixed_tokens, "Mixed tokenization should keep English words.")
    _require("1905" in mixed_tokens and "2024" in mixed_tokens, "Mixed tokenization should keep numbers.")
    _require("ai_report" in mixed_tokens, "Mixed tokenization should keep underscore tokens.")
    _require(any(contains_cjk(token) for token in mixed_tokens), "Mixed tokenization should keep Chinese tokens.")
    log_lines.append(f"mixed_segments={mixed_segments}")
    log_lines.append(f"mixed_tokens={mixed_tokens}")

    original_jieba = ours_utils.jieba
    try:
        ours_utils.jieba = None
        fallback_tokens = tokenize_for_bm25("中国AI在2024年发布。")
    finally:
        ours_utils.jieba = original_jieba
    _require("ai" in fallback_tokens, "Fallback tokenization should keep English words.")
    _require("2024" in fallback_tokens, "Fallback tokenization should keep numbers.")
    _require(any(contains_cjk(token) for token in fallback_tokens), "Fallback tokenization should keep CJK characters.")
    log_lines.append(f"simulated_no_jieba_fallback_tokens={fallback_tokens}")

    english_question = "What year did Albert Einstein publish the special-relativity paper in 1905?"
    english_keywords = extract_question_keywords(english_question)
    _require("what" not in english_keywords, "English question stopwords should be filtered.")
    _require("albert" in english_keywords and "einstein" in english_keywords, "English keywords missing entity words.")
    _require("1905" in english_keywords, "Question keyword extraction should keep numbers.")
    log_lines.append(f"english_keywords={english_keywords}")

    chinese_question = "《三体》的作者是谁，获得了哪一年雨果奖？"
    chinese_keywords = extract_question_keywords(chinese_question)
    _require("谁" not in chinese_keywords and "哪一年" not in chinese_keywords, "Chinese question words should be filtered.")
    _require(any(keyword in chinese_keywords for keyword in ["三体", "作者", "雨果奖"]), "Chinese keywords missing expected content.")
    log_lines.append(f"chinese_keywords={chinese_keywords}")

    number_text = "In 2024, accuracy rose by 12.5% from 98 samples to 110.5 cases."
    numbers = extract_numbers(number_text)
    _require("2024" in numbers, "Number extraction should keep years.")
    _require("12.5%" in numbers, "Number extraction should keep percentages.")
    _require("110.5" in numbers, "Number extraction should keep decimals.")
    log_lines.append(f"numbers={numbers}")

    mixed_number_text = "NASA在2024年发布AI_report，增长12.5%。样本数为1,234。"
    mixed_numbers = extract_numbers(mixed_number_text)
    _require("2024" in mixed_numbers, "Number extraction should keep years adjacent to Chinese text.")
    _require("12.5%" in mixed_numbers, "Number extraction should keep percentages adjacent to Chinese text.")
    _require("1,234" in mixed_numbers, "Number extraction should keep comma-separated numbers.")
    log_lines.append(f"mixed_numbers={mixed_numbers}")

    entity_text = "Albert Einstein worked with NASA and OpenAI_team on long-context QA."
    entities = extract_entity_like_terms(entity_text)
    _require("Albert Einstein" in entities, "Entity extraction should keep capitalized name phrases.")
    _require("NASA" in entities, "Entity extraction should keep all-caps acronyms.")
    _require("OpenAI_team" in entities, "Entity extraction should keep underscore tokens.")
    _require("long-context" in entities, "Entity extraction should keep hyphenated tokens.")
    log_lines.append(f"entities={entities}")

    keywords = extract_question_keywords("What did NASA publish in 2024 about long-context QA?")
    protected_segment = "NASA published a long-context QA report in 2024 with 12.5% gains."
    unrelated_segment = "This paragraph describes cooking and weather."
    protected_bonus = compute_protection_bonus(protected_segment, keywords)
    unrelated_bonus = compute_protection_bonus(unrelated_segment, keywords)
    _require(protected_bonus > unrelated_bonus, "Protection bonus should prefer keyword/entity/number evidence.")
    _require(protected_bonus > 0, "Protection bonus should be positive for protected evidence.")
    log_lines.append(f"protection_keywords={keywords}")
    log_lines.append(f"protected_bonus={protected_bonus}")
    log_lines.append(f"unrelated_bonus={unrelated_bonus}")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print("\n".join(log_lines))


if __name__ == "__main__":
    main()

