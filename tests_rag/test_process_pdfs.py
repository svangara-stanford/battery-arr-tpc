"""Offline tests for the non-content page classification logic in process_pdfs.py."""

from battery_aar.rag.scripts.process_pdfs import (
    absorb_sandwiched_pages,
    classify_page,
    clean_text,
    digit_token_ratio,
    find_body_start,
    find_boilerplate,
    norm_header,
    page_comma_density,
    salvage_leading_content,
)


def _page(paragraphs: list[str]) -> tuple[list[str], str]:
    return paragraphs, "\n\n".join(paragraphs)


def test_clean_text_folds_ligatures_and_rejoins_hyphenation():
    assert clean_text("deﬁnition of elec-\ntrode behavior") == "definition of electrode behavior"


def test_clean_text_collapses_whitespace():
    assert clean_text("too   many    spaces\nand\nlines") == "too many spaces and lines"


def test_norm_header_strips_leading_and_trailing_page_numbers():
    assert norm_header(["232 References"]) == "references"
    assert norm_header(["References 233"]) == "references"


def test_norm_header_keeps_short_keyword_headers_intact():
    # Regression: a naive digit-strip must not eat the leading letter of "Index".
    assert norm_header(["Index"]) == "index"


def test_norm_header_skips_a_bare_page_number_block():
    assert norm_header(["XIII", "Contents"]) == "contents"


def test_norm_header_returns_empty_when_nothing_usable():
    assert norm_header(["42", "7"]) == ""


def test_page_comma_density_and_digit_token_ratio():
    text = "alpha, 12 beta, 45"
    assert page_comma_density(text) == 50.0  # 2 commas / 4 words * 100
    assert digit_token_ratio(text) == 0.5  # "12" and "45" out of 4 tokens


def test_classify_page_blank():
    assert classify_page(*_page([])) == "blank"


def test_classify_page_preface():
    assert classify_page(*_page(["Preface", "Thanks to everyone who helped."])) == "preface"


def test_classify_page_about_author_before_frontmatter_check():
    # "about the author" must win over the broader "authors" frontmatter cue.
    assert classify_page(*_page(["About the Author", "Jane Doe is a scientist."])) == "about"


def test_classify_page_frontmatter_companion_website():
    assert classify_page(*_page(["About the Companion Website", "Visit example.com"])) == "frontmatter"


def test_classify_page_toc_low_comma_density():
    paragraphs = ["Contents", "1 Introduction 1", "2 Batteries 45"]
    assert classify_page(*_page(paragraphs)) == "toc"


def test_classify_page_index_by_comma_density():
    paragraphs = [
        "Index",
        "Alpha, 12 Beta, 45 Gamma, 88 Delta, 21 Epsilon, 5 Zeta, 9 Eta, 3 Theta, 4",
    ]
    assert classify_page(*_page(paragraphs)) == "index"


def test_classify_page_references_by_header():
    paragraphs = ["References", "Smith et al. 2020. Some paper title."]
    assert classify_page(*_page(paragraphs)) == "references"


def test_classify_page_references_by_citation_density_without_header():
    paragraphs = [
        "Continuing Discussion",
        "[1] A, B, C. [2] D, E, F. [3] G, H, I. [4] J, K, L. "
        "[5] M, N, O. [6] P, Q, R. [7] S, T, U. [8] V, W, X.",
    ]
    assert classify_page(*_page(paragraphs)) == "references"


def test_classify_page_glossary():
    assert classify_page(*_page(["Glossary", "SEI: solid electrolyte interphase"])) == "glossary"


def test_classify_page_plain_content():
    paragraphs = [
        "3.2 Terms Used for Charging",
        "The current and voltage curves for such discharges are shown in Figure 3.1.",
    ]
    assert classify_page(*_page(paragraphs)) == "content"


def test_salvage_leading_content_keeps_prose_above_heading():
    long_prose = (
        "This is a long paragraph of chapter prose that continues for a while to make "
        "sure the char count clears the two hundred character minimum threshold required "
        "by the salvage function so it counts as legitimate content rather than being "
        "discarded as too short to bother keeping. It talks about batteries in detail."
    )
    paragraphs = ["Running header", long_prose, "References", "[1] Smith 2020", "[2] Doe 2021"]
    result = salvage_leading_content("references", paragraphs)
    assert result is not None
    assert "Running header" in result
    assert long_prose in result
    assert "References" not in result
    assert "[1] Smith 2020" not in result


def test_salvage_leading_content_none_when_heading_at_top():
    paragraphs = ["References", "[1] Smith 2020"]
    assert salvage_leading_content("references", paragraphs) is None


def test_salvage_leading_content_none_when_leading_prose_too_short():
    paragraphs = ["Hi", "References", "[1] Smith 2020"]
    assert salvage_leading_content("references", paragraphs) is None


def test_salvage_leading_content_none_when_no_bare_heading_found():
    paragraphs = ["Some chapter text with no standalone heading paragraph.", "[1] A", "[2] B"]
    assert salvage_leading_content("references", paragraphs) is None


def test_salvage_leading_content_none_for_non_split_label():
    assert salvage_leading_content("glossary", ["Glossary", "term: definition"]) is None


def test_absorb_sandwiched_pages_relabels_lone_content_page():
    labels = ["preface", "content", "preface"]
    assert absorb_sandwiched_pages(labels) == ["preface", "preface", "preface"]


def test_absorb_sandwiched_pages_leaves_mismatched_neighbors_alone():
    labels = ["preface", "content", "toc"]
    assert absorb_sandwiched_pages(labels) == ["preface", "content", "toc"]


def test_find_body_start_skips_short_copyright_pages():
    labels = ["content", "content", "content"]
    pages = [[], [], []]
    texts = [
        "All rights reserved. ISBN 123. Library of Congress data.",
        "Real chapter prose. " * 20,  # long, low digit ratio -> body anchor
        "More chapter prose.",
    ]
    assert find_body_start(labels, pages, texts) == 1


def test_find_body_start_falls_back_to_zero_when_nothing_qualifies():
    labels = ["content"]
    pages = [[]]
    texts = ["too short"]
    assert find_body_start(labels, pages, texts) == 0


def test_find_boilerplate_flags_watermark_repeated_across_pages():
    watermark = "Downloaded from Wiley Online Library"
    pages = [[watermark, f"unique text {i}"] for i in range(5)]
    boilerplate = find_boilerplate(pages)
    assert watermark in boilerplate
    assert "unique text 0" not in boilerplate


def test_find_boilerplate_ignores_text_below_repeat_threshold():
    pages = [["shared once"], ["different"], ["also different"]]
    assert find_boilerplate(pages) == set()
