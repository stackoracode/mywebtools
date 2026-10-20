from mywebtools.i18n import join_words, only_latin_english, uses_cjk
from mywebtools.textclean import clean_text


def test_ascii_only_strips_non_ascii():
    assert clean_text("Héllo  wörld", ascii_only=True) == "Hllo wrld"


def test_cjk_text_survives_by_default():
    assert clean_text("你好，世界  Hello") == "你好,世界 Hello"  # NFKC turns the full-width comma into ','


def test_remove_dates_english_and_cjk():
    text = "Meeting March 5, 2024 at 10:30 then 2024年6月20日 下午3:15 and 2024-06-20"
    cleaned = clean_text(text, remove_dates=True)
    for token in ("March", "2024", "10:30", "6月", "3:15", "06-20"):
        assert token not in cleaned
    assert "Meeting" in cleaned


def test_join_words_cjk_vs_latin():
    assert join_words(["你", "好", "世界"]) == "你好世界"
    assert join_words(["Hello", "world"]) == "Hello world"
    assert join_words(["iPhone", "拍摄"]) == "iPhone 拍摄"


def test_language_helpers():
    assert only_latin_english(["eng"])
    assert not only_latin_english(["eng", "chi_sim"])
    assert uses_cjk(["chi_sim"])


def test_years_next_to_cjk_are_removed():
    assert clean_text("拍摄于2024夏天 20240620", remove_dates=True) == "拍摄于 夏天"
