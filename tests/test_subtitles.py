import asyncio
import json

import httpx
import pytest

from mywebtools import subtitles
from mywebtools.errors import ToolError
from mywebtools.llm import LLM
from mywebtools.subtitles import Cue, SubStyle


def test_wrap_cjk_and_latin():
    assert subtitles.wrap("今天我们去海边拍一条视频吧朋友们", 8) == ["今天我们去海边拍", "一条视频吧朋友们"]
    assert subtitles.wrap("the quick brown fox jumps over the lazy dog", 16) == ["the quick brown", "fox jumps over", "the lazy dog"]


def test_split_script_by_sentences():
    chunks = subtitles.split_script("大家好。今天去海边！\nHello there. How are you?", 20)
    assert chunks == ["大家好。", "今天去海边！", "Hello there.", "How are you?"]


def test_align_script_follows_speech_timing():
    segments = [{"start": 1.0, "end": 2.0, "text": "aaaa"}, {"start": 5.0, "end": 6.0, "text": "bbbb"}]
    cues = subtitles.align_script(["AAAA", "BBBB"], segments, 6.0)
    assert cues[0].start == pytest.approx(1.0) and cues[0].end == pytest.approx(2.0)
    assert cues[1].start == pytest.approx(5.0, abs=0.01) and cues[1].end == pytest.approx(6.0)


def test_align_without_segments_is_proportional():
    cues = subtitles.align_script(["ab", "abcdef"], [], 8.0)
    assert cues[0].end == pytest.approx(2.0) and cues[1].end == pytest.approx(8.0)


def test_fit_cues_splits_long_lines_in_time():
    cues = subtitles.fit_cues([Cue(0, 6, "一二三四五六七八九十一二三四五六七八")], 3, max_lines=2)
    assert len(cues) == 3 and cues[0].start == 0 and cues[-1].end == pytest.approx(6)
    assert all(len(c.text.split("\n")) <= 2 for c in cues)


def test_srt_roundtrip_and_offset():
    cues = [Cue(0.5, 2.25, "Hi"), Cue(3, 4, "两行\n字幕")]
    text = subtitles.to_srt(cues, offset=10)
    assert "00:00:10,500 --> 00:00:12,250" in text
    back = subtitles.parse_srt(text)
    assert [(c.start, c.end, c.text) for c in back] == [(10.5, 12.25, "Hi"), (13.0, 14.0, "两行\n字幕")]


def test_ass_style_and_escaping():
    ass = subtitles.to_ass([Cue(1, 2, "a{b}\nc")], SubStyle(size=70, color="#ff0000", box=True), 1080, 1920, offset=1)
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    assert ",70,&H000000FF," in ass  # red text in &HAABBGGRR
    assert "Dialogue: 0,0:00:02.00,0:00:03.00,Default,,0,0,0,,a｛b｝\\Nc" in ass
    assert f",{int(1920 * 0.24)},1\n" in ass  # bottom margin clears the Shorts caption area


def _llm(reply):
    def handler(request):
        cues = json.loads(json.loads(request.content)["input"])["cues"]
        return httpx.Response(200, json={"output_text": json.dumps({"cues": reply(cues)}, ensure_ascii=False)})

    return LLM(base_url="https://x/v1", api_key="k", model="m", effort="", transport=httpx.MockTransport(handler))


def test_ai_edit_keeps_timing_and_bilingual():
    cues = [Cue(0, 1, "helo"), Cue(1, 2, "wrld")]
    fixed = asyncio.run(subtitles.ai_edit(cues, "fix", llm=_llm(lambda c: ["hello", "world"])))
    assert [(c.start, c.text) for c in fixed] == [(0, "hello"), (1, "world")]
    both = asyncio.run(subtitles.ai_edit(fixed, "bilingual", target="zh-Hans", llm=_llm(lambda c: ["你好", "世界"])))
    assert both[0].text == "hello\n你好"


def test_ai_edit_rejects_count_mismatch():
    with pytest.raises(ToolError) as exc:
        asyncio.run(subtitles.ai_edit([Cue(0, 1, "a"), Cue(1, 2, "b")], "fix", llm=_llm(lambda c: ["ab"])))
    assert exc.value.code == "llm.count_mismatch"


def test_auto_max_chars_depends_on_script():
    assert subtitles.auto_max_chars("中文字幕测试", 1080, 64) < subtitles.auto_max_chars("English subtitle", 1080, 64)


def test_alignment_ignores_punctuation():
    segments = [{"start": 0.0, "end": 1.0, "text": "大家好"}, {"start": 2.0, "end": 3.0, "text": "今天去海边"}]
    cues = subtitles.align_script(["大家好。", "今天去海边！"], segments, 3.0)
    assert cues[1].start == pytest.approx(2.0, abs=0.01)
