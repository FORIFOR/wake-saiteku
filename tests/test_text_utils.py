import unittest

from utils.text_utils import dedupe_transcript


class TestDedupeTranscript(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(dedupe_transcript(""), "")

    def test_no_dup(self):
        s = "今日は雨です。明日は晴れ？よかったね！"
        self.assertEqual(dedupe_transcript(s), s)

    def test_simple_dup_question(self):
        s = "今日の天気は?今日の天気は?今日の天気は?"
        self.assertEqual(dedupe_transcript(s), "今日の天気は?")

    def test_mixed_dup(self):
        s = "A。A。B！B！C?C?D。"
        self.assertEqual(dedupe_transcript(s), "A。B！C?D。")

    def test_spaces_are_ignored_for_compare(self):
        # 余計なスペースはstripで同一視
        s = "A 。A。 A。A。"
        self.assertEqual(dedupe_transcript(s), "A。")

    def test_full_repeat_no_punct(self):
        # 句読点が無く全文が繰り返されているケースを1回へ圧縮
        s = "今日の天気は今日の天気は今日の天気は"
        self.assertEqual(dedupe_transcript(s), "今日の天気は")

    def test_space_separated_repeat(self):
        # スペース区切りの繰り返しも1回へ圧縮
        s = "おはよう おはよう おはよう"
        self.assertEqual(dedupe_transcript(s), "おはよう")


if __name__ == "__main__":
    unittest.main()
