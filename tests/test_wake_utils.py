import unittest

from utils.wake_utils import recent_text_from_history, is_wake_in_text, squash_repeated_tokens


class TestWakeUtils(unittest.TestCase):
    def test_recent_text_window(self):
        now = 100.0
        hist = [("A", now - 3.0), ("B", now - 2.0), ("C", now - 0.1)]
        self.assertEqual(recent_text_from_history(hist, now, 2.5), "BC")

    def test_require_both_order_ok(self):
        text = "ノイズもしもし...なんか...サイテクです"
        self.assertTrue(is_wake_in_text(text, require_both=True))

    def test_require_both_wrong_order(self):
        text = "サイテクと言った後にもしもし"
        self.assertFalse(is_wake_in_text(text, require_both=True))

    def test_require_any(self):
        self.assertTrue(is_wake_in_text("サイテクだけ", require_both=False))
        self.assertTrue(is_wake_in_text("もしもしだけ", require_both=False))

    def test_empty_text(self):
        self.assertFalse(is_wake_in_text("", require_both=True))

    def test_squash_repeated_tokens_ascii(self):
        s = "a b b b c  c   d"
        self.assertEqual(squash_repeated_tokens(s), "a b c d")

    def test_squash_repeated_tokens_jp(self):
        s = "し ましし ましし ましし もしもし"
        self.assertEqual(squash_repeated_tokens(s), "し ましし もしもし")


if __name__ == "__main__":
    unittest.main()
