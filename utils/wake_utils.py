import re
from typing import List, Tuple, Optional


DEFAULT_ALT_REGEX = r"(サイテク|さいてく|ｻｲﾃｸ|さいテク|サイテック|サイトク|さいとく)"


def recent_text_from_history(text_history: List[Tuple[str, float]], now: float, window_sec: float) -> str:
    """時刻付きテキスト履歴から時間窓内の文字列を連結して返す。"""
    return "".join(text for text, ts in text_history if now - ts <= window_sec)


def is_wake_in_text(text: str, require_both: bool = True, alt_regex: str = DEFAULT_ALT_REGEX) -> bool:
    """テキスト中にWake Wordが含まれるかを判定。

    - require_both=True の場合: 「もしもし」に続いて alt_regex にマッチがあること。
    - require_both=False の場合: 「もしもし」か alt_regex のどちらかがあればTrue。
    """
    if not text:
        return False
    if require_both:
        pattern = rf"もしもし.*{alt_regex}"
        return re.search(pattern, text) is not None
    return (re.search(r"もしもし", text) is not None) or (re.search(alt_regex, text) is not None)


def find_wake_match(text: str, require_both: bool = True, alt_regex: str = DEFAULT_ALT_REGEX) -> Optional[str]:
    """テキスト中のWake一致部分（スニペット）を返す。なければNone。
    - require_both=True の場合は『もしもし ... サイテク系』の一致区間
    - False の場合は『もしもし』または『サイテク系』の一致区間
    """
    if not text:
        return None
    if require_both:
        m = re.search(rf"(もしもし.*?{alt_regex})", text)
        return m.group(1) if m else None
    m1 = re.search(r"(もしもし)", text)
    if m1:
        return m1.group(1)
    m2 = re.search(rf"({alt_regex})", text)
    return m2.group(1) if m2 else None


def squash_repeated_tokens(text: str) -> str:
    """空白区切りトークンの連続重複を1つに圧縮して返す（ログ表示用）。
    例: 'a b b  b c' -> 'a b c'
    """
    if not text:
        return text
    tokens = text.split()
    if not tokens:
        return ""
    out = [tokens[0]]
    for tok in tokens[1:]:
        if tok != out[-1]:
            out.append(tok)
    return " ".join(out)
