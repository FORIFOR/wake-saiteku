import re
from typing import List, Tuple


def _minimal_repeating_unit(s: str) -> Tuple[str, int]:
    """最小の繰り返し単位と繰り返し回数を返す。
    例: 'abcabcabc' -> ('abc', 3), 'aaaa' -> ('a', 4), 'abcab' -> ('abcab', 1)
    """
    n = len(s)
    if n == 0:
        return s, 0
    # KMPのlps（部分一致テーブル）
    lps = [0] * n
    j = 0
    for i in range(1, n):
        while j > 0 and s[i] != s[j]:
            j = lps[j - 1]
        if s[i] == s[j]:
            j += 1
            lps[i] = j
    p = n - lps[-1]  # 推定周期
    if p != 0 and n % p == 0:
        unit = s[:p]
        times = n // p
        return unit, times
    return s, 1


def dedupe_transcript(text: str) -> str:
    """連続する同一文の重複を簡易除去する。

    - 句点や感嘆符、疑問符（全角/半角）で文っぽく分割し、
      隣接して同一の文が続く場合に2つ目以降を落とす。
    - ストリーミングASRの累積出力や重複結合対策の最小実装。
    """
    if not text:
        return text
    parts: List[str] = re.findall(r"[^。！？!?]+[。！？!?]?", text)
    out: List[str] = []
    prev = None
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if prev is not None and p == prev:
            continue
        out.append(p)
        prev = p
    result = "".join(out)

    # 句読点で切れなかった場合でも全文の繰り返しを1回に圧縮
    collapsed_no_space = "".join(result.split())
    unit, times = _minimal_repeating_unit(collapsed_no_space)
    # 短すぎる単位（誤圧縮）を避けるため、ある程度の長さのみを対象
    if times >= 2 and len(unit) >= 3:
        return unit
    return result
