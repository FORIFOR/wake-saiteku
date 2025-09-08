#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
"""
audio_device_manager.py (macOS / Raspberry Pi)
- 安全な音声デバイス自動選択＆健全性検証
- 優先度: Bluetooth → USB → Built-in → Other（WAKE_AUDIO_PREFER で変更可）
- 厳密ヘルス: 実際にストリームを開けるデバイスのみ採用（仮承認なし）
- macOS では既定で Bluetooth の「入出力同時オープン検証」を無効化
  （HFP入力16k + A2DP出力48k を許容）。必要なら WAKE_AUDIO_REQUIRE_BT_DUPLEX=1
- 名前優先（OpenComm2 等）、対話選択、設定保存(~/.wake-saiteku/audio.json)

依存:
  pip install sounddevice numpy
"""

from __future__ import annotations
import os
import json
import time
import platform
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import sounddevice as sd

# ====== 設定保存 ======
CFG_DIR = os.path.join(os.path.expanduser("~"), ".wake-saiteku")
CFG_FILE = os.path.join(CFG_DIR, "audio.json")

# ====== デバイス分類ヒント ======
BT_HINTS = [
    "airpods", "bose", "shokz", "opencomm", "sony", "jabra", "anker", "beats",
    "bluetooth", "bt-w", "hfp", "hsp"
]
USB_HINTS = ["usb", "uac", "scarlett", "focusrite", "audio codec", "uhs", "ust"]
BUILTIN_HINTS = ["built-in", "internal speaker", "internal microphone", "default"]

# 環境差が大きいので既定ブロックなし（必要に応じて WAKE_AUDIO_BLOCKLIST で設定）
BLOCKLIST_DEFAULT: List[str] = []

@dataclass
class Dev:
    index: int
    name: str
    hostapi: str
    max_in: int
    max_out: int
    default_sr: float
    category: str  # "bluetooth" | "usb" | "builtin" | "other"

    def short(self) -> str:
        io = []
        if self.max_in > 0:
            io.append(f"IN:{self.max_in}")
        if self.max_out > 0:
            io.append(f"OUT:{self.max_out}")
        return f"[{self.index}] {self.name} — {self.hostapi} ({'/'.join(io) or 'no-io'}, {self.category}, {int(self.default_sr)}Hz)"

def _hostapi_name(idx: int) -> str:
    try:
        apis = sd.query_hostapis()
        return apis[idx]["name"] if 0 <= idx < len(apis) else "unknown"
    except Exception:
        return "unknown"

def _categorize(name: str, hostapi: str) -> str:
    n = name.lower(); ha = hostapi.lower()
    if any(k in n for k in BT_HINTS) or "bluetooth" in ha:
        return "bluetooth"
    if any(k in n for k in USB_HINTS):
        return "usb"
    if any(k in n for k in BUILTIN_HINTS) or ("core audio" in ha and "built" in n):
        return "builtin"
    return "other"

def enumerate_devices(blocklist: Optional[List[str]] = None) -> List[Dev]:
    bl = [b.lower() for b in (blocklist or [])]
    raw = sd.query_devices()
    results: List[Dev] = []
    for idx, d in enumerate(raw):
        name = d.get("name", "?")
        if any(b in name.lower() for b in bl):
            continue
        ha = _hostapi_name(d.get("hostapi", -1))
        results.append(
            Dev(
                index=idx,
                name=name,
                hostapi=ha,
                max_in=int(d.get("max_input_channels", 0)),
                max_out=int(d.get("max_output_channels", 0)),
                default_sr=float(d.get("default_samplerate", 48000.0)),
                category=_categorize(name, ha),
            )
        )
    return results

# ====== 健康プローブ ======
def _probe_input(dev: Dev, samplerate: int) -> Tuple[bool, Optional[str]]:
    if dev.max_in <= 0:
        return False, "no input channels"
    try:
        with sd.InputStream(
            device=dev.index,
            channels=1,
            samplerate=samplerate,  # 例: 16000（HFP/HSP）
            dtype="int16",
            blocksize=0,
        ):
            sd.sleep(150)
        return True, None
    except Exception as e:
        return False, str(e)

def _probe_output(dev: Dev, samplerate_hint: int) -> Tuple[bool, Optional[str]]:
    if dev.max_out <= 0:
        return False, "no output channels"
    # 出力はデバイス既定 SR を優先（A2DP は概ね 44100/48000）
    sr_candidates = [int(dev.default_sr)]
    if int(dev.default_sr) not in (44100, 48000):
        sr_candidates += [48000, 44100]
    # 最後の手段としてヒント SR も試す
    if samplerate_hint not in sr_candidates:
        sr_candidates.append(samplerate_hint)

    last_err = None
    for sr in sr_candidates:
        try:
            with sd.OutputStream(
                device=dev.index,
                channels=1,
                samplerate=sr,
                dtype="int16",
                blocksize=0,
            ) as s:
                s.write(np.zeros((max(1, sr // 20), 1), dtype=np.int16))  # 50ms 無音
            return True, None
        except Exception as e:
            last_err = e
            continue
    return False, (str(last_err) if last_err else "unknown")

def _probe_duplex(in_dev: Dev, out_dev: Dev, samplerate: int) -> Tuple[bool, Optional[str]]:
    """入出力同時オープン（HFP/HSP 一体型など）。macOS で別デバイスの BT 入出力は失敗しやすい。"""
    try:
        with sd.Stream(
            samplerate=samplerate,
            dtype="int16",
            channels=(1, 1),
            blocksize=0,
            device=(in_dev.index, out_dev.index),
        ):
            sd.sleep(120)
        return True, None
    except Exception as e:
        return False, str(e)

# ====== スコアリングと優先度 ======
def _score(dev: Dev, kind: str) -> int:
    base = {"bluetooth": 300, "usb": 200, "builtin": 100, "other": 50}.get(dev.category, 50)
    ch = dev.max_in if kind == "in" else dev.max_out
    return base + min(ch, 2) * 5

def _prefer_order() -> List[str]:
    env = os.getenv("WAKE_AUDIO_PREFER", "bluetooth,usb,builtin,other").lower()
    return [x.strip() for x in env.split(",") if x.strip()]

# ====== 保存/復元 ======
def _load_saved_choice() -> Optional[dict]:
    try:
        with open(CFG_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return None

def _save_choice(entry: dict) -> None:
    os.makedirs(CFG_DIR, exist_ok=True)
    with open(CFG_FILE, "w") as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)

def _find_by_signature(devs: List[Dev], sig: dict) -> Optional[int]:
    for d in devs:
        if d.name == sig.get("name") and d.hostapi == sig.get("hostapi"):
            if sig.get("role") == "input" and d.max_in > 0:
                return d.index
            if sig.get("role") == "output" and d.max_out > 0:
                return d.index
    return None

# ====== 名前マッチ & 対話 ======
def _split_env_names(key: str) -> List[str]:
    v = os.getenv(key, "")
    return [s.strip().lower() for s in v.split(",") if s.strip()]

def _name_match(dev: Dev, patterns: List[str]) -> bool:
    if not patterns:
        return False
    n = dev.name.lower()
    return any(p in n for p in patterns)

def _interactive_pick(title: str, candidates: List[Dev], kind: str, samplerate: int) -> Optional[int]:
    print(f"\n=== {title} ({kind}) ===")
    for d in candidates:
        ok, err = (_probe_input(d, samplerate) if kind == "in" else _probe_output(d, samplerate))
        mark = "✅" if ok else "❌"
        reason = "" if ok else f"  ({err})"
        print(f"{mark} {d.short()}{reason}")
    try:
        raw = input("番号を入力（Enterでスキップ）: ").strip()
        if not raw:
            return None
        return int(raw)
    except Exception:
        return None

# ====== メイン選択ロジック ======
def choose_devices(
    samplerate: int = 16000,
    blocklist: Optional[List[str]] = None,
    allow_interactive: bool = False,
    remember: bool = True,
    strict_health: bool = True,
    require_pair_for_bt: Optional[bool] = None,
) -> Tuple[Optional[int], Optional[int]]:
    """
    入力/出力デバイス index を返す（見つからなければ None）。
    - strict_health=True: プローブに通るデバイスのみ採用（仮承認なし）
    - require_pair_for_bt:
        * None（既定）: macOS では False、その他 OS では True
        * True: BT を含む場合、入出力同時オープンに成功したペアのみ採用
        * False: 片側ごとの健康判定のみ（macOS ではこちらが現実的）
    """
    if require_pair_for_bt is None:
        require_pair_for_bt = (platform.system().lower() != "darwin")
    env_force_pair = os.getenv("WAKE_AUDIO_REQUIRE_BT_DUPLEX", "")
    if env_force_pair:
        require_pair_for_bt = env_force_pair.lower() in {"1", "true", "yes", "on"}

    # ブロックリスト
    block = (blocklist or []) + BLOCKLIST_DEFAULT
    devs = enumerate_devices(block)

    # index/name 強制（ただし健康に通らないなら無効）
    def _resolve_env(raw: Optional[str], for_in: bool) -> Optional[int]:
        if not raw:
            return None
        raw = raw.strip()
        if raw.isdigit():
            idx = int(raw)
            d = next((x for x in devs if x.index == idx), None)
            if not d:
                return None
            ok = _probe_input(d, samplerate)[0] if for_in else _probe_output(d, samplerate)[0]
            return idx if ok else None
        cand = [x for x in devs if (x.max_in > 0 if for_in else x.max_out > 0) and raw.lower() in x.name.lower()]
        for d in sorted(cand, key=lambda x: -_score(x, "in" if for_in else "out")):
            ok = _probe_input(d, samplerate)[0] if for_in else _probe_output(d, samplerate)[0]
            if ok:
                return d.index
        return None

    env_in_raw = os.getenv("WAKE_AUDIO_INPUT") or os.getenv("AUDIO_INPUT_DEVICE")
    env_out_raw = os.getenv("WAKE_AUDIO_OUTPUT") or os.getenv("AUDIO_OUTPUT_DEVICE")
    in_idx = _resolve_env(env_in_raw, True)
    out_idx = _resolve_env(env_out_raw, False)

    # 保存済み復元
    if in_idx is None or out_idx is None:
        saved = _load_saved_choice() or {}
        if in_idx is None and "input" in saved:
            cand = _find_by_signature(devs, saved["input"]); in_idx = cand if cand is not None else in_idx
        if out_idx is None and "output" in saved:
            cand = _find_by_signature(devs, saved["output"]); out_idx = cand if cand is not None else out_idx

    # 健康判定ヘルパ
    def usable_in(d: Dev) -> bool:
        return _probe_input(d, samplerate)[0]
    def usable_out(d: Dev) -> bool:
        return _probe_output(d, samplerate)[0]

    # 名前優先
    pref_in_names = _split_env_names("WAKE_AUDIO_PREF_IN_NAME")
    pref_out_names = _split_env_names("WAKE_AUDIO_PREF_OUT_NAME")
    pair_same = os.getenv("WAKE_AUDIO_PAIR_SAME_NAME", "1").lower() in {"1", "true", "yes", "on"}

    # カテゴリ分配
    order = _prefer_order()
    cats_in = {c: [] for c in order}; cats_out = {c: [] for c in order}
    others_in: List[Dev] = []; others_out: List[Dev] = []
    for d in devs:
        if d.max_in > 0:
            (cats_in.get(d.category) or others_in).append(d) if d.category in cats_in else others_in.append(d)
        if d.max_out > 0:
            (cats_out.get(d.category) or others_out).append(d) if d.category in cats_out else others_out.append(d)

    def pick_best(cats: dict, others: List[Dev], kind: str) -> Optional[int]:
        for cat in order:
            pool = sorted(cats.get(cat, []), key=lambda x: -_score(x, kind))
            for d in pool:
                if (usable_in(d) if kind == "in" else usable_out(d)):
                    return d.index
        for d in sorted(others, key=lambda x: -_score(x, kind)):
            if (usable_in(d) if kind == "in" else usable_out(d)):
                return d.index
        return None

    # --- 名前優先（まず同名ペアを強く優先） ---
    if in_idx is None and pref_in_names:
        if pair_same and not pref_out_names:
            for din in sorted([d for d in devs if d.max_in > 0 and _name_match(d, pref_in_names)],
                              key=lambda x: -_score(x, "in")):
                if not usable_in(din):
                    continue
                paired_outs = [d for d in devs if d.max_out > 0 and d.name == din.name]
                for dout in sorted(paired_outs, key=lambda x: -_score(x, "out")):
                    if usable_out(dout):
                        in_idx = din.index
                        out_idx = dout.index if out_idx is None else out_idx
                        break
                if in_idx is not None:
                    break
        if in_idx is None:
            for din in sorted([d for d in devs if d.max_in > 0], key=lambda x: -_score(x, "in")):
                if _name_match(din, pref_in_names) and usable_in(din):
                    in_idx = din.index
                    break

    if out_idx is None and pref_out_names:
        for dout in sorted([d for d in devs if d.max_out > 0], key=lambda x: -_score(x, "out")):
            if _name_match(dout, pref_out_names) and usable_out(dout):
                out_idx = dout.index
                break

    # --- カテゴリ優先（通常ロジック） ---
    if in_idx is None:
        in_idx = pick_best(cats_in, others_in, "in")
    if out_idx is None:
        out_idx = pick_best(cats_out, others_out, "out")

    # 対話選択
    if allow_interactive:
        in_cands = [d for d in devs if d.max_in > 0]
        out_cands = [d for d in devs if d.max_out > 0]
        picked_in = _interactive_pick("入力デバイスを選択", in_cands, "in", samplerate)
        if picked_in is not None:
            in_idx = picked_in
        picked_out = _interactive_pick("出力デバイスを選択", out_cands, "out", samplerate)
        if picked_out is not None:
            out_idx = picked_out

    # --- 最終ヘルス審査（厳密） ---
    din = next((d for d in devs if d.index == in_idx), None) if in_idx is not None else None
    dout = next((d for d in devs if d.index == out_idx), None) if out_idx is not None else None

    # 個別ヘルス
    if din and not usable_in(din):
        in_idx = None; din = None
    if dout and not usable_out(dout):
        out_idx = None; dout = None

    # BT を含むときのデュープレックス検証
    if require_pair_for_bt and din and dout and (din.category == "bluetooth" or dout.category == "bluetooth"):
        ok, _ = _probe_duplex(din, dout, samplerate)
        if not ok:
            # 出力を他へ（入力優先）。それでもダメなら入力も他へ。
            alt_out = pick_best(cats_out, others_out, "out")
            if alt_out is not None and alt_out != out_idx:
                out_idx = alt_out
                dout = next((d for d in devs if d.index == out_idx), None)
                if din and dout:
                    ok2, _ = _probe_duplex(din, dout, samplerate)
                    if not ok2:
                        alt_in = pick_best(cats_in, others_in, "in")
                        if alt_in is not None:
                            in_idx = alt_in
                            din = next((d for d in devs if d.index == in_idx), None)
            # 個別ヘルス再確認
            if din and not usable_in(din): in_idx = None
            if dout and not usable_out(dout): out_idx = None

    # 保存
    if remember:
        sig = {}
        if in_idx is not None:
            din = next((d for d in devs if d.index == in_idx), None)
            if din: sig["input"] = {"name": din.name, "hostapi": din.hostapi, "role": "input"}
        if out_idx is not None:
            dout = next((d for d in devs if d.index == out_idx), None)
            if dout: sig["output"] = {"name": dout.name, "hostapi": dout.hostapi, "role": "output"}
        if sig: _save_choice(sig)

    return in_idx, out_idx

# ====== CLI ======
def _print_device_table(devs: List[Dev]) -> None:
    print("\n📱 利用可能なオーディオデバイス:")
    print("ID | 入力 | 出力 | デバイス名")
    print("-" * 60)
    for d in devs:
        print(f"{d.index:2d} | {d.max_in:4d} | {d.max_out:4d} | {d.name}")

def main():
    bl = [s.strip().lower() for s in os.getenv("WAKE_AUDIO_BLOCKLIST", "").split(",") if s.strip()]
    devs = enumerate_devices(bl)
    _print_device_table(devs)

    in_idx, out_idx = choose_devices(
        samplerate=int(os.getenv("WAKE_SAMPLE_RATE", "16000")),
        blocklist=bl,
        allow_interactive=os.getenv("WAKE_AUDIO_SELECT", "0").lower() in {"1", "true", "yes", "on"},
        remember=True,
        strict_health=True,
        require_pair_for_bt=None,  # macOS=False, その他=True（環境変数で上書き可）
    )

    def _name(idx: Optional[int]) -> str:
        if idx in (None, -1):
            return "なし"
        d = next((x for x in devs if x.index == idx), None)
        return d.name if d else f"(id {idx})"

    print(f"\n選択結果: 入力={in_idx}（{_name(in_idx)}）, 出力={out_idx}（{_name(out_idx)}）")

if __name__ == "__main__":
    print(f"OS: {platform.platform()}")
    try:
        print("Host APIs:", [ha["name"] for ha in sd.query_hostapis()])
    except Exception:
        pass
    main()