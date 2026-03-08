#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


EPS = 1e-9


@dataclass
class Interval:
    xmin: float
    xmax: float
    text: str

    @property
    def dur(self) -> float:
        return self.xmax - self.xmin


@dataclass
class IntervalTier:
    name: str
    xmin: float
    xmax: float
    intervals: List[Interval]


@dataclass
class TextGrid:
    xmin: float
    xmax: float
    tiers: List[IntervalTier]

    def get_interval_tier(self, name: str) -> Optional[IntervalTier]:
        for t in self.tiers:
            if t.name == name:
                return t
        return None


_RE_FLOAT = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _parse_float_from_line(line: str) -> float:
    m = _RE_FLOAT.search(line)
    if not m:
        raise ValueError(f"float parse failed: {line!r}")
    return float(m.group(0))


def _parse_int_from_line(line: str) -> int:
    m = re.search(r"[-+]?\d+", line)
    if not m:
        raise ValueError(f"int parse failed: {line!r}")
    return int(m.group(0))


def _parse_quoted_text(line: str) -> str:
    # e.g. text = ""  / text = "tɕ"
    m = re.search(r'=\s*"(.*)"\s*$', line)
    if not m:
        raise ValueError(f"text parse failed: {line!r}")
    return m.group(1)


def _fmt_praat_number(x: float) -> str:
    # Input files are fairly simple (e.g., 0, 1.49, 6.838). We'll keep similar.
    if abs(x - round(x)) < 1e-12:
        return str(int(round(x)))
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s


def read_textgrid(path: str) -> TextGrid:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    # Basic header sanity
    if not any("Object class" in ln and "TextGrid" in ln for ln in lines[:10]):
        raise ValueError(f"Not a TextGrid text file: {path}")

    # Global xmin/xmax
    tg_xmin = None
    tg_xmax = None
    for ln in lines:
        if tg_xmin is None and re.match(r"^\s*xmin\s*=", ln):
            tg_xmin = _parse_float_from_line(ln)
        elif tg_xmax is None and re.match(r"^\s*xmax\s*=", ln):
            tg_xmax = _parse_float_from_line(ln)
        if tg_xmin is not None and tg_xmax is not None:
            break
    if tg_xmin is None or tg_xmax is None:
        raise ValueError(f"Failed to parse global xmin/xmax: {path}")

    # tier count: first "size = <int>" after "tiers? <exists>"
    tier_size = None
    for i, ln in enumerate(lines):
        if "tiers? <exists>" in ln:
            # search forward a little
            for j in range(i, min(i + 30, len(lines))):
                if re.match(r"^\s*size\s*=", lines[j]):
                    tier_size = _parse_int_from_line(lines[j])
                    break
            break
    if tier_size is None:
        raise ValueError(f"Failed to parse tier size: {path}")

    tiers: List[IntervalTier] = []

    # Parse tiers by scanning item [k]:
    i = 0
    item_re = re.compile(r"^\s*item\s*\[(\d+)\]\s*:\s*$")
    while i < len(lines):
        m = item_re.match(lines[i])
        if not m:
            i += 1
            continue

        # Parse one tier (assumed IntervalTier)
        # Expected fields exist in the next lines; allow blank lines.
        cls = None
        name = None
        xmin = None
        xmax = None
        n_intervals = None
        intervals: List[Interval] = []

        i += 1
        while i < len(lines):
            ln = lines[i]
            # Next tier starts
            if item_re.match(ln):
                break

            if cls is None and re.search(r'^\s*class\s*=\s*"', ln):
                cls = _parse_quoted_text(ln)
            elif name is None and re.search(r'^\s*name\s*=\s*"', ln):
                name = _parse_quoted_text(ln)
            elif xmin is None and re.match(r"^\s*xmin\s*=", ln):
                xmin = _parse_float_from_line(ln)
            elif xmax is None and re.match(r"^\s*xmax\s*=", ln):
                xmax = _parse_float_from_line(ln)
            elif n_intervals is None and re.search(r"intervals:\s*size\s*=", ln):
                n_intervals = _parse_int_from_line(ln)
            elif re.search(r"^\s*intervals\s*\[\d+\]\s*:\s*$", ln):
                # interval block: next 3 lines xmin/xmax/text (allow blanks between)
                # Move to next meaningful lines for xmin/xmax/text
                def _next_nonempty(idx: int) -> int:
                    while idx < len(lines) and lines[idx].strip() == "":
                        idx += 1
                    return idx

                j = _next_nonempty(i + 1)
                if j >= len(lines) or not re.match(r"^\s*xmin\s*=", lines[j]):
                    raise ValueError(f"Malformed interval xmin near line {j+1} in {path}")
                ixmin = _parse_float_from_line(lines[j])

                j = _next_nonempty(j + 1)
                if j >= len(lines) or not re.match(r"^\s*xmax\s*=", lines[j]):
                    raise ValueError(f"Malformed interval xmax near line {j+1} in {path}")
                ixmax = _parse_float_from_line(lines[j])

                j = _next_nonempty(j + 1)
                if j >= len(lines) or not re.match(r"^\s*text\s*=", lines[j]):
                    raise ValueError(f"Malformed interval text near line {j+1} in {path}")
                itxt = _parse_quoted_text(lines[j])

                intervals.append(Interval(ixmin, ixmax, itxt))
                i = j  # continue after text line
            i += 1

        if cls is None or name is None or xmin is None or xmax is None:
            raise ValueError(f"Malformed tier block in {path} (missing class/name/xmin/xmax)")
        if cls != "IntervalTier":
            raise ValueError(f"Unsupported tier class {cls!r} in {path} (expected IntervalTier)")
        if n_intervals is not None and len(intervals) != n_intervals:
            # Some files might have mismatch due to formatting; keep parsed count but warn via exception to be safe.
            raise ValueError(
                f"Interval count mismatch in {path}: header says {n_intervals}, parsed {len(intervals)}"
            )
        tiers.append(IntervalTier(name=name, xmin=xmin, xmax=xmax, intervals=intervals))

        # do not i += 1 here; loop continues from current i (already at next tier or end)

    if len(tiers) != tier_size:
        # Some TextGrids may include nested "item []:" plus "item [k]"—we only count actual tiers.
        # If mismatch, still proceed but ensure tiers parsed is non-empty.
        if len(tiers) == 0:
            raise ValueError(f"Parsed 0 tiers from {path} (expected {tier_size})")

    return TextGrid(xmin=tg_xmin, xmax=tg_xmax, tiers=tiers)


def write_textgrid(tg: TextGrid, path: str) -> None:
    # Canonical Praat TextGrid (text) output, similar to dataset formatting.
    out: List[str] = []
    out.append('File type = "ooTextFile"')
    out.append('Object class = "TextGrid"')
    out.append("")
    out.append(f"xmin = {_fmt_praat_number(tg.xmin)} ")
    out.append(f"xmax = {_fmt_praat_number(tg.xmax)} ")
    out.append("tiers? <exists> ")
    out.append(f"size = {len(tg.tiers)} ")
    out.append("item []: ")
    for ti, tier in enumerate(tg.tiers, start=1):
        out.append(f"    item [{ti}]:")
        out.append('        class = "IntervalTier" ')
        out.append(f'        name = "{tier.name}" ')
        out.append(f"        xmin = {_fmt_praat_number(tier.xmin)} ")
        out.append(f"        xmax = {_fmt_praat_number(tier.xmax)} ")
        out.append(f"        intervals: size = {len(tier.intervals)} ")
        for ii, itv in enumerate(tier.intervals, start=1):
            out.append(f"        intervals [{ii}]:")
            out.append(f"            xmin = {_fmt_praat_number(itv.xmin)} ")
            out.append(f"            xmax = {_fmt_praat_number(itv.xmax)} ")
            # escape quotes if any (rare)
            txt = itv.text.replace('"', r"\"")
            out.append(f'            text = "{txt}" ')

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")
    os.replace(tmp, path)


def fix_last_phone_interval(
    tg: TextGrid,
    tier_name: str,
    fixed_dur: float,
    min_dur: float,
    silence_labels: Sequence[str],
) -> Tuple[bool, str]:
    """
    Returns (changed, reason).
    """
    tier = tg.get_interval_tier(tier_name)
    if tier is None:
        return False, f"missing_tier:{tier_name}"
    if not tier.intervals:
        return False, "empty_intervals"

    last = tier.intervals[-1]
    last_text = last.text
    last_dur = last.dur
    if last_text in silence_labels:
        return False, "already_ends_with_silence"
    if last_dur + EPS < min_dur:
        return False, f"last_dur<{min_dur}"

    # Only meaningful if we can carve out fixed_dur and leave remainder for silence.
    if last_dur <= fixed_dur + EPS:
        return False, f"last_dur<=fixed_dur({fixed_dur})"

    old_start, old_end = last.xmin, last.xmax
    new_end = old_start + fixed_dur
    if new_end > old_end:
        new_end = old_end

    # Apply
    last.xmax = new_end
    rem = old_end - new_end
    if rem > EPS:
        tier.intervals.append(Interval(new_end, old_end, ""))
    # Keep tier xmax unchanged (should already equal old_end in most cases)
    # but do not force; writing keeps the tier header values as stored.
    return True, f"fixed_last:{last_text}, old_dur={last_dur:.6f}, rem={rem:.6f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="data_path/silent_speech_dataset/voiced",
    )
    ap.add_argument(
        "--glob",
        dest="glob_pat",
        default="*/data/textgrid/tg_*.TextGrid",
        help='glob pattern under --root (default: "*/data/textgrid/tg_*.TextGrid")',
    )
    ap.add_argument("--tier", default="phones", help='tier name (default: "phones")')
    ap.add_argument("--min_dur", type=float, default=0.3, help="min duration to consider (default: 0.3)")
    ap.add_argument("--fixed_dur", type=float, default=0.3, help="fixed duration for last phone (default: 0.3)")
    ap.add_argument(
        "--silence_label",
        default="",
        help='silence label (default: "" (empty string))',
    )
    ap.add_argument(
        "--also_silence",
        action="append",
        default=[],
        help='additional labels treated as silence (repeatable), e.g. --also_silence sp',
    )
    ap.add_argument("--write", action="store_true", help="actually overwrite files")
    ap.add_argument("--backup_ext", default="", help='optional backup ext, e.g. ".bak" (default: disabled)')
    ap.add_argument("--limit", type=int, default=0, help="process only first N files (0 = all)")
    ap.add_argument("--print_examples", type=int, default=20, help="print up to N changed file examples")
    args = ap.parse_args()

    root = args.root
    pat = os.path.join(root, args.glob_pat)
    paths = sorted(glob.glob(pat))
    if args.limit and args.limit > 0:
        paths = paths[: args.limit]

    silence_labels = [args.silence_label] + list(args.also_silence)

    total = 0
    parsed_ok = 0
    changed = 0
    skipped = 0
    errors: List[Tuple[str, str]] = []
    examples: List[Tuple[str, str]] = []

    for p in paths:
        total += 1
        try:
            tg = read_textgrid(p)
            parsed_ok += 1
            did_change, reason = fix_last_phone_interval(
                tg,
                tier_name=args.tier,
                fixed_dur=args.fixed_dur,
                min_dur=args.min_dur,
                silence_labels=silence_labels,
            )
            if did_change:
                changed += 1
                if len(examples) < args.print_examples:
                    examples.append((p, reason))
                if args.write:
                    if args.backup_ext:
                        shutil.copy2(p, p + args.backup_ext)
                    write_textgrid(tg, p)
            else:
                skipped += 1
        except Exception as e:
            errors.append((p, str(e)))

    mode = "WRITE" if args.write else "DRY-RUN"
    print(f"[{mode}] scanned={total}, parsed_ok={parsed_ok}, changed={changed}, skipped={skipped}, errors={len(errors)}")
    if examples:
        print("\nChanged examples:")
        for p, r in examples:
            print(f"- {p} :: {r}")
    if errors:
        print("\nErrors (first 20):")
        for p, msg in errors[:20]:
            print(f"- {p} :: {msg}")
        if len(errors) > 20:
            print(f"... {len(errors)-20} more errors")


if __name__ == "__main__":
    main()

