"""Compare extracted report table vs ids.js / BIBEK_IDS.

Uses path-rows from the markdown table (already validated against report layout),
avoids re-parsing raw USGS ID formats.

Usage:
    uv run python scripts/compare_icimod_vs_bibek.py
"""

import re
from pathlib import Path
from collections import defaultdict


def parse_markdown_table(filepath: str) -> list[dict]:
    text = Path(filepath).read_text()
    rows = []
    for line in text.splitlines():
        if line.startswith("| ") and not line.startswith("| Path-"):
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) >= 5:
                rows.append({
                    "path_row": parts[0],       # e.g. "150-34"
                    "image_raw": parts[1],       # e.g. "LE71500342005259PFS00"
                    "date": parts[2],            # e.g. "16/09/2005"
                    "purpose": parts[3],
                    "sub_basin": parts[4],
                })
    return rows


def markdown_date_to_ymd(date_str: str) -> str:
    """Convert DD/MM/YYYY from markdown to YYYYMMDD."""
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", date_str)
    if m:
        return f"{m.group(3)}{int(m.group(2)):02d}{int(m.group(1)):02d}"
    return date_str


def main():
    markdown_path = "analysis/icimod_landsat_images.md"
    exporter_path = "google_earth_scripts/export_hkh_rebuild.py"

    # 1. Parse report rows from markdown
    rows = parse_markdown_table(markdown_path)

    # Build set of report IDs: sensor_PPPRRR_YYYYMMDD
    report_ids = set()
    report_by_pr = defaultdict(list)  # path_row -> [normalized_id, ...]
    for r in rows:
        pr_parts = r["path_row"].split("-")
        pr = pr_parts[0].strip() + pr_parts[1].strip().zfill(3)  # "150-34" -> "150034"
        # Determine sensor from raw image ID
        il = r["image_raw"].lower()
        if il.startswith("le7") or il.startswith("l71") or il.startswith("l721"):
            sensor = "LE07"
        elif il.startswith("lt5") or il.startswith("l5"):
            sensor = "LT05"
        else:
            sensor = "UNK"
        ymd = markdown_date_to_ymd(r["date"])
        nid = f"{sensor}_{pr}_{ymd}"
        report_ids.add(nid)
        report_by_pr[pr].append(nid)

    # 2. Parse Bibek IDs
    with open(exporter_path) as f:
        exp_text = f.read()
    bibek_ids = set(re.findall(r'"LE07_\d{6}_\d{8}"', exp_text))
    bibek_ids = {b.strip('"') for b in bibek_ids}

    # 3. Build Bibek by path/row
    bibek_by_pr = defaultdict(list)
    for bid in bibek_ids:
        pr = bid.split("_")[1]
        bibek_by_pr[pr].append(bid)

    # 4. Normalize Bibek path/rows: report uses 6-digit like "15034", Bibek too
    report_prs = set(report_by_pr.keys())
    bibek_prs = set(bibek_by_pr.keys())

    both = report_ids & bibek_ids
    only_report = report_ids - bibek_ids
    only_bibek = bibek_ids - report_ids

    print(f"{'='*70}")
    print(f"ICIMOD REPORT vs BIBEK IDS COMPARISON")
    print(f"{'='*70}")
    print(f"Report LE07+LT05 rows:  {len(report_ids)}  ({len(rows)} rows from markdown)")
    print(f"Bibek IDs:              {len(bibek_ids)}")
    print(f"In both (exact match):  {len(both)}")
    print(f"Only in report:         {len(only_report)}")
    print(f"Only in Bibek:          {len(only_bibek)}")

    print(f"\n--- Path/Row Coverage ---")
    print(f"Report path/rows:  {len(report_prs)}")
    print(f"Bibek path/rows:   {len(bibek_prs)}")
    print(f"Both:              {len(report_prs & bibek_prs)}")
    only_in_report_pr = report_prs - bibek_prs
    only_in_bibek_pr = bibek_prs - report_prs
    if only_in_report_pr:
        print(f"Path/rows only in report: {sorted(only_in_report_pr)}")
    if only_in_bibek_pr:
        print(f"Path/rows only in Bibek:  {sorted(only_in_bibek_pr)}")

    # 5. For each shared path/row, compare Bibek's choice vs report
    print(f"\n--- Per Path/Row Comparison (shared) ---")
    shared_prs = sorted(report_prs & bibek_prs)
    for pr in shared_prs:
        bb = sorted(bibek_by_pr[pr])
        rr = sorted(report_by_pr[pr])
        report_dates = sorted(set(r.split("_")[2] for r in rr))
        bibek_dates = sorted(set(b.split("_")[2] for b in bb))

        match_set = set(bb) & set(rr)
        le07_report = [r for r in rr if r.startswith("LE07")]
        lt05_report = [r for r in rr if r.startswith("LT05")]

        if match_set:
            status = "✓"
        elif not le07_report and lt05_report:
            status = "LT05→LE07"
        else:
            status = "✗ different date"

        extra = f"  Report: {len(le07_report)} LE07 + {len(lt05_report)} LT05"
        if len(le07_report) > len(bb):
            extra += f" (extra LE07: {[d for d in report_dates if d not in bibek_dates]})"

        print(f"  {pr[:3]}-{pr[3:]}: {status}")
        print(f"    Bibek: {', '.join(b.split('_')[2] for b in bb)}")
        print(f"    Report LE07: {', '.join(sorted(set(r.split('_')[2] for r in le07_report)))}")
        if lt05_report:
            print(f"    Report LT05: {', '.join(sorted(set(r.split('_')[2] for r in lt05_report)))}")

    # 6. Summary table of Bibek-only (the decisions needed)
    print(f"\n--- Bibek IDs NOT matching report (decisions needed) ---")
    for bid in sorted(only_bibek):
        pr = bid.split("_")[1]
        bibek_date = bid.split("_")[2]
        # What does report have for this path/row?
        if pr in report_by_pr:
            report_dates = sorted(set(r.split("_")[2] for r in report_by_pr[pr]))
            le07_report = [r for r in report_by_pr[pr] if r.startswith("LE07")]
            lt05_report = [r for r in report_by_pr[pr] if r.startswith("LT05")]
            print(f"  {bid}")
            if le07_report:
                print(f"    Report LE07 dates: {', '.join(sorted(set(r.split('_')[2] for r in le07_report)))}")
            if lt05_report:
                print(f"    Report LT05 dates: {', '.join(sorted(set(r.split('_')[2] for r in lt05_report)))}")
        else:
            print(f"  {bid}")
            print(f"    (path/row {pr} not found in any report table)")

    # 7. Basin attribution for each path/row from report
    print(f"\n--- Basin by Path/Row (from report tables) ---")
    basin_by_pr = {}
    for r in rows:
        pr = r["path_row"].replace("-", "")
        # Determine which table this came from
        # We can infer from the markdown section
        pass  # The markdown section info isn't in the row dict
    
    # Count report-only LE07 scenes (extra analysis dates)
    le07_only = [r for r in only_report if r.startswith("LE07")]
    lt05_only = [r for r in only_report if r.startswith("LT05")]
    print(f"Report-only LE07 scenes: {len(le07_only)} (alternate dates for Bibek path/rows)")
    print(f"Report-only LT05 scenes: {len(lt05_only)} (Landsat 5 — Bibek used L7 instead)")


if __name__ == "__main__":
    main()
