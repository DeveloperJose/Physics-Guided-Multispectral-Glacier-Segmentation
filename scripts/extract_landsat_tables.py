"""Extract Landsat image tables from ICIMOD report text to markdown.

Usage:
    uv run python scripts/extract_landsat_tables.py
    uv run python scripts/extract_landsat_tables.py --report analysis/references/icimod_report.txt --output analysis/icimod_landsat_images.md
"""

import re
import argparse
from pathlib import Path


TABLE_MARKERS = {
    "Table 5.1: Landsat images used for the Amu Darya basin": 0,
    "Table 6.1: Landsat images used for the Indus basin": 1,
    "Table 7.1: Landsat images used for the Ganges basin": 2,
    "Table 8.1: Landsat images used for the Brahmaputra basin": 3,
    "Table 9.1: Landsat images used for the Irrawady basin": 4,
}

TABLE_NAMES = [
    "Amu Darya",
    "Indus",
    "Ganges",
    "Brahmaputra",
    "Irrawady",
]


def find_table_extent(lines: list[str], start_marker: str) -> tuple[int, int]:
    """Find start and end line indices for a table region."""
    si = None
    for i, l in enumerate(lines):
        if start_marker in l:
            si = i
            break
    if si is None:
        return (0, 0)

    # Track lines since last image+date to detect table end
    lines_since_last_img = 0

    for j in range(si + 1, len(lines)):
        l = lines[j]
        # Stop at next non-continued table marker
        if re.match(r"\s*Table \d+\.\d+:", l):
            if "continued" in l.lower() and start_marker.split(":")[0].split()[-1] in l:
                continue
            else:
                return (si, j)
        # Check for image line
        img_at = re.search(r"((?:[Ll][Ee]7|[Ll]71\d|[Ll][Tt]5|[Ll]5)\d[0-9A-Za-z_]*)", l)
        dt_at = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", l)
        if img_at and dt_at:
            lines_since_last_img = 0
        else:
            lines_since_last_img += 1
        # 30+ lines without any image means table is over
        if lines_since_last_img > 30:
            return (si, j)
        # Also stop early at obvious section-start patterns
        stripped = l.strip()
        stripped_lower = stripped.lower()
        if lines_since_last_img > 5 and stripped:
            # Section header: "word basin"
            if re.match(r"^\w+\s+(sub-)?basin", stripped_lower):
                return (si, j)
            # Page-basin headers: "7 Ganges Basin"
            if re.match(r"^\d+\s+", stripped) and "basin" in stripped_lower:
                return (si, j)
            # Narrative section starters
            if re.match(r"^(mapping|inventory|glacier area|number, area|glacier classes)", stripped_lower):
                return (si, j)
    return (si, len(lines))


def pathrow_from_image(img: str) -> str | None:
    """Extract WRS path-row from the image ID itself.

    LE7 format:  LE7 + PPPRRR + YYYYDOY...  -> PPP-RRR
    L71 format:  L71 + PPPRRR + _RRR...
    L5 format:   L5  + PPPRRR + ...
    LT5 format:  LT5 + PPPRRR + ...
    """
    il = img.lower()
    if il.startswith("le7"):
        # LE7PPPRRR...
        pr = img[3:9]  # 6 chars: path+row
        if len(pr) == 6 and pr.isdigit():
            return f"{int(pr[:3]):d}-{int(pr[3:]):02d}"
    elif il.startswith("l71"):
        # L71PPPRRR_...
        pr = img[3:9]
        if len(pr) == 6 and pr.isdigit():
            return f"{int(pr[:3]):d}-{int(pr[3:]):02d}"
    elif il.startswith("l5") or il.startswith("lt5"):
        skip = 3 if il.startswith("lt5") else 2
        pr = img[skip:skip+6]
        if len(pr) == 6 and pr.isdigit():
            return f"{int(pr[:3]):d}-{int(pr[3:]):02d}"
    elif il.startswith("l721"):
        # L721PPPRRR_... (Landsat 7 with alternate prefix)
        pr = img[4:10]
        if len(pr) == 6 and pr.isdigit():
            return f"{int(pr[:3]):d}-{int(pr[3:]):02d}"
    return None


def extract_table_rows(lines: list[str], start_line: int, end_line: int) -> list[dict]:
    """Extract image rows from a table region.

    Handles:
    - Multi-line entries (path-row on first line, continuation lines without)
    - Sub-basin text wrapping to next line
    - Both LE7 and L5 image ID formats
    """
    rows = []
    current_pr = None
    # Track sub-basin accumulation for multi-line sub-basin
    pending_sub_basin = ""

    # Regex for path-row pattern
    pr_pattern = re.compile(r"(\d{3})\s*-?\s*(\d{2})")
    # Image ID pattern
    img_pattern = re.compile(r"((?:[Ll][Ee]7|[Ll]711|[Ll]721|[Ll][Tt]5|[Ll]5)\d[0-9A-Za-z_]*)")

    for line in lines[start_line:end_line]:
        stripped = line.strip()
        if not stripped:
            continue

        # Check for image ID FIRST
        img_match = img_pattern.search(stripped)
        date_match = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", stripped)

        # Look for path-row ONLY before the image ID on this line
        pr_from_img = pathrow_from_image(img_match.group(1)) if img_match else None
        if pr_from_img:
            # Image ID itself encodes the path/row — this is the most reliable source
            current_pr = pr_from_img

        # Also look for path-row before the image ID (for sub-basin annotation)
        has_pr = None
        if img_match and not pr_from_img:
            before_img = stripped[: img_match.start()]
            has_pr = pr_pattern.search(before_img)
        elif not img_match:
            has_pr = pr_pattern.search(stripped)

        # Check for "analysis" or "correction" keyword
        purpose_match = re.search(r"(analysis|correction)", stripped.lower())

        if has_pr and img_match and date_match:
            # New image with path-row on this line
            current_pr = f"{has_pr.group(1)}-{has_pr.group(2).zfill(2)}"
            img = img_match.group(1)
            date_str = date_match.group(0)
            purpose = purpose_match.group(1) if purpose_match else ""

            # Sub-basin: text after the purpose keyword
            sub_basin = ""
            if purpose_match:
                after_purpose = stripped[purpose_match.end():].strip()
                # Clean up leading/trailing punctuation
                after_purpose = after_purpose.strip(",; ")
                sub_basin = after_purpose
            # Strip any path-row pattern that leaked into sub-basin
            sub_basin = re.sub(r"\d{3}\s*-?\s*\d{2,3}", "", sub_basin).strip(",; ")

            # Normalize sensor prefix
            il = img.lower()
            if il.startswith("le7") or il.startswith("l711") or il.startswith("l721"):
                sensor = "LE07"
            elif il.startswith("lt5") or il.startswith("l5"):
                sensor = "LT05"
            else:
                sensor = "UNK"

            rows.append({
                "path_row": current_pr,
                "image": img,
                "sensor": sensor,
                "date": date_str,
                "purpose": purpose,
                "sub_basin": sub_basin,
                "_raw": stripped,
            })
            pending_sub_basin = ""

        elif img_match and date_match and current_pr and not has_pr:
            # Continuation line: same path-row, no path-row on this line
            img = img_match.group(1)
            date_str = date_match.group(0)
            purpose = purpose_match.group(1) if purpose_match else ""

            # Sub-basin: text after the purpose keyword
            sub_basin = ""
            if purpose_match:
                after_purpose = stripped[purpose_match.end():].strip()
                after_purpose = after_purpose.strip(",; ")
                sub_basin = after_purpose
            # Strip any path-row pattern that leaked into sub-basin
            sub_basin = re.sub(r"\d{3}\s*-?\s*\d{2,3}", "", sub_basin).strip(",; ")

            # If no purpose keyword found, this might be a continuation of
            # the previous line's sub-basin or a sub-basin-only line
            if not purpose_match:
                if current_pr:
                    if rows:
                        extra = stripped.strip(",; ")
                        # Strip any path-row pattern from continuation text
                        extra = re.sub(r"\d{3}\s*-?\s*\d{2,3}\s*", "", extra).strip(",; ")
                        if extra:
                            if rows[-1]["sub_basin"]:
                                rows[-1]["sub_basin"] += ", " + extra
                            else:
                                rows[-1]["sub_basin"] = extra
                continue

            il = img.lower()
            if il.startswith("le7") or il.startswith("l711") or il.startswith("l721"):
                sensor = "LE07"
            elif il.startswith("lt5") or il.startswith("l5"):
                sensor = "LT05"
            else:
                sensor = "UNK"

            rows.append({
                "path_row": current_pr,
                "image": img,
                "sensor": sensor,
                "date": date_str,
                "purpose": purpose,
                "sub_basin": sub_basin,
                "_raw": stripped,
            })
            pending_sub_basin = ""

        elif not img_match:
            # No image on this line — might be sub-basin continuation or new path-row header
            # Update current_pr if a path-row pattern is found
            if has_pr:
                current_pr = f"{has_pr.group(1)}-{has_pr.group(2).zfill(2)}"

            if not current_pr:
                continue

            # Stop collecting sub-basin on section headers or narrative starts
            stripped_lower = stripped.lower()
            # Single word followed by "basin" = section header
            if re.match(r"^\w+\s+basin", stripped_lower):
                continue
            # Narrative starts with determiners or conjunctions
            if re.match(r"^(the|this|these|those|altogether|however|therefore|thus|with|and|but|for|nor|yet|so|because|although|while|since|after|before|during|within|without|across|among|between|through|throughout|from|into|onto|upon|about|above|below|beneath|underneath)", stripped_lower):
                continue
            # Past-tense verbs common in report narrative
            if re.match(r"^(was|were|had|has|have|been|being|found|shows|shown|based|located|originates|contains|includes|comprises|summarised|summarized|presented|indicates|described|mapped|measured|estimated|calculated|derived|used|obtained|classified|categorised|categorized)", stripped_lower):
                continue
            # Table/figure references
            if re.match(r"^(table|figure)", stripped_lower):
                continue
            # Page numbers
            if re.match(r"^\d+$", stripped.strip()):
                continue
            # Long narrative lines (probably not sub-basin)
            if len(stripped) > 120:
                continue

            # Check if it looks like sub-basin text (comma-separated basin names)
            if stripped and not stripped.startswith("Table") and not stripped.startswith("Figure"):
                # Check for page numbers or section headers
                if not re.match(r"^\d+$", stripped.strip()):
                    # Append to last row's sub-basin if it looks like continuation
                    if rows and not any(
                        x in stripped.lower()
                        for x in ["table", "figure", "the status", "hindu kush"]
                    ):
                        extra = stripped.strip(",; ")
                        # Strip path-row pattern from continuation text
                        extra = re.sub(r"\d{3}\s*-?\s*\d{2,3}\s*", "", extra).strip(",; ")
                        if extra and rows[-1]["sub_basin"]:
                            rows[-1]["sub_basin"] += " " + extra
                        elif extra and not rows[-1]["sub_basin"]:
                            rows[-1]["sub_basin"] = extra

    return rows


def format_date(date_str: str) -> str:
    """Normalize date format to DD/MM/YYYY."""
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", date_str)
    if m:
        d, mo, y = m.groups()
        return f"{int(d):02d}/{int(mo):02d}/{y}"
    return date_str


def row_to_markdown(row: dict) -> str:
    """Format a row to markdown line: path-row image date purpose sub-basin"""
    parts = [
        row["path_row"],
        row["image"],
        format_date(row["date"]),
        row["purpose"],
        row["sub_basin"],
    ]
    return " ".join(parts)


def extract_all(report_path: str | Path) -> str:
    """Extract all Landsat image tables and return as markdown."""
    lines = Path(report_path).read_text(errors="ignore").splitlines()
    output = []
    output.append("# ICIMOD Landsat Images Used for Glacier Mapping (from report tables)")
    output.append("")
    output.append(
        "Auto-extracted from the ICIMOD report. "
        "Path-rows, image IDs, and dates are as listed in the tables. "
        "Multi-line entries are collapsed into single rows."
    )
    output.append("")
    output.append("---")
    output.append("")

    for marker, idx in TABLE_MARKERS.items():
        # Extract the table name for the heading
        name_match = re.search(r"for the (.+)", marker)
        basin_name = name_match.group(1) if name_match else f"Table {idx + 5}.1"
        tnum = idx + 5

        si, ei = find_table_extent(lines, marker)
        if si == 0 and ei == 0:
            output.append(f"## Table {tnum}.1: Landsat images for {basin_name}")
            output.append("")
            output.append("*Table not found in report text.*")
            output.append("")
            output.append("---")
            output.append("")
            continue

        output.append(f"## Table {tnum}.1: Landsat images for {basin_name}")
        output.append("")
        output.append("| Path-Row | Image | Date | Used for | Sub-basin |")
        output.append("|---|---|---|---|---|")

        rows = extract_table_rows(lines, si, ei)

        # Also check for "Table X.1 continued" sections
        for ci, l in enumerate(lines):
            if f"Table {tnum}.1 continued" in l and ci > si:
                # Find end
                c_end = find_table_extent(lines, f"Table {tnum}.1 continued")
                # Use a broader end - until next table
                c_ei = ei
                for cj in range(ci + 1, len(lines)):
                    if re.match(r"\s*Table \d+\.\d+:", lines[cj]) and "continued" not in lines[cj]:
                        c_ei = cj
                        break
                more_rows = extract_table_rows(lines, ci, c_ei)
                # Don't duplicate rows that cross the page break
                existing_images = {r["image"] for r in rows}
                for r in more_rows:
                    if r["image"] not in existing_images:
                        rows.append(r)
                        existing_images.add(r["image"])

        for row in rows:
            out = row_to_markdown(row)
            # Markdown table row
            pr = row["path_row"]
            img = row["image"]
            dt = format_date(row["date"])
            purpose = row["purpose"]
            sb = row["sub_basin"]
            output.append(f"| {pr} | {img} | {dt} | {purpose} | {sb} |")

        output.append("")
        output.append("---")
        output.append("")

    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description="Extract ICIMOD report Landsat image tables to markdown."
    )
    parser.add_argument(
        "--report",
        default="analysis/references/icimod_report.txt",
        help="Path to ICIMOD report text extraction.",
    )
    parser.add_argument(
        "--output",
        default="analysis/icimod_landsat_images.md",
        help="Output markdown file path.",
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"Report not found: {report_path}")
        return 1

    markdown = extract_all(report_path)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown)
    print(f"Written to: {out_path}")
    print(f"  Lines: {len(markdown.splitlines())}")

    # Summary stats
    lines = report_path.read_text(errors="ignore").splitlines()
    total_rows = 0
    for marker in TABLE_MARKERS:
        si, ei = find_table_extent(lines, marker)
        if si:
            rows = extract_table_rows(lines, si, ei)
            total_rows += len(rows)
    print(f"  Total rows extracted: {total_rows}")


if __name__ == "__main__":
    main()
