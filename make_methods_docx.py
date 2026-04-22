"""Convert prior_analysis/motor_vs_cognitive_methods.txt into a .docx.

Preserves headings (numbered sections), bullet-like sub-items, and
monospaced blocks (indented code/paths). Output:
    prior_analysis/motor_vs_cognitive_methods.docx
"""

from __future__ import annotations

import _paths  # noqa: F401  (sys.path bootstrap, safe under both invocations)

import re
from pathlib import Path

from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.style import WD_STYLE_TYPE

SRC = Path("prior_analysis/motor_vs_cognitive_methods.txt")
DST = Path("prior_analysis/motor_vs_cognitive_methods.docx")


# Section headings in the source are of the form:
#   "1. DATA" / "2. PER-EDIT SIGNALS" ... followed by a dashed underline.
SECTION_RE = re.compile(r"^\s*(\d+)\.\s+([A-Z][A-Z0-9 \-]+)$")
SUBITEM_RE = re.compile(r"^\s{4}([a-z])\)\s+(.*)$")      # "    a) ..."
ROMAN_RE = re.compile(r"^\s{4}(i{1,3}|iv|v)\)\s+(.*)$")   # "    i) ..."


def main() -> None:
    text = SRC.read_text()
    lines = text.splitlines()

    doc = Document()

    # Title
    title = doc.add_heading("Motor slip vs cognitive error", level=0)
    for run in title.runs:
        run.font.size = Pt(22)

    subtitle = doc.add_paragraph(
        "Classification of edit-level errors in the CogARC Experiment 2 dataset"
    )
    subtitle.runs[0].italic = True
    subtitle.runs[0].font.size = Pt(13)

    # Tighter defaults
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    # Simple state machine over the source lines
    i = 0
    skip_until_non_dash = False
    in_code_block = False

    def flush_code(buffer):
        if not buffer:
            return
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.left_indent = Pt(18)
        for k, line in enumerate(buffer):
            if k > 0:
                p.add_run().add_break()
            run = p.add_run(line)
            run.font.name = "Consolas"
            run.font.size = Pt(9.5)

    code_buffer: list[str] = []

    # Skip the leading banner (first two non-empty title lines + dashed rule)
    # which we've already rendered via the Title/Subtitle above.
    skip_leading = True

    while i < len(lines):
        ln = lines[i]

        if skip_leading:
            if ln.startswith("====="):
                skip_leading = False
            i += 1
            continue

        # Section heading (e.g. "1. DATA") followed by a dashed underline
        m = SECTION_RE.match(ln)
        if m and i + 1 < len(lines) and set(lines[i + 1].strip()) == {"-"}:
            flush_code(code_buffer); code_buffer = []
            in_code_block = False
            num, name = m.group(1), m.group(2).title()
            doc.add_heading(f"{num}. {name}", level=1)
            i += 2  # consume heading + dashed rule
            continue

        # Blank line: end any current code block (we treat a blank line as a
        # paragraph break, so we accumulate code only across adjacent indented
        # lines).
        if not ln.strip():
            flush_code(code_buffer); code_buffer = []
            in_code_block = False
            i += 1
            continue

        # Indented lines (4+ leading spaces) are typically code/path/constants.
        # Exception: sub-items like "    a) ..." or "    i) ..." which should
        # render as regular paragraphs with a small indent.
        sub = SUBITEM_RE.match(ln) or ROMAN_RE.match(ln)
        if sub:
            flush_code(code_buffer); code_buffer = []
            in_code_block = False
            letter, body = sub.group(1), sub.group(2)
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Pt(18)
            r = p.add_run(f"{letter})  ")
            r.bold = True
            p.add_run(body)
            # Consume subsequent continuation lines (indented more)
            j = i + 1
            while j < len(lines) and lines[j].startswith("       ") and lines[j].strip():
                p.add_run(" " + lines[j].strip())
                j += 1
            i = j
            continue

        # Hyphen bullet list at any indent ("    - foo"). CHECK BEFORE
        # the generic indented-code branch so bulleted lines don't get
        # misclassified as code.
        if ln.lstrip().startswith("- "):
            flush_code(code_buffer); code_buffer = []
            p = doc.add_paragraph(ln.lstrip()[2:], style="List Bullet")
            p.paragraph_format.left_indent = Pt(18)
            # Absorb continuation lines that are more deeply indented
            j = i + 1
            while j < len(lines) and lines[j].startswith("       ") and lines[j].strip() \
                    and not lines[j].lstrip().startswith("- "):
                p.add_run(" " + lines[j].strip())
                j += 1
            i = j
            continue

        if ln.startswith("    "):
            code_buffer.append(ln[4:] if ln.startswith("    ") else ln)
            in_code_block = True
            i += 1
            continue

        # Normal paragraph — coalesce wrapped lines until blank or a new
        # structural element.
        flush_code(code_buffer); code_buffer = []
        buf = [ln.rstrip()]
        j = i + 1
        while j < len(lines):
            nxt = lines[j]
            if not nxt.strip():
                break
            if SECTION_RE.match(nxt) and j + 1 < len(lines) and set(lines[j + 1].strip()) == {"-"}:
                break
            if SUBITEM_RE.match(nxt) or ROMAN_RE.match(nxt):
                break
            if nxt.startswith("    "):
                break
            if nxt.lstrip().startswith("- "):
                break
            buf.append(nxt.rstrip())
            j += 1
        doc.add_paragraph(" ".join(s.strip() for s in buf))
        i = j

    flush_code(code_buffer)

    DST.parent.mkdir(parents=True, exist_ok=True)
    doc.save(DST)
    print(f"wrote {DST}")


if __name__ == "__main__":
    main()
