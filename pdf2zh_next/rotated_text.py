#!/usr/bin/env python3
"""Translate text printed sideways: tables on their side, vertical labels.

The engine only translates horizontal text. Text turned 90 degrees is kept as
it is, untranslated, and text turned the other way is lost. This module works
around that without touching the engine:

- `prepare` appends, for every page with sideways text worth translating, a
  copy of that page turned so the text reads horizontally. The engine then
  translates the copies like any other page.
- `merge` removes the untranslated sideways text from the real page, lays the
  translated text of its copy over it, turned back, and deletes the copies.

If anything fails the copies are deleted and the files stay as the engine
left them.
"""
import argparse
import json
import os
import re

import fitz

# MuPDF prints its warnings to stdout, where this module prints its JSON result.
fitz.TOOLS.mupdf_display_errors(False)

UP, DOWN = (0, -1), (0, 1)  # PyMuPDF line direction: text reading bottom-to-top / top-to-bottom
TURN = {UP: 90, DOWN: 270}  # page rotation that makes that text horizontal
BACK = {UP: 90, DOWN: -90}  # rotation that lays the turned copy back over the page
# Margin stamps and lines without words are not worth a copy of the page.
# Every copy is one more page for the engine: a document with very many sideways
# pages must still finish within the job's time limit.
MAX_COPIES = int(os.environ.get("ROTATED_MAX_COPIES", "25"))
NOISE = re.compile(r"^(arXiv:\S+.*|[\W\d_]+)$")
WORD = re.compile(r"[^\W\d_]{2,}")


def sideways_lines(page):
    """Lines of text turned 90 degrees, by direction: {UP: [(rect, text)], DOWN: [...]}."""
    found = {}
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            dx, dy = line["dir"]
            if abs(dx) > 0.02 or abs(abs(dy) - 1) > 0.02:
                continue
            text = "".join(span["text"] for span in line["spans"]).strip()
            if text:
                found.setdefault(UP if dy < 0 else DOWN, []).append((fitz.Rect(line["bbox"]), text))
    return found


def worth_translating(texts):
    texts = [t for t in texts if not NOISE.match(t)]
    words = [w for t in texts for w in WORD.findall(t)]
    letters = sum(len(w) for w in words)
    if letters < 12 or not any(len(w) >= 4 for w in words):
        return False
    # Rows of single short tokens (tick labels, the words of an example sentence
    # spread along an axis) would lose their positions if translated as a paragraph.
    single = sum(1 for t in texts if len(t.split()) <= 1)
    if len(texts) >= 12 and single >= 0.8 * len(texts) and letters / len(texts) < 7:
        return False
    return True


def prepare(src, dst):
    """Write dst = src plus turned copies of the pages with sideways text. Returns the plan."""
    doc = fitz.open(src)
    pages = doc.page_count
    wanted, skipped = [], []
    for number in range(pages):
        page = doc[number]
        found = sideways_lines(page)
        for direction in (UP, DOWN):
            lines = found.get(direction, [])
            if not worth_translating([t for _, t in lines]):
                continue
            if page.rotation or max(abs(a - b) for a, b in zip(page.cropbox, page.mediabox)) > 0.5:
                skipped.append({"page": number, "reason": "rotated or cropped page"})
                continue
            wanted.append((number, direction, len(lines)))
    for number, _, _ in wanted[MAX_COPIES:]:
        skipped.append({"page": number, "reason": "too many sideways pages"})
    wanted = wanted[:MAX_COPIES]
    copies = []
    for number, direction, lines in wanted:  # all the copies first, then turn them
        doc.fullcopy_page(number)
        copies.append({"page": number, "copy": doc.page_count - 1, "dir": list(direction), "lines": lines})
    for c in copies:
        copy = doc[c["copy"]]
        # The copy is deleted after the translation: it does not need the links of the page.
        doc.xref_set_key(copy.xref, "Annots", "null")
        copy = doc.reload_page(copy)
        copy.set_rotation(TURN[tuple(c["dir"])])
        copy.remove_rotation()
    if copies:
        doc.save(dst, garbage=3, deflate=True)
    return {"pages": pages, "copies": copies, "skipped": skipped}


def keep_translated_text(page):
    """Leave on a translated copy only its horizontal text: what the engine translated."""
    turned = [rect for lines in sideways_lines(page).values() for rect, _ in lines]
    for rect in turned:  # untranslated text the engine passed through
        page.add_redact_annot(rect, fill=False)
    if turned:
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE, graphics=fitz.PDF_REDACT_LINE_ART_NONE,
                              text=fitz.PDF_REDACT_TEXT_REMOVE)
    page.add_redact_annot(page.rect, fill=False)
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_REMOVE, graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED,
                          text=fitz.PDF_REDACT_TEXT_NONE)


def overlap(text, source_words):
    found = re.findall(r"[^\W\d_]{4,}", text.lower())
    return sum(w in source_words for w in found) / max(1, len(found))


def translated_on_left(target, prepared, copies):
    """In a side-by-side bilingual file, whether the translated half is the left one.

    Decided on the turned copies: all their text is translated, so the half that
    still matches the source is the original.
    """
    left = right = 0.0
    for c in copies:
        source = prepared[c["copy"]]
        words = set(re.findall(r"[^\W\d_]{4,}", source.get_text().lower()))
        page = target[c["copy"]]
        w, h = source.rect.width, source.rect.height
        left += overlap(page.get_text(clip=fitz.Rect(0, 0, w, h)), words)
        right += overlap(page.get_text(clip=fitz.Rect(w, 0, 2 * w, h)), words)
    return left <= right


def merge_file(plan, prepared, layers, path):
    target = fitz.open(path)
    pages, copies = plan["pages"], plan["copies"]
    total = pages + len(copies)
    if target.page_count != total:
        raise ValueError(f"{target.page_count} pages, expected {total}")
    ratio = target[0].rect.width / prepared[0].rect.width
    if abs(ratio - 1) < 0.01:
        shift = False
    elif abs(ratio - 2) < 0.01:
        shift = not translated_on_left(target, prepared, copies)
    else:
        raise ValueError("unknown page layout")
    for c in copies:
        page = target[c["page"]]
        direction = tuple(c["dir"])
        area = fitz.Rect(prepared[c["page"]].rect)
        if shift:
            area += (area.width, 0, area.width, 0)
        # the untranslated sideways text the engine left on the real page
        removed = 0
        for rect, _ in sideways_lines(page).get(direction, []):
            if ((rect.tl + rect.br) / 2) in area:
                page.add_redact_annot(rect, fill=False)
                removed += 1
        if removed:
            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE, graphics=fitz.PDF_REDACT_LINE_ART_NONE,
                                  text=fitz.PDF_REDACT_TEXT_REMOVE)
        page.show_pdf_page(area, layers, c["copy"], rotate=BACK[direction])
    target.delete_pages(from_page=pages, to_page=total - 1)
    save(target, path)


def strip(plan, path):
    """Delete the turned copies and nothing else."""
    target = fitz.open(path)
    pages, total = plan["pages"], plan["pages"] + len(plan["copies"])
    if target.page_count != total:
        raise ValueError(f"{target.page_count} pages, expected {total}")
    target.delete_pages(from_page=pages, to_page=total - 1)
    save(target, path)


def save(doc, path):
    tmp = path + ".tmp"
    doc.save(tmp, garbage=3, deflate=True)
    doc.close()
    os.replace(tmp, path)


def merge(plan, prepared_path, mono, dual=None):
    """Put the translated sideways text into the translated files and delete the copies."""
    report = {"copies": len(plan["copies"])}
    prepared = layers = None
    try:
        prepared = fitz.open(prepared_path)
        with open(mono, "rb") as fh:
            layers = fitz.open(stream=fh.read(), filetype="pdf")
        for c in plan["copies"]:
            keep_translated_text(layers[c["copy"]])
    except Exception as exc:
        layers = None
        report["error"] = f"{type(exc).__name__}: {exc}"[:300]
    for name, path in (("mono", mono), ("dual", dual)):
        if not path:
            continue
        try:
            if layers is None:
                raise RuntimeError("no translated layers")
            merge_file(plan, prepared, layers, path)
            report[name] = "merged"
        except Exception as exc:
            report[name] = f"failed ({type(exc).__name__}: {exc})"[:300]
            try:  # at least hand over the files without the extra pages
                strip(plan, path)
            except Exception as exc2:
                report[name] += f"; copies not deleted ({type(exc2).__name__}: {exc2})"[:200]
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="append turned copies of the pages with sideways text")
    p.add_argument("original")
    p.add_argument("prepared", help="written only when there is something to translate")
    p.add_argument("--plan", required=True)
    m = sub.add_parser("merge", help="put the translated sideways text back and delete the copies")
    m.add_argument("--plan", required=True)
    m.add_argument("--prepared", required=True)
    m.add_argument("--mono", required=True)
    m.add_argument("--dual")
    args = ap.parse_args()
    if args.command == "prepare":
        plan = prepare(args.original, args.prepared)
        with open(args.plan, "w") as fh:
            json.dump(plan, fh)
        print(json.dumps(plan))
    else:
        with open(args.plan) as fh:
            plan = json.load(fh)
        print(json.dumps(merge(plan, args.prepared, args.mono, args.dual)))


if __name__ == "__main__":
    main()
