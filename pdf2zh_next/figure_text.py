#!/usr/bin/env python3
"""Translate the text inside raster figures of a PDF.

OCR finds the labels of each embedded image, an LLM translates them, and the
image is redrawn: each label is covered with its own background colour and the
translation is written in the same place, fitted to the free space around it.
Lines, arrows and everything that is not text stay untouched.
"""
import argparse
import json
import os
import re
import subprocess
import tempfile

import cv2
import fitz
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

TESSERACT_LANGS = {"en": "eng", "es": "spa", "pt": "por", "fr": "fra", "de": "deu", "it": "ita"}
LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "pt": "Portuguese", "fr": "French", "de": "German",
    "it": "Italian", "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "id": "Indonesian",
    "ar": "Arabic", "hi": "Hindi", "bn": "Bengali",
}
FONT_CANDIDATES = [
    os.path.expanduser("~/.cache/babeldoc/fonts/NotoSans-Regular.ttf"),
    "/root/.cache/babeldoc/fonts/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
MIN_IMAGE_SIDE = 200  # icons and logos are not worth it


# ---------------------------------------------------------------- image utils

def image_rgb(doc, xref, smask=0):
    """RGB pixels of an image; transparent areas are composited on white (the page)."""
    pix = fitz.Pixmap(doc, xref)
    if pix.colorspace and pix.colorspace.n not in (1, 3):
        pix = fitz.Pixmap(fitz.csRGB, pix)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n).astype(np.float32)
    alpha = None
    if pix.alpha:
        alpha = arr[:, :, -1:] / 255.0
        arr = arr[:, :, :-1]
    elif smask:
        mask = fitz.Pixmap(doc, smask)
        if (mask.width, mask.height) == (pix.width, pix.height):
            alpha = np.frombuffer(mask.samples, dtype=np.uint8).reshape(
                mask.height, mask.width, mask.n)[:, :, :1].astype(np.float32) / 255.0
    if alpha is not None:
        arr = arr * alpha + 255.0 * (1.0 - alpha)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return np.clip(arr[:, :, :3], 0, 255).astype(np.uint8).copy()


def split_text_and_structure(rgb):
    """Dark pixels split into glyph-sized components (text) and the rest (lines, borders)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dark = (gray < 150).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    h_img, w_img = gray.shape
    ws, hs, areas = stats[1:, 2], stats[1:, 3], stats[1:, 4]
    glyphish = (areas >= 15) & (hs >= 6) & (hs <= 0.06 * h_img) & (ws <= 0.06 * w_img)
    glyph_h = float(np.median(hs[glyphish])) if glyphish.any() else 20.0
    keep = (
        (hs <= 2.2 * glyph_h) & (ws <= 3.0 * glyph_h) & (areas >= 6)
        & ~(ws > 6 * hs) & ~((hs > 6 * ws) & (hs > 1.5 * glyph_h))
    )
    keep_ids = np.nonzero(keep)[0] + 1
    text = np.isin(labels, keep_ids)
    structure = dark.astype(bool) & ~text
    return text, structure, glyph_h


# ------------------------------------------------------------------------ OCR

def detect_lines(text_mask, glyph_h):
    """Text lines found by geometry: glyphs joined horizontally, never vertically."""
    k = max(3, int(glyph_h * 0.9))
    joined = cv2.dilate(text_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    _, comp = cv2.connectedComponents(joined, connectivity=8)
    ys, xs = np.nonzero(text_mask)
    if not len(ys):
        return []
    ids = comp[ys, xs]
    order = np.argsort(ids, kind="stable")
    ids, ys, xs = ids[order], ys[order], xs[order]
    _, starts = np.unique(ids, return_index=True)
    lines = []
    for x0, x1, y0, y1 in zip(np.minimum.reduceat(xs, starts), np.maximum.reduceat(xs, starts),
                              np.minimum.reduceat(ys, starts), np.maximum.reduceat(ys, starts)):
        if y1 - y0 + 1 < 0.5 * glyph_h or x1 - x0 + 1 < 0.5 * glyph_h:
            continue
        lines.append({"x0": int(x0), "y0": int(y0), "x1": int(x1) + 1, "y1": int(y1) + 1})
    return lines


def ocr_line(clean, line, lang, tmp):
    """Best reading of one line: normal size first, enlarged when unsure."""
    best = ("", 0.0)
    for scale, psm in ((1.0, 7), (2.0, 7), (2.0, 8)):
        text, conf = ocr_crop(clean, line, lang, tmp, scale, psm)
        if conf > best[1]:
            best = (text, conf)
        if best[1] >= 60:
            break
    return best


def wordlike(text):
    """Real words, not the noise tesseract reads in arrows or symbols."""
    letters = sum(ch.isalpha() for ch in text)
    return letters >= 2 and letters >= 0.6 * len(text.replace(" ", ""))


def ocr_crop(clean, line, lang, tmp, scale, psm):
    # Only the line itself, on a white margin: neighbouring pixels confuse tesseract.
    h = line["y1"] - line["y0"]
    crop = clean[line["y0"]:line["y1"], line["x0"]:line["x1"]]
    factor = max(scale, 40.0 / h if h < 30 else 1.0)
    if factor != 1.0:
        crop = cv2.resize(crop, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    pad = max(8, int(0.6 * crop.shape[0]))
    crop = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    path = os.path.join(tmp, "line.png")
    cv2.imwrite(path, crop)
    out = subprocess.run(
        ["tesseract", path, "stdout", "-l", lang, "--psm", str(psm), "tsv"],
        capture_output=True, text=True, check=True,
    ).stdout
    words, confs = [], []
    for row in out.splitlines()[1:]:
        cols = row.split("\t")
        if len(cols) >= 12 and cols[11].strip():
            words.append(cols[11].strip())
            confs.append(float(cols[10]))
    text = re.sub(r"(\w)- (\w)", r"\1-\2", " ".join(words))
    return text, (sum(confs) / len(confs) if confs else 0.0)


def read_lines(text_mask, glyph_h, lang):
    clean = np.where(text_mask, 0, 255).astype(np.uint8)
    lines = []
    with tempfile.TemporaryDirectory() as tmp:
        for line in detect_lines(text_mask, glyph_h):
            text, conf = ocr_line(clean, line, lang, tmp)
            if (conf >= 40 and re.search(r"[A-Za-z0-9]", text)) or (conf >= 10 and wordlike(text)):
                lines.append({**line, "text": text, "conf": conf})
    return lines


def separated(a, b, structure):
    """True when a line or border runs between two stacked lines."""
    x0, x1 = max(a["x0"], b["x0"]), min(a["x1"], b["x1"])
    if x1 <= x0:
        return True
    strip = structure[a["y1"]:b["y0"], x0:x1]
    if strip.size == 0:
        return False
    return bool((strip.mean(axis=1) > 0.5).any())


def group_labels(lines, structure):
    lines = sorted(lines, key=lambda l: l["y0"])
    labels = []
    for line in lines:
        h = line["y1"] - line["y0"]
        target = None
        for label in labels:
            last = label["lines"][-1]
            lh = last["y1"] - last["y0"]
            gap = line["y0"] - last["y1"]
            overlap = min(line["x1"], last["x1"]) - max(line["x0"], last["x0"])
            if (-0.3 * h < gap < 0.9 * max(h, lh) and overlap > 0
                    and max(h, lh) / max(1, min(h, lh)) < 1.5
                    and not separated(last, line, structure)):
                target = label
                break
        if target:
            target["lines"].append(line)
        else:
            labels.append({"lines": [line]})
    for label in labels:
        ls = label["lines"]
        text = ""
        for l in ls:
            if text.endswith("-"):
                text += l["text"]
            else:
                text += (" " if text else "") + l["text"]
        label["text"] = text
        label["x0"] = min(l["x0"] for l in ls)
        label["y0"] = min(l["y0"] for l in ls)
        label["x1"] = max(l["x1"] for l in ls)
        label["y1"] = max(l["y1"] for l in ls)
    return labels


def worth_translating(text):
    words = re.findall(r"[A-Za-z]{2,}", text)
    if not words:
        return False
    if re.fullmatch(r"[A-Za-z]?[x×X]", text.strip()):
        return False
    return True


# ---------------------------------------------------------------- translation

def translate_labels(texts, li, lo, context, reference=None, max_chars=None, source=None):
    key = os.environ["GROQ_API_KEY"]
    model = os.environ.get("TRANSLATION_GROQ_MODEL", "openai/gpt-oss-120b")
    src = "the source language" if li in ("", "auto") else LANGUAGE_NAMES.get(li, li)
    dst = LANGUAGE_NAMES.get(lo, lo)
    prompt = (
        f"You translate the text labels of one figure from an academic paper, from {src} to {dst}.\n"
        f"Figure caption: {context or 'unknown'}\n"
        + (f"Original text of the same page:\n{source[:2500]}\n" if source and reference else "")
        + (f"Translation of that page (spaces between words were lost in extraction):\n{reference[:2500]}\n" if reference else "")
        + "\n"
        "Rules:\n"
        + ("- Consistency first: when the translated page renders a concept of a label (e.g. 'embedding', "
           "'feed-forward'), use exactly that rendering in the label.\n" if reference else "")
        + f"- Otherwise use the standard {dst} terminology of the field; keep acronyms, symbols, variable names, "
        "function names (e.g. Softmax, ReLU) and proper names unchanged.\n"
        "- Translate descriptive and generic words (e.g. 'Linear', 'Input', 'Output'); keep a term in the source "
        "language only when the field normally leaves it untranslated.\n"
        f"- Use the capitalization conventions of {dst} (no title case unless {dst} uses it).\n"
        "- Keep normal spacing and punctuation; translate connectors such as '&' or 'and'.\n"
        "- max_chars is the room each label has at its original size. Stay within it when a natural wording "
        "allows: drop articles, use the common short form, or abbreviate the same way the original does "
        "(e.g. 'Norm' -> 'norm.'). Never cut words arbitrarily; longer text will be shrunk to fit.\n"
        "- If a label should not be translated, return it unchanged.\n"
        "- Return only JSON: {\"translations\": [...]} with exactly one string per input label, same order.\n\n"
        f"Labels: {json.dumps([{'text': t, 'max_chars': m} for t, m in zip(texts, max_chars or [None] * len(texts))], ensure_ascii=False)}"
    )
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_completion_tokens": 4096,
        "response_format": {"type": "json_object"},
    }
    effort = os.environ.get("FIGURE_REASONING_EFFORT", "medium")
    if effort:
        body["reasoning_effort"] = effort
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"}, json=body, timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    out = json.loads(content)["translations"]
    if len(out) != len(texts):
        raise ValueError(f"expected {len(texts)} translations, got {len(out)}")
    clean = [re.sub("[\u2010\u2011\u2012\u2013]", "-", str(t)).strip() or s for t, s in zip(out, texts)]
    return clean, data.get("usage", {})


# ------------------------------------------------------------------ rendering

def free_box(label, structure, taken, img_w, img_h):
    """Largest rectangle around the label free of lines and of other labels."""
    h = label["y1"] - label["y0"]
    x0, y0, x1, y1 = label["x0"], label["y0"], label["x1"], label["y1"]
    max_w = max(2.6 * (x1 - x0), (x1 - x0) + 6 * h)
    max_h = (y1 - y0) + 1.5 * h
    occupied = structure | taken
    step = 2

    def clear(a, b, c, d):
        return a >= 0 and b >= 0 and c <= img_w and d <= img_h and not occupied[b:d, a:c].any()

    grow = True
    while grow:
        grow = False
        if x1 - x0 + step <= max_w and clear(x0 - step, y0, x0, y1):
            x0 -= step
            grow = True
        if x1 - x0 + step <= max_w and clear(x1, y0, x1 + step, y1):
            x1 += step
            grow = True
        if y1 - y0 + step <= max_h and clear(x0, y0 - step, x1, y0):
            y0 -= step
            grow = True
        if y1 - y0 + step <= max_h and clear(x0, y1, x1, y1 + step):
            y1 += step
            grow = True
    lh = float(np.median([l["y1"] - l["y0"] for l in label["lines"]]))
    margin = int(0.15 * lh)
    return x0 + margin, y0 + margin // 2, x1 - margin, y1 - margin // 2


def wrap(words, n):
    """Split words into n lines with the shortest longest line (by characters)."""
    if n == 1 or len(words) <= 1:
        return [" ".join(words)]
    best = None
    total = len(" ".join(words))

    def rec(start, k, acc):
        nonlocal best
        if k == 1:
            lines = acc + [" ".join(words[start:])]
            score = max(len(l) for l in lines)
            if best is None or score < best[0]:
                best = (score, lines)
            return
        for end in range(start + 1, len(words) - k + 2):
            rec(end, k - 1, acc + [" ".join(words[start:end])])

    rec(0, min(n, len(words)), [])
    return best[1] if best else [" ".join(words)]


def original_size(label, font_path):
    font = ImageFont.truetype(font_path, 100)
    sizes = []
    for l in label["lines"]:
        width = font.getlength(l["text"])
        if width > 0:
            sizes.append(100 * (l["x1"] - l["x0"]) / width)
    by_height = 1.25 * np.median([l["y1"] - l["y0"] for l in label["lines"]])
    return float(min(np.median(sizes), by_height)) if sizes else float(by_height)


def layout(text, font_path, base_size, box_w, box_h, prefer_lines):
    """Lines, font size and horizontal squeeze (1.0 = none) that fit the box."""
    words = text.split()
    best = None
    for squeeze in (1.0, 0.94, 0.88):
        for n in range(1, min(4, len(words)) + 1):
            lines = wrap(words, n)
            size = base_size
            while size >= max(6, base_size * 0.55):
                font = ImageFont.truetype(font_path, int(size))
                widths = [font.getlength(l) * squeeze for l in lines]
                if max(widths) <= box_w and n * size * 1.18 <= box_h:
                    break
                size -= 1
            fits = size >= max(6, base_size * 0.55)
            # a squeezed glyph reads like a slightly smaller one
            visual = size * (1.0 if n == prefer_lines else 0.85) * (1 - (1 - squeeze) * 0.6)
            score = (fits, visual)
            if best is None or score > best[0]:
                best = (score, lines, int(size), squeeze)
    return best[1], best[2], best[3]


def capacity(label, labels, text_mask, structure, font_path):
    """Characters that fit around a label at its original size."""
    h_img, w_img = text_mask.shape
    taken = np.zeros_like(text_mask)
    for other in labels:
        if other is not label:
            taken[other["y0"]:other["y1"], other["x0"]:other["x1"]] = True
    fx0, fy0, fx1, fy1 = free_box(label, structure, taken, w_img, h_img)
    base = original_size(label, font_path)
    per_line = int((fx1 - fx0) / (0.52 * base))
    lines = max(1, int((fy1 - fy0) / (1.18 * base)))
    return max(len(label["text"]), per_line * lines)


def redraw(rgb, labels, translations, text_mask, structure, font_path):
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    h_img, w_img = text_mask.shape
    drawn = np.zeros_like(text_mask)
    for label, translated in zip(labels, translations):
        x0, y0, x1, y1 = label["x0"], label["y0"], label["x1"], label["y1"]
        lh = int(np.median([l["y1"] - l["y0"] for l in label["lines"]]))
        pad = max(2, int(0.25 * lh))
        ex0, ey0 = max(0, x0 - pad), max(0, y0 - pad)
        ex1, ey1 = min(w_img, x1 + pad), min(h_img, y1 + pad)
        region = rgb[ey0:ey1, ex0:ex1]
        glyphs = text_mask[ey0:ey1, ex0:ex1]
        lines_px = structure[ey0:ey1, ex0:ex1]
        background = region[~glyphs & ~lines_px]
        bg = tuple(int(v) for v in np.median(background, axis=0)) if background.size else (255, 255, 255)
        ink = region[glyphs]
        fg = tuple(int(v) for v in np.median(ink, axis=0)) if ink.size else (0, 0, 0)

        taken = drawn.copy()
        for other in labels:
            if other is not label:
                taken[other["y0"]:other["y1"], other["x0"]:other["x1"]] = True
        fx0, fy0, fx1, fy1 = free_box(label, structure, taken, w_img, h_img)

        # Cover only the glyphs of this label's lines (dilated), never lines or borders.
        own = np.zeros_like(text_mask)
        for l in label["lines"]:
            own[max(0, l["y0"] - 2):l["y1"] + 2, max(0, l["x0"] - 2):l["x1"] + 2] = True
        cover = cv2.dilate((text_mask & own)[ey0:ey1, ex0:ex1].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        cover &= ~lines_px
        patch = np.array(img)[ey0:ey1, ex0:ex1]
        patch[cover] = bg
        img.paste(Image.fromarray(patch), (ex0, ey0))
        draw = ImageDraw.Draw(img)

        base_size = original_size(label, font_path)
        lines, size, squeeze = layout(translated, font_path, base_size, fx1 - fx0, fy1 - fy0, len(label["lines"]))
        font = ImageFont.truetype(font_path, size)
        line_h = size * 1.18
        total_h = line_h * len(lines)
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        top = min(max(cy - total_h / 2, fy0), fy1 - total_h)
        for i, line in enumerate(lines):
            full_w = int(font.getlength(line)) + 2
            w = max(1, int(full_w * squeeze))
            left = int(min(max(cx - w / 2, fx0), fx1 - w))
            y = int(top + i * line_h)
            glyphs = Image.new("L", (full_w, int(line_h) + 2), 0)
            ImageDraw.Draw(glyphs).text((0, line_h / 2), line, font=font, fill=255, anchor="lm")
            if w != full_w:
                glyphs = glyphs.resize((w, glyphs.height), Image.LANCZOS)
            img.paste(Image.new("RGB", glyphs.size, fg), (left, y), glyphs)
            drawn[y:y + glyphs.height, left:left + w] = True
    return np.array(img)


# ------------------------------------------------------------------- pipeline

def caption_for(page, rect):
    best = None
    for block in page.get_text("blocks"):
        bx0, by0, bx1, by1, text = block[:5]
        if re.match(r"\s*(Figure|Fig\.|Figura)\s*\d", text) and abs(by0 - rect.y1) < 80:
            best = " ".join(text.split())
    return best


def source_overlap(text, source_words):
    words = re.findall(r"[a-z]{4,}", text.lower())
    return sum(w in source_words for w in words) / max(1, len(words))


def target_slots(target, page_index, rect, src_page):
    """Where an original image sits in a translated PDF: (page, rect, shared).

    Mono keeps the page; a side-by-side dual puts both halves on one page (the
    translated half is the one whose text is not the source text); an
    alternating dual puts the translated page first or second. In a dual both
    halves can draw the same image object, so it must be covered, not replaced.
    """
    src_pages = len(src_page.parent)
    width = src_page.rect.width
    if len(target) == src_pages:
        page = target[page_index]
        if abs(page.rect.width - 2 * width) >= 2:
            return [(page, rect, False)]
        source_words = set(re.findall(r"[a-z]{4,}", src_page.get_text().lower()))
        h = page.rect.height
        left = source_overlap(page.get_text(clip=fitz.Rect(0, 0, width, h)), source_words)
        right = source_overlap(page.get_text(clip=fitz.Rect(width, 0, 2 * width, h)), source_words)
        return [(page, rect if left < right else rect + (width, 0, width, 0), True)]
    if len(target) == 2 * src_pages:
        source_words = set(re.findall(r"[a-z]{4,}", src_page.get_text().lower()))
        pair = [target[2 * page_index], target[2 * page_index + 1]]
        translated = min(pair, key=lambda p: source_overlap(p.get_text(), source_words))
        return [(translated, rect, True)]
    return []


def replace_in(target, slots, png):
    placed = 0
    for page, rect, shared in slots:
        if shared:
            page.insert_image(rect, stream=png, keep_proportion=False, overlay=True)
            placed += 1
            continue
        for info in page.get_images(full=True):
            rects = page.get_image_rects(info[0])
            for r in rects:
                if abs(r.x0 - rect.x0) < 2 and abs(r.y0 - rect.y0) < 2 and abs(r.x1 - rect.x1) < 2:
                    if len(rects) == 1:
                        page.replace_image(info[0], stream=png)
                        if "/SMask" in target.xref_object(info[0]):  # the old transparency no longer applies
                            target.xref_set_key(info[0], "SMask", "null")
                    else:
                        page.insert_image(rect, stream=png, keep_proportion=False, overlay=True)
                    placed += 1
                    break
    return placed


def translate_figures(pdf_path, targets, li, lo, debug_dir=None, dry_run=False):
    """OCR and translate the figures of pdf_path, then put them into each translated PDF.

    targets: list of (input_pdf, output_pdf). The translated text of the first
    target is used as terminology reference.
    """
    doc = fitz.open(pdf_path)
    outs = [(fitz.open(src), dst) for src, dst in targets]
    font_path = next(p for p in FONT_CANDIDATES if os.path.exists(p))
    lang = TESSERACT_LANGS.get(li, "eng")  # "auto" and unknown languages read as English
    report = []
    done = set()
    for page in doc:
        reference = None
        if outs:
            slot = target_slots(outs[0][0], page.number, page.rect, page)
            if slot:
                ref_page, ref_rect, _ = slot[0]
                reference = " ".join(ref_page.get_text(clip=ref_rect).split())
        for info in page.get_images(full=True):
            xref = info[0]
            if xref in done or min(info[2], info[3]) < MIN_IMAGE_SIDE:
                continue
            done.add(xref)
            rects = page.get_image_rects(xref)
            covered = sum(abs(r & page.rect) for r in rects)
            if rects and covered > 0.6 * abs(page.rect):
                # a scanned page, not a figure: whole-page OCR handles it
                report.append({"page": page.number + 1, "xref": xref, "skipped": "page scan"})
                continue
            try:
                entry = translate_one(doc, page, xref, info[1], outs, li, lo, lang, font_path, reference, debug_dir,
                                      dry_run)
            except Exception as exc:  # keep the text-only translation for this figure
                entry = {"page": page.number + 1, "xref": xref, "error": f"{type(exc).__name__}: {exc}"[:300]}
            report.append(entry)
    for target, dst in outs:
        if any(e.get("replaced", {}).get(os.path.basename(dst)) for e in report) or target.name != dst:
            tmp = dst + ".tmp"
            target.save(tmp, garbage=3, deflate=True)
            target.close()
            os.replace(tmp, dst)
    return report


def translate_one(doc, page, xref, smask, outs, li, lo, lang, font_path, reference, debug_dir, dry_run):
    if True:
        if True:
            rgb = image_rgb(doc, xref, smask)
            text_mask, structure, glyph_h = split_text_and_structure(rgb)
            labels = group_labels(read_lines(text_mask, glyph_h, lang), structure)
            todo = [l for l in labels if worth_translating(l["text"])]
            entry = {"page": page.number + 1, "xref": xref, "size": [rgb.shape[1], rgb.shape[0]],
                     "labels": [l["text"] for l in todo]}
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
                dbg = Image.fromarray(np.where(text_mask[..., None], np.array([200, 0, 0], np.uint8), rgb))
                d = ImageDraw.Draw(dbg)
                for l in labels:
                    d.rectangle([l["x0"], l["y0"], l["x1"], l["y1"]],
                                outline=(220, 0, 0) if worth_translating(l["text"]) else (0, 0, 220), width=3)
                dbg.save(os.path.join(debug_dir, f"p{page.number + 1}-x{xref}-labels.png"))
            rects = page.get_image_rects(xref)
            if not todo or not rects:
                return entry
            context = caption_for(page, rects[0])
            entry["context"] = context
            if dry_run:
                translations, usage = [l["text"] for l in todo], {}
            else:
                caps = [capacity(l, labels, text_mask, structure, font_path) for l in todo]
                entry["max_chars"] = caps
                translations, usage = translate_labels([l["text"] for l in todo], li, lo, context, reference, caps,
                                                       " ".join(page.get_text().split()))
            entry["translations"] = translations
            entry["usage"] = usage
            changed = [(l, t) for l, t in zip(todo, translations) if t.strip() != l["text"].strip()]
            if not changed:
                return entry
            out = redraw(rgb, [c[0] for c in changed], [c[1] for c in changed], text_mask, structure, font_path)
            png = cv2.imencode(".png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))[1].tobytes()
            entry["replaced"] = {}
            for target, dst in outs:
                slots = []
                for r in rects:
                    slots += target_slots(target, page.number, r, page)
                entry["replaced"][os.path.basename(dst)] = replace_in(target, slots, png)
            if debug_dir:
                Image.fromarray(out).save(os.path.join(debug_dir, f"p{page.number + 1}-x{xref}-after.png"))
                Image.fromarray(rgb).save(os.path.join(debug_dir, f"p{page.number + 1}-x{xref}-before.png"))
            return entry


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("original", help="source PDF (figures are read from here)")
    ap.add_argument("--apply", nargs=2, action="append", default=[], metavar=("TRANSLATED", "OUT"),
                    help="translated PDF (mono or dual) and where to write it with translated figures")
    ap.add_argument("--li", default="en")
    ap.add_argument("--lo", default="es")
    ap.add_argument("--debug-dir")
    ap.add_argument("--dry-run", action="store_true", help="OCR and layout only, no translation call")
    args = ap.parse_args()
    report = translate_figures(args.original, args.apply, args.li, args.lo, args.debug_dir, args.dry_run)
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
