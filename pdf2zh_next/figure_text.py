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
# The redrawn labels use Noto Sans (Latin, Greek, Cyrillic) and read left to right,
# and OCR needs the source language installed: other pairs are skipped.
LATIN_TARGETS = {"en", "es", "pt", "fr", "de", "it", "id"}


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

def detect_lines(text_mask, glyph_h, join=0.9):
    """Text lines found by geometry: glyphs joined horizontally, never vertically."""
    k = max(3, int(glyph_h * join))
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


def read_lines(text_mask, glyph_h, lang, join=0.9):
    clean = np.where(text_mask, 0, 255).astype(np.uint8)
    lines = []
    with tempfile.TemporaryDirectory() as tmp:
        for line in detect_lines(text_mask, glyph_h, join):
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
        "- Be consistent: the same source words get the same translation in every label (e.g. labels that "
        "differ only by a number).\n"
        "- Keep normal spacing and punctuation; translate connectors such as '&' or 'and'.\n"
        "- max_chars is the room each label has at its original size. Stay within it when a natural wording "
        "allows: drop articles, use the common short form, or abbreviate the same way the original does "
        "(e.g. 'Norm' -> 'norm.'). Never cut words arbitrarily and never drop a word that carries meaning: "
        "a complete translation in smaller letters is better than a shorter one with less meaning.\n"
        "- Labels come from OCR: fix obvious misreadings using the page text (e.g. 'Al' for 'AI'); a lone dash "
        "between two steps of a process is usually an arrow, write it as '→'.\n"
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


# ---------------------------------------------------- text drawn as images
#
# Some diagrams draw boxes and arrows as vector graphics and put each label in its
# own small image, often only the letters with a transparent background. The
# letters are read from the image itself; the page is rendered to learn the colour
# behind them and the free room inside the box. The old letters are covered with
# a patch of that colour shaped like them (lines and arrows stay), and the
# translation is written as real PDF text.

LABEL_ZOOM = 8  # render scale to sample the colour behind a label (about 576 dpi)
PAGE_ZOOM = 4  # render scale to measure the free room around labels
BOLD_WEIGHT = 0.108  # stroke width / font size: regular sans about 0.09, bold about 0.12
BOLD_FONTS = [p.replace("Regular", "Bold") for p in FONT_CANDIDATES if "Regular" in p]


def image_alpha(doc, xref, smask):
    """Opacity (0..1) of an image from its soft mask, or None."""
    if not smask:
        return None
    try:
        base = fitz.Pixmap(doc, xref)
        mask = fitz.Pixmap(doc, smask)
    except Exception:
        return None
    if (base.width, base.height) != (mask.width, mask.height):
        return None
    return np.frombuffer(mask.samples, np.uint8).reshape(
        mask.height, mask.width, mask.n)[:, :, 0].astype(np.float32) / 255.0


def text_tile(doc, xref, smask):
    """True for an image that only carries text: one ink colour shaped by its transparency."""
    alpha = image_alpha(doc, xref, smask)
    if alpha is None:
        return False
    base = fitz.Pixmap(doc, xref)
    if base.colorspace and base.colorspace.n not in (1, 3):
        base = fitz.Pixmap(fitz.csRGB, base)
    n = base.colorspace.n if base.colorspace else 1
    arr = np.frombuffer(base.samples, np.uint8).reshape(base.height, base.width, base.n)[:, :, :n]
    ink = alpha > 0.5
    if not ink.any() or ink.mean() > 0.45:
        return False
    return float(arr[ink].std(axis=0).max()) < 25


def render(page, zoom, clip=None):
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()


def main_colour(rgb):
    flat = rgb.reshape(-1, 3).astype(np.int32)
    codes, counts = np.unique((flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2], return_counts=True)
    c = int(codes[np.argmax(counts)])
    return np.array([c >> 16, (c >> 8) & 255, c & 255], np.int32)


def best_mask(doc, page, box, xrefs, items):
    """Among images with the same pixels, the one whose mask matches what the page shows at box."""
    clip = render(page, min(4.0, 1500.0 / max(1.0, box.width)), box)
    ink = np.abs(clip.astype(np.int32) - main_colour(clip)).max(axis=2) > 60
    best, score = xrefs[0], -1.0
    for xref in xrefs:
        alpha = image_alpha(doc, xref, items[xref][1])
        if alpha is None:
            continue
        a = cv2.resize(alpha, (ink.shape[1], ink.shape[0]), interpolation=cv2.INTER_AREA) > 0.5
        union = (a | ink).sum()
        s = (a & ink).sum() / union if union else 0.0
        if s > score:
            best, score = xref, s
    return best


def label_placements(doc, page):
    """Where small transparent images and text-only images sit: [{box, xref, smask}], and the text-only xrefs."""
    items = {info[0]: info for info in page.get_images(full=True) if info[1]}
    found, tiles = [], set()
    for info in page.get_image_info(xrefs=True):
        xref = info.get("xref")
        if xref not in items:
            continue  # opaque small images are usually formulas, photos or logos
        item = items[xref]
        box = fitz.Rect(info["bbox"]) & page.rect
        if box.width < 3 or box.height < 2.5:
            continue
        # Images with identical pixels and different masks (one black square shaped into
        # different words) share a fingerprint, so the placement may name the wrong one.
        twins = [x for x, it in items.items() if (it[2], it[3]) == (item[2], item[3])]
        if len(twins) > 1:
            xref = best_mask(doc, page, box, twins, items)
            item = items[xref]
        if min(item[2], item[3]) >= MIN_IMAGE_SIDE:
            if not text_tile(doc, xref, item[1]):
                continue
            tiles.update(twins)
        if all(max(abs(u - v) for u, v in zip(box, p["box"])) > 0.5 for p in found):
            found.append({"box": box, "xref": xref, "smask": item[1]})
    return found, tiles


def stroke_width(mask):
    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    return 2.0 * m.sum() / perimeter if perimeter else 0.0


def read_label_box(doc, page, place, lang, regular):
    """Text lines of one label image (page coordinates). Fills place["bg"], ["rgb"], ["ink"]."""
    box = place["box"]
    clip = render(page, min(LABEL_ZOOM, 4000.0 / max(1.0, box.width)), box)
    bg = main_colour(clip)
    if (np.abs(clip.astype(np.int32) - bg).max(axis=2) <= 16).mean() < 0.5:
        return "background not uniform"
    pix = fitz.Pixmap(doc, place["xref"])
    if pix.colorspace and pix.colorspace.n not in (1, 3):
        pix = fitz.Pixmap(fitz.csRGB, pix)
    n = pix.colorspace.n if pix.colorspace else 1
    base = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :n].astype(np.float32)
    if n == 1:
        base = np.repeat(base, 3, axis=2)
    alpha = image_alpha(doc, place["xref"], place["smask"])
    if alpha is None:
        alpha = np.ones(base.shape[:2], np.float32)
    # The image as the page shows it: composited on the colour behind it.
    rgb = np.clip(base * alpha[..., None] + bg.astype(np.float32) * (1 - alpha[..., None]), 0, 255).astype(np.uint8)
    diff = np.abs(rgb.astype(np.int32) - bg).max(axis=2)
    ink = diff > 60
    count, comp, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    if count <= 1:
        return []
    ws, hs, areas = stats[1:, 2], stats[1:, 3], stats[1:, 4]
    sized = (areas >= 6) & (hs >= 3)
    if not sized.any():
        return []
    glyph_h = float(np.median(hs[sized]))
    keep = ((hs <= 2.2 * glyph_h) & (ws <= 3.0 * glyph_h) & (areas >= 4)
            & ~(ws > 6 * hs) & ~((hs > 6 * ws) & (hs > 1.5 * glyph_h)))
    text = np.isin(comp, np.nonzero(keep)[0] + 1)
    place["bg"] = tuple(int(v) for v in bg)
    # what the cover must hide: the letters with their soft edges
    near = cv2.dilate(text.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    sx, sy = pix.width / box.width, pix.height / box.height
    # About a point beyond the letters: at screen zoom the old letters blur that far.
    r = max(2, int(round(1.0 * sx)))
    place["pad"] = r + 2
    place["sx"], place["sy"] = sx, sy
    cover = cv2.dilate(((diff > 12) & near).astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).astype(bool)
    # Never paint over lines or borders the page draws next to the letters (a diamond, an arrow).
    shown = np.abs(clip.astype(np.int32) - bg).max(axis=2) > 40
    shown = cv2.resize(shown.astype(np.uint8), (pix.width, pix.height), interpolation=cv2.INTER_NEAREST).astype(bool)
    own = cv2.dilate((alpha > 0.05).astype(np.uint8) if place["smask"] else near.astype(np.uint8),
                     np.ones((5, 5), np.uint8)).astype(bool)
    structure = cv2.dilate((shown & ~own).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    place["ink"] = cover & ~structure
    lines = []
    # One image holds one label: words of a larger title are further apart than the median glyph.
    for line in read_lines(text, glyph_h, lang, join=2.5):
        sub = text[line["y0"]:line["y1"], line["x0"]:line["x1"]]
        strong = sub & (diff[line["y0"]:line["y1"], line["x0"]:line["x1"]] >= 0.7 * diff.max())
        ink_rgb = rgb[line["y0"]:line["y1"], line["x0"]:line["x1"]][strong if strong.any() else sub]
        unit = regular.text_length(line["text"], 1)
        size_px = (line["x1"] - line["x0"]) / unit if unit else 1.0
        lines.append({
            "text": line["text"], "px": (line["x0"], line["y0"], line["x1"], line["y1"]),
            "x0": box.x0 + line["x0"] / sx, "y0": box.y0 + line["y0"] / sy,
            "x1": box.x0 + line["x1"] / sx, "y1": box.y0 + line["y1"] / sy,
            "weight": stroke_width(sub) / size_px,
            "fg": tuple(int(v) for v in np.median(ink_rgb, axis=0)),
        })
    return lines


def group_box_lines(lines, place):
    """Consecutive lines of the same size and weight form one label."""
    labels = []
    for line in sorted(lines, key=lambda l: l["y0"]):
        h = line["y1"] - line["y0"]
        last = labels[-1]["lines"][-1] if labels else None
        if last:
            lh = last["y1"] - last["y0"]
            gap = line["y0"] - last["y1"]
            overlap = min(line["x1"], last["x1"]) - max(line["x0"], last["x0"])
            if (gap < 0.9 * max(h, lh) and overlap > 0 and max(h, lh) / max(1e-3, min(h, lh)) < 1.25
                    and line["bold"] == last["bold"]):
                labels[-1]["lines"].append(line)
                continue
        labels.append({"lines": [line], "place": place, "box": place["box"], "bg": place["bg"]})
    for label in labels:
        ls = label["lines"]
        text = ""
        for l in ls:
            text += l["text"] if text.endswith("-") else (" " if text else "") + l["text"]
        label["text"] = text
        label["x0"], label["y0"] = min(l["x0"] for l in ls), min(l["y0"] for l in ls)
        label["x1"], label["y1"] = max(l["x1"] for l in ls), max(l["y1"] for l in ls)
        label["bold"] = ls[0]["bold"]
        label["fg"] = ls[0]["fg"]
        label["lh"] = float(np.median([l["y1"] - l["y0"] for l in ls]))
    return labels


def label_size(label, font):
    """Font size (pt) that gives the original line widths."""
    sizes = [(l["x1"] - l["x0"]) / font.text_length(l["text"], 1) for l in label["lines"]
             if font.text_length(l["text"], 1) > 0]
    by_height = 1.45 * label["lh"]
    return float(min(np.median(sizes), by_height)) if sizes else by_height


class Room:
    """Free space around a label: the rows and columns that only show its background colour."""

    def __init__(self, label, labels, page_rgb):
        z = self.z = PAGE_ZOOM
        lh = self.lh = label["lh"]
        w, h = label["x1"] - label["x0"], label["y1"] - label["y0"]
        self.max_w, self.max_h = max(2.6 * w, w + 6 * lh), h + 1.5 * lh
        self.label = label
        H, W = page_rgb.shape[:2]
        self.ox, self.oy = max(0, int((label["x0"] - self.max_w) * z)), max(0, int((label["y0"] - self.max_h) * z))
        wx1 = min(W, int((label["x1"] + self.max_w) * z) + 1)
        wy1 = min(H, int((label["y1"] + self.max_h) * z) + 1)
        win = page_rgb[self.oy:wy1, self.ox:wx1].astype(np.int32)
        self.occ = np.abs(win - np.array(label["bg"], np.int32)).max(axis=2) > 24
        for l in label["lines"]:
            a, b, c, d = self.px(l, 0.15 * lh)
            self.occ[b:d, a:c] = False
        for other in labels:
            if other is not label:
                a, b, c, d = self.px(other, 0.15 * other["lh"])
                self.occ[b:d, a:c] = True
        self.cx = (label["x0"] + label["x1"]) / 2
        self.cy = (label["y0"] + label["y1"]) / 2
        # vertical room: a narrow column through the centre of the label
        c = int(self.cx * z) - self.ox
        col = self.occ[:, max(0, c - z):c + z + 1].any(axis=1)
        r0, r1 = int(label["y0"] * z) - self.oy, int(np.ceil(label["y1"] * z)) - self.oy
        limit = self.max_h * z
        while r0 > 0 and not col[r0 - 1] and r1 - r0 < limit:
            r0 -= 1
        while r1 < len(col) and not col[r1] and r1 - r0 < limit:
            r1 += 1
        m = 0.1 * lh
        self.top, self.bottom = (r0 + self.oy) / z + m, (r1 + self.oy) / z - m

    def px(self, r, pad=0.0):
        z = self.z
        return (max(0, int((r["x0"] - pad) * z) - self.ox), max(0, int((r["y0"] - pad) * z) - self.oy),
                max(0, int(np.ceil((r["x1"] + pad) * z)) - self.ox),
                max(0, int(np.ceil((r["y1"] + pad) * z)) - self.oy))

    def span(self, y0, y1):
        """Free horizontal run (pt) through the label centre for the rows y0..y1, or None."""
        z = self.z
        r0, r1 = int(y0 * z) - self.oy, int(np.ceil(y1 * z)) - self.oy
        if r0 < 0 or r1 > self.occ.shape[0] or r1 <= r0:
            return None
        cols = self.occ[r0:r1].any(axis=0)
        c = int(self.cx * z) - self.ox
        if cols[c]:
            return None
        left, right = np.nonzero(cols[:c])[0], np.nonzero(cols[c:])[0]
        xl = left[-1] + 1 if len(left) else 0
        xr = c + right[0] if len(right) else len(cols)
        extra = (self.max_w - (self.label["x1"] - self.label["x0"])) / 2
        m = 0.15 * self.lh
        lo = max((xl + self.ox) / z, self.label["x0"] - extra) + m
        hi = min((xr + self.ox) / z, self.label["x1"] + extra) - m
        return (lo, hi) if hi > lo else None

    def chars(self, size):
        """Characters that fit at a font size, for the translator's max_chars."""
        rows = max(1, int((self.bottom - self.top) / (1.18 * size)))
        sp = self.span(self.cy - 0.6 * size, self.cy + 0.6 * size)
        width = sp[1] - sp[0] if sp else self.label["x1"] - self.label["x0"]
        return int(width / (0.52 * size)) * rows


def layout_pt(text, font, base, room, prefer_lines):
    """Lines placed in the free room: [(line, left, top)], font size (pt) and horizontal squeeze."""
    words = text.split()
    best = None
    floor = max(3.0, base * 0.55)

    def place(lines, size, squeeze):
        line_h = size * 1.18
        total = line_h * len(lines)
        if total > room.bottom - room.top:
            return None
        top = min(max(room.cy - total / 2, room.top), room.bottom - total)
        out = []
        for i, line in enumerate(lines):
            y0 = top + i * line_h
            sp = room.span(y0, y0 + line_h)
            w = font.text_length(line, size) * squeeze
            if not sp or w > sp[1] - sp[0]:
                return None
            out.append((line, min(max(room.cx - w / 2, sp[0]), sp[1] - w), y0))
        return out

    for squeeze in (1.0, 0.94, 0.88):
        for n in range(1, min(4, len(words)) + 1):
            lines = wrap(words, n)
            size, placed = base, None
            while size >= floor and not placed:
                placed = place(lines, size, squeeze)
                if not placed:
                    size -= 0.1
            fits = placed is not None
            if not fits:
                size = floor
                line_h = size * 1.18
                top = room.cy - line_h * len(lines) / 2
                placed = [(l, room.cx - font.text_length(l, size) * squeeze / 2, top + i * line_h)
                          for i, l in enumerate(lines)]
            visual = size * (1.0 if n == prefer_lines else 0.85) * (1 - (1 - squeeze) * 0.6)
            score = (fits, visual)
            if best is None or score > best[0]:
                best = (score, placed, size, squeeze)
    return best[1], best[2], best[3]


def subset_font(path, text):
    """Only the glyphs used, so the PDF does not grow by a whole font per figure."""
    try:
        import io
        from fontTools import subset
        opts = subset.Options()
        opts.name_IDs = ["*"]
        opts.notdef_outline = True
        font = subset.load_font(path, opts)
        sub = subset.Subsetter(opts)
        sub.populate(text=text + " ")
        sub.subset(font)
        buf = io.BytesIO()
        subset.save_font(font, buf, opts)
        return buf.getvalue()
    except Exception:
        with open(path, "rb") as fh:
            return fh.read()


def cover_paths(place, labels):
    """Outlines (pt) of the letters being replaced, padded: filled with the background colour they hide them.

    Vector shapes, not an image: a transparent image is blended with white at its
    soft edges and would leave a light outline around every letter.
    """
    keep = np.zeros_like(place["ink"])
    for label in labels:
        for l in label["lines"]:
            x0, y0, x1, y1 = l["px"]
            pad = max(place["pad"], int(0.25 * (y1 - y0)))
            keep[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] = True
    mask = (place["ink"] & keep).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    box, sx, sy = place["box"], place["sx"], place["sy"]
    paths = []
    for c in contours:
        c = cv2.approxPolyDP(c, 0.7, True).reshape(-1, 2)
        if len(c) >= 3:
            paths.append([fitz.Point(box.x0 + (x + 0.5) / sx, box.y0 + (y + 0.5) / sy) for x, y in c])
    return paths


def write_labels(page, offset, labels, translations, fonts):
    """Cover the replaced letters and write the translations as PDF text."""
    dx, dy = offset
    names = {}
    for bold, path in fonts.items():
        used = "".join(t for l, t in zip(labels, translations) if l["bold"] == bold)
        if used:
            names[bold] = "FigSansBold" if bold else "FigSans"
            page.insert_font(fontname=names[bold], fontbuffer=subset_font(path, used))
    places = {}
    for label in labels:
        places.setdefault(id(label["place"]), (label["place"], []))[1].append(label)
    shape = page.new_shape()
    for place, own in places.values():
        paths = cover_paths(place, own)
        for path in paths:
            shape.draw_polyline([pt + (dx, dy) for pt in path])
        if paths:
            shape.finish(color=None, fill=tuple(v / 255 for v in place["bg"]), width=0, closePath=True)
    for label in labels:
        font = label["font"]
        placed, size, squeeze = label["layout"]
        line_h = size * 1.18
        asc, desc = font.ascender, font.descender
        fg = tuple(v / 255 for v in label["fg"])
        for line, left, top in placed:
            base_y = top + (line_h - (asc - desc) * size) / 2 + asc * size
            point = fitz.Point(left + dx, base_y + dy)
            morph = (point, fitz.Matrix(squeeze, 1)) if squeeze != 1.0 else None
            shape.insert_text(point, line, fontsize=size, fontname=names[label["bold"]], color=fg, morph=morph)
    shape.commit(overlay=True)


def translate_label_images(doc, page, places, outs, li, lo, lang, reference, debug_dir, dry_run, cache):
    entry = {"page": page.number + 1, "kind": "label images", "images": len(places)}
    regular = next(p for p in FONT_CANDIDATES if os.path.exists(p))
    bold_path = next((p for p in BOLD_FONTS if os.path.exists(p)), regular)
    fonts = {False: regular, True: bold_path}
    measure = {k: fitz.Font(fontfile=v) for k, v in fonts.items()}
    labels, skipped = [], 0
    for place in places:
        lines = read_label_box(doc, page, place, lang, measure[False])
        if isinstance(lines, str):
            skipped += 1
            continue
        for l in lines:
            l["bold"] = l["weight"] > BOLD_WEIGHT and bold_path != regular
        if lines:
            labels += group_box_lines(lines, place)
    todo = [l for l in labels if worth_translating(l["text"])]
    entry["labels"] = [l["text"] for l in todo]
    if skipped:
        entry["skipped"] = f"{skipped} on a non-uniform background"
    if not todo:
        return entry
    page_rgb = render(page, PAGE_ZOOM)
    for l in todo:
        l["font"] = measure[l["bold"]]
        l["size"] = label_size(l, l["font"])
        l["room"] = Room(l, labels, page_rgb)
    union = fitz.Rect(todo[0]["box"])
    for l in todo[1:]:
        union |= l["box"]
    context = caption_for(page, union)
    entry["context"] = context
    # Labels repeated on other pages (a logo, a legend) are translated once per document.
    new = [l for l in todo if l["text"] not in cache]
    usage = {}
    if dry_run:
        cache.update({l["text"]: l["text"] for l in new})
    elif new:
        caps = [max(len(l["text"]), l["room"].chars(0.8 * l["size"])) for l in new]
        entry["max_chars"] = caps
        out, usage = translate_labels([l["text"] for l in new], li, lo, context, reference, caps,
                                      " ".join(page.get_text().split()))
        cache.update({l["text"]: t for l, t in zip(new, out)})
    translations = [cache[l["text"]] for l in todo]
    entry["translations"] = translations
    entry["usage"] = usage
    changed = [(l, t) for l, t in zip(todo, translations) if t.strip() != l["text"].strip()]
    if not changed:
        return entry
    for l, t in changed:
        l["layout"] = layout_pt(t, l["font"], l["size"], l["room"], len(l["lines"]))
    entry["replaced"] = {}
    for target, dst in outs:
        placed = 0
        for tpage, trect, _ in target_slots(target, page.number, page.rect, page):
            write_labels(tpage, (trect.x0 - page.rect.x0, trect.y0 - page.rect.y0),
                         [c[0] for c in changed], [c[1] for c in changed], fonts)
            placed += 1
        entry["replaced"][os.path.basename(dst)] = placed
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        clip = union + (-12, -12, 12, 12)
        Image.fromarray(render(page, 3, clip)).save(os.path.join(debug_dir, f"p{page.number + 1}-labels-before.png"))
        if outs:
            for tpage, trect, _ in target_slots(outs[0][0], page.number, page.rect, page):
                off = (trect.x0 - page.rect.x0, trect.y0 - page.rect.y0, trect.x0 - page.rect.x0, trect.y0 - page.rect.y0)
                Image.fromarray(render(tpage, 3, clip + off)).save(
                    os.path.join(debug_dir, f"p{page.number + 1}-labels-after.png"))
    return entry


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
    if li not in TESSERACT_LANGS or lo not in LATIN_TARGETS:
        return [{"skipped": f"unsupported language pair {li}->{lo}"}]
    doc = fitz.open(pdf_path)
    outs = [(fitz.open(src), dst) for src, dst in targets]
    font_path = next(p for p in FONT_CANDIDATES if os.path.exists(p))
    lang = TESSERACT_LANGS.get(li, "eng")  # "auto" and unknown languages read as English
    report = []
    done = set()
    label_cache = {}
    for page in doc:
        reference = None
        if outs:
            slot = target_slots(outs[0][0], page.number, page.rect, page)
            if slot:
                ref_page, ref_rect, _ = slot[0]
                reference = " ".join(ref_page.get_text(clip=ref_rect).split())
        places, tiles = label_placements(doc, page)
        if places:
            try:
                entry = translate_label_images(doc, page, places, outs, li, lo, lang, reference, debug_dir, dry_run,
                                               label_cache)
            except Exception as exc:  # keep the text-only translation for these labels
                entry = {"page": page.number + 1, "kind": "label images", "error": f"{type(exc).__name__}: {exc}"[:300]}
            if entry.get("labels") or entry.get("error"):
                report.append(entry)
        for info in page.get_images(full=True):
            xref = info[0]
            if xref in done or xref in tiles or min(info[2], info[3]) < MIN_IMAGE_SIDE:
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
