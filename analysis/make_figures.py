"""Figures and re-analysis data for Research Report 02 (Mask R-CNN forgery detection).

Reads the competition dataset, the training notebooks, and the team deck, and writes:
  data/masks.csv            one row per forged training image: size, instances, regions,
                            and what each notebook version's mask loader produced for it
  data/loader_summary.csv   the loader check aggregated by notebook version and split
  data/run_a_epochs.csv     per-epoch losses of the November run, parsed from its notebook log
  figures/*.svg|*.html      charts (rasterize with rasterize_charts.ps1)
  figures/*.png             composite image figures

Usage:  python -I make_figures.py <dataset_dir> <notebook_dir>
  dataset_dir   the unzipped Kaggle competition data (train_images/, train_masks/, ...)
  notebook_dir  the folder holding the train_maskrcnn_pytorch*.ipynb copies
Needs numpy, Pillow, and OpenCV (connected components only).
"""
import base64, csv, io, json, os, re, statistics as st, sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

if len(sys.argv) < 2:
    sys.exit("usage: python make_figures.py <dataset_dir> [notebook_dir]")
DATASET = sys.argv[1]
NOTEBOOKS = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figures")
DECK = os.path.join(HERE, "..", "sources", "deck-media")
FORGED = os.path.join(DATASET, "train_images", "forged")
MASKS = os.path.join(DATASET, "train_masks")

NB_NOV = "train_maskrcnn_pytorch.ipynb"        # November 24-25 Colab run (identical copy: "(1)")
NB_DEC = "train_maskrcnn_pytorch (2).ipynb"    # December 3 lab-GPU notebook


# ---------- the two mask loaders, copied from the notebooks ----------
def loader_nov(mask, h, w):
    """ForgeryDataset.__getitem__ mask handling, November notebook."""
    if len(mask.shape) == 3:
        mask = mask[:, :, 0] if mask.shape[2] > 0 else mask.sum(axis=2)
    mask = (mask > 0).astype(np.uint8)
    if mask.shape[:2] != (h, w):
        mask = np.array(Image.fromarray(mask).resize((w, h), Image.NEAREST))
    return mask


def loader_dec(mask, h, w):
    """ForgeryDataset.__getitem__ mask handling, December notebooks (rotation omitted)."""
    if len(mask.shape) == 3:
        if mask.shape[0] == 1:
            mask = mask[0, :, :]
        elif mask.shape[2] == 1:
            mask = mask[:, :, 0]
        else:
            mask = mask.sum(axis=2)
    mask = (mask > 0).astype(np.uint8)
    if mask.shape[:2] != (h, w):
        mask = np.array(Image.fromarray(mask).resize((w, h), Image.NEAREST))
    return mask


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else inter / union


def bbox(m):
    ys, xs = np.nonzero(m)
    return None if len(xs) == 0 else (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def split_ids():
    """Reproduce the notebooks' 80/20 split: listdir order, then a seed-42 shuffle."""
    files = [f for f in os.listdir(FORGED) if f.endswith((".png", ".jpg", ".jpeg"))]
    np.random.seed(42)
    np.random.shuffle(files)
    cut = int(len(files) * 0.8)
    return {os.path.splitext(f)[0]: ("train" if i < cut else "val") for i, f in enumerate(files)}, files


def reanalyze():
    split, order = split_ids()
    rows = []
    for name in sorted(split, key=int):
        img = Image.open(os.path.join(FORGED, name + ".png"))
        w, h = img.size
        raw = np.load(os.path.join(MASKS, name + ".npy"))
        k = raw.shape[0] if raw.ndim == 3 else 1
        truth = (raw.reshape(-1, h, w).max(axis=0) > 0).astype(np.uint8)
        n_regions = cv2.connectedComponents(truth, connectivity=8)[0] - 1
        nov, dec = loader_nov(raw, h, w), loader_dec(raw, h, w)
        tb = bbox(truth)
        fill = truth.sum() / ((tb[2] - tb[0] + 1) * (tb[3] - tb[1] + 1))
        rows.append({
            "id": name, "split": split[name], "width": w, "height": h, "mask_shape": "x".join(map(str, raw.shape)),
            "instances": k, "regions": n_regions, "coverage": round(truth.mean(), 5), "box_fill": round(fill, 4),
            "nov_empty": int(nov.sum() == 0), "nov_iou": round(iou(nov, truth), 4),
            "dec_exact": int(np.array_equal(dec, truth)), "dec_iou": round(iou(dec, truth), 4),
            "dec_bbox": " ".join(map(str, bbox(dec) or ())),
        })
    with open(os.path.join(DATA, "masks.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    summary = []
    for part in ("train", "val", "all"):
        sel = [r for r in rows if part == "all" or r["split"] == part]
        multi = [r for r in sel if r["instances"] > 1]
        summary.append({
            "split": part, "images": len(sel), "multi_instance": len(multi),
            "nov_empty_targets": sum(r["nov_empty"] for r in sel),
            "nov_median_iou": round(st.median(r["nov_iou"] for r in sel), 4),
            "dec_exact_targets": sum(r["dec_exact"] for r in sel),
            "dec_corrupted": sum(1 - r["dec_exact"] for r in sel),
            "dec_median_iou_multi": round(st.median(r["dec_iou"] for r in multi), 4) if multi else "",
        })
    with open(os.path.join(DATA, "loader_summary.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summary[0]))
        wr.writeheader()
        wr.writerows(summary)

    # The December notebook printed the first three validation samples; matching them checks the split.
    val = [os.path.splitext(f)[0] for f in order[int(len(order) * 0.8):]][:3]
    by_id = {r["id"]: r for r in rows}
    check = [(by_id[v]["height"], by_id[v]["width"], by_id[v]["dec_bbox"]) for v in val]
    print("first validation samples (h, w, December-loader box):", check)
    for s in summary:
        print(s)
    return rows


def parse_run_a():
    """Per-epoch losses printed by the November notebook's training loop."""
    nb = json.load(open(os.path.join(NOTEBOOKS, NB_NOV), encoding="utf-8"))
    text = "".join("".join(o.get("text", [])) for c in nb["cells"] if c["cell_type"] == "code"
                   for o in c.get("outputs", []) if o.get("output_type") == "stream")
    pat = re.compile(r"Train Loss: ([\d.]+) \| Train Mask Loss: ([\d.]+)\s+Val Loss: ([\d.]+) \| "
                     r"Val Mask Loss: ([\d.]+)\s+Learning Rate: ([\d.]+)")
    ep = [tuple(map(float, m.groups())) for m in pat.finditer(text)]
    secs = lambda t: int(t.split(":")[0]) * 60 + int(t.split(":")[1])
    train_t = [secs(t) for t in re.findall(r"Epoch \d+: 100%[^\[]*\[(\d+:\d+)<", text)]
    val_t = [secs(t) for t in re.findall(r"Evaluating: 100%[^\[]*\[(\d+:\d+)<", text)]
    last = re.findall(r"Epoch \d+: 100%.*?loss=([\d.]+), mask=([\d.]+)", text)
    with open(os.path.join(DATA, "run_a_epochs.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["epoch", "train_loss", "train_mask_loss", "val_loss", "val_mask_loss", "learning_rate",
                     "train_s", "val_s", "last_batch_loss", "last_batch_mask_loss"])
        for i, e in enumerate(ep):
            wr.writerow([i + 1, *e, train_t[i], val_t[i], *last[i]])
    print(f"run A: {len(ep)} epochs, training {sum(train_t)} s, validation {sum(val_t)} s;",
          sum(float(m) == 0 for _, m in last), "epochs end on a batch with zero mask loss")
    return ep


# ---------- SVG charts (house palette, light surface) ----------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FONT = "Arial, Helvetica, sans-serif"


def panel(x0, y0, w, h, title, xmax, ymin, ymax, yticks, series, fmt=str, xlabel="Epoch"):
    def sx(x):
        return x0 + (x - 1) / (xmax - 1) * w

    def sy(y):
        return y0 + h - (y - ymin) / (ymax - ymin) * h
    out = [f'<text x="{x0}" y="{y0 - 10}" font-size="11.5" font-weight="bold" fill="{INK}">{title}</text>']
    for t in yticks:
        y = sy(t)
        out.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{x0 - 6}" y="{y + 3.5:.1f}" font-size="9.5" fill="{MUTED}" text-anchor="end">{fmt(t)}</text>')
    out.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{y0 + h}" y2="{y0 + h}" stroke="{AXIS}" stroke-width="1"/>')
    for t in range(1, xmax + 1):
        out.append(f'<text x="{sx(t):.1f}" y="{y0 + h + 14}" font-size="9.5" fill="{MUTED}" text-anchor="middle">{t}</text>')
    out.append(f'<text x="{x0 + w / 2}" y="{y0 + h + 29}" font-size="9.5" fill="{INK2}" text-anchor="middle">{xlabel}</text>')
    for label, color, pts in series:
        path = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)
        out.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2" '
                   f'stroke-linejoin="round" stroke-linecap="round"/>')
        for x, y in pts:
            out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>')
    return out


def legend(x, y, items):
    out, cx = [], x
    for label, color in items:
        out.append(f'<line x1="{cx}" x2="{cx + 16}" y1="{y - 3.5}" y2="{y - 3.5}" stroke="{color}" stroke-width="2.5" stroke-linecap="round"/>')
        out.append(f'<text x="{cx + 22}" y="{y}" font-size="10" fill="{INK2}">{label}</text>')
        cx += 22 + len(label) * 5.6 + 18
    return out


def write_chart(name, width, height, body):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" font-family="{FONT}">'
           f'<rect width="{width}" height="{height}" fill="#ffffff"/>' + "".join(body) + "</svg>")
    with open(os.path.join(FIGS, name + ".svg"), "w", encoding="utf-8") as f:
        f.write(svg)
    with open(os.path.join(FIGS, name + ".html"), "w", encoding="utf-8") as f:
        f.write(f'<!doctype html><html><head><meta charset="utf-8"><style>html,body{{margin:0;background:#fff}}'
                f'svg{{display:block}}</style></head><body>{svg}</body></html>')


def chart_run_a(ep):
    items = [("Train", SERIES[0]), ("Validation", SERIES[1])]
    body = legend(56, 20, items)
    body += panel(56, 58, 262, 160, "Total loss (epoch mean)", len(ep), 0.15, 0.35, [0.15, 0.2, 0.25, 0.3, 0.35],
                  [("Train", SERIES[0], [(i, e[0]) for i, e in enumerate(ep, 1)]),
                   ("Validation", SERIES[1], [(i, e[2]) for i, e in enumerate(ep, 1)])], fmt=lambda t: f"{t:.2f}")
    body += panel(386, 58, 262, 160, "Mask loss (epoch mean)", len(ep), 0.1, 0.25, [0.1, 0.15, 0.2, 0.25],
                  [("Train", SERIES[0], [(i, e[1]) for i, e in enumerate(ep, 1)]),
                   ("Validation", SERIES[1], [(i, e[3]) for i, e in enumerate(ep, 1)])], fmt=lambda t: f"{t:.2f}")
    write_chart("fig2_run_a_losses", 672, 256, body)


# ---------- composite image figures ----------
def arial(size, bold=False):
    return ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf", size)


def labeled_row(panels, out_name, gap=28, label_h=34, scale_h=None):
    """panels: list of (label, image). Images scaled to a common height."""
    hgt = scale_h or min(im.height for _, im in panels)
    scaled = [(lab, im.resize((round(im.width * hgt / im.height), hgt), Image.LANCZOS)) for lab, im in panels]
    width = sum(im.width for _, im in scaled) + gap * (len(scaled) - 1)
    canvas = Image.new("RGB", (width, hgt + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = arial(round(label_h * 0.62), bold=True) if label_h else None
    x = 0
    for lab, im in scaled:
        canvas.paste(im, (x, label_h))
        if lab:
            draw.text((x, 2), lab, fill=(28, 23, 18), font=font)
        x += im.width + gap
    canvas.save(os.path.join(FIGS, out_name), optimize=True)
    print(out_name, canvas.size)
    return canvas


def overlay(name, mask=None, color=(180, 74, 40)):
    """The forged image with a mask tinted over it (defaults to the true union mask)."""
    img = Image.open(os.path.join(FORGED, name + ".png")).convert("RGB")
    if mask is None:
        raw = np.load(os.path.join(MASKS, name + ".npy"))
        mask = raw.reshape(-1, img.height, img.width).max(axis=0) > 0
    arr = np.array(img).astype(np.float32)
    tint = np.array(color, dtype=np.float32)
    m = mask.astype(bool)
    arr[m] = arr[m] * 0.45 + tint * 0.55
    out = Image.fromarray(arr.astype(np.uint8))
    edge = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    a = np.array(out)
    a[edge] = color
    return Image.fromarray(a)


def notebook_images(nb_name):
    nb = json.load(open(os.path.join(NOTEBOOKS, nb_name), encoding="utf-8"))
    imgs = []
    for c in nb["cells"]:
        for o in c.get("outputs", []):
            if "image/png" in o.get("data", {}):
                imgs.append(Image.open(io.BytesIO(base64.b64decode(o["data"]["image/png"]))).convert("RGB"))
    return imgs


def square(img, mask_name, side):
    """A square crop centred on the forged pixels, scaled to side x side."""
    raw = np.load(os.path.join(MASKS, mask_name + ".npy"))
    m = raw.reshape(-1, img.height, img.width).max(axis=0)
    ys, xs = np.nonzero(m)
    s = min(img.width, img.height)
    cx = min(max(int(xs.mean()) - s // 2, 0), img.width - s)
    cy = min(max(int(ys.mean()) - s // 2, 0), img.height - s)
    return img.crop((cx, cy, cx + s, cy + s)).resize((side, side), Image.LANCZOS)


def stack(tiles, out_name, gap=26):
    width = max(t.width for t in tiles)
    canvas = Image.new("RGB", (width, sum(t.height for t in tiles) + gap * (len(tiles) - 1)), "white")
    y = 0
    for t in tiles:
        canvas.paste(t, ((width - t.width) // 2, y))
        y += t.height + gap
    canvas.save(os.path.join(FIGS, out_name), optimize=True)
    print(out_name, canvas.size)


def figures(rows):
    by_id = {r["id"]: r for r in rows}
    imgs = notebook_images(NB_DEC)
    pred = imgs[8:18]  # the ten saved validation predictions, in the order the notebook printed them

    # Figure 1: four of the scientific image types, with their true masks.
    picks = [("19633", "Specimen photo"), ("55541", "Histology"), ("3367", "Cell culture"), ("6811", "Western blot")]
    labeled_row([(f"{lab}, {by_id[p]['instances']} instance{'s' if by_id[p]['instances'] > 1 else ''}", overlay(p))
                 for p, lab in picks], "fig1_dataset_examples.png", scale_h=360, label_h=60, gap=40)

    # Figure 5: what the December loader hands the model for a two-instance mask.
    name = "55541"
    r = by_id[name]
    raw = np.load(os.path.join(MASKS, name + ".npy"))
    dec = loader_dec(raw, r["height"], r["width"])
    labeled_row([("True mask (2 instances)", overlay(name)),
                 ("Training target from the December loader", overlay(name, dec, color=(42, 120, 214)))],
                "fig5_loader_bug.png", scale_h=520, label_h=30)

    # Figure 3: the December notebook's own loss plot (run B).
    imgs[7].save(os.path.join(FIGS, "fig3_run_b_plot.png"), optimize=True)

    # Figure 4: two saved predictions (left: target as loaded, right: prediction above 0.5).
    stack([pred[i] for i in (1, 2)], "fig4_predictions.png")

    # Cover: three dataset images with their true masks, as square tiles.
    tiles = [square(overlay(p), p, 520) for p in ("19633", "55541", "37451")]
    labeled_row([("", t) for t in tiles], "cover_strip.png", label_h=0, gap=8)

    # Dissemination: the title slide of the team's deck (an AI-generated slide image).
    if os.path.exists(os.path.join(DECK, "image11.jpg")):
        Image.open(os.path.join(DECK, "image11.jpg")).convert("RGB").save(os.path.join(FIGS, "fig6_title_slide.png"), optimize=True)


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)
    rows = reanalyze()
    ep = parse_run_a()
    chart_run_a(ep)
    figures(rows)
