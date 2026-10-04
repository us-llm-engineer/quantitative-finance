"""Build uniform README gallery images: each row of two figures shares one canvas aspect ratio,
so both images render at the same height and fill their table cell. Source PNGs are unchanged;
outputs go to figures/readme/. Usage: python scripts/make_readme_gallery.py"""
from math import sqrt
from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "readme"
WIDTH = 1600
R = "fig-4.{}-l4-20261003-full-04-analysis"
ROWS = [("fig-1.0", "fig-1.7"), ("fig-1.2", "fig-1.6"),
        ("fig-5.1", "fig-5.3"), ("fig-5.2", "fig-5.15"), ("fig-5.4", "fig-5.14"), ("fig-5.5", "fig-5.6"),
        ("fig-5.7", "fig-5.16"), ("fig-5.8", "fig-5.9"), ("fig-5.10", "fig-5.13"), ("fig-5.11", "fig-5.12"),
        (R.format(5), R.format(6))]


def trim(im, pad=12):
    bg = Image.new(im.mode, im.size, (255, 255, 255))
    box = ImageChops.difference(im, bg).getbbox()
    if box:
        im = im.crop((max(box[0] - pad, 0), max(box[1] - pad, 0), min(box[2] + pad, im.width), min(box[3] + pad, im.height)))
    return im


def fit(im, ratio):
    h = round(WIDTH / ratio)
    s = min(WIDTH / im.width, h / im.height)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    canvas = Image.new("RGB", (WIDTH, h), (255, 255, 255))
    canvas.paste(im, ((WIDTH - im.width) // 2, (h - im.height) // 2))
    return canvas


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for a, b in ROWS:
        ims = [trim(Image.open(ROOT / "figures" / f"{n}.png").convert("RGB")) for n in (a, b)]
        ratio = sqrt((ims[0].width / ims[0].height) * (ims[1].width / ims[1].height))
        for n, im in zip((a, b), ims):
            fit(im, ratio).save(OUT / f"{n}.png", optimize=True)
            print(f"{n}: canvas {WIDTH}x{round(WIDTH / ratio)}")


if __name__ == "__main__":
    main()
