"""
Hand-annotation tool for the gold mask set (segmentation/gold/gold_set.csv).

Model masks are deliberately NOT shown, so the annotation is not anchored on UNet or SAM.

    python segmentation/annotate.py [--annotator MM]

Controls
  left-drag            trace the lesion border freehand (or click to add single vertices)
  right-click          undo the last stroke / vertex
  enter                save the closed polygon and go to the next image
  u                    toggle "border uncertain" for this image (saved in the log)
  n                    no single clear lesion (skip; logged, no mask written)
  r                    clear the polygon
  b                    back to the previous image (its saved mask is kept until you re-save)
  q                    quit (progress is kept; rerun to resume)
Use the toolbar zoom/pan freely; clicks are ignored while a toolbar tool is active.
Include the whole pigmented/erythematous lesion; exclude hair, ruler marks and surrounding erythema
that is not part of the lesion. Trace where you would put a biopsy margin's inner edge.
"""
import argparse
import datetime as dt
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "uncertaintyNet-main/datasets_768/ISIC_clinical"
GOLD = ROOT / "segmentation/gold"


def polygon_mask(points, shape):
    canvas = Image.new("L", (shape[1], shape[0]), 0)
    if len(points) >= 3:
        ImageDraw.Draw(canvas).polygon([tuple(p) for p in points], fill=255)
    return canvas


class Annotator:
    def __init__(self, todo, annotator):
        self.todo, self.k, self.annotator = todo, 0, annotator
        self.strokes, self.drawing, self.uncertain = [], False, False
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title("MedirAI gold mask annotation")
        for ev, fn in [("button_press_event", self.press), ("motion_notify_event", self.motion),
                       ("button_release_event", self.release), ("key_press_event", self.key)]:
            self.fig.canvas.mpl_connect(ev, fn)
        self.load()

    # --- state -------------------------------------------------------------------------------------
    def points(self):
        return [p for s in self.strokes for p in s]

    def load(self):
        if self.k >= len(self.todo):
            print("All images annotated.")
            plt.close(self.fig)
            return
        self.isic_id = self.todo[self.k]
        self.img = np.array(Image.open(IMAGES / f"{self.isic_id}.jpg").convert("RGB"))
        self.strokes, self.uncertain = [], False
        self.ax.clear()
        self.ax.imshow(self.img)
        self.line, = self.ax.plot([], [], "-", color="cyan", lw=1.5)
        self.ax.set_axis_off()
        self.redraw()

    def redraw(self):
        pts = self.points()
        if pts:
            xs, ys = zip(*(pts + [pts[0]]))
            self.line.set_data(xs, ys)
        else:
            self.line.set_data([], [])
        done = len(list((GOLD / "masks").glob("*.png"))) + len(skipped_ids())
        self.ax.set_title(f"{self.isic_id}   [{done}/{len(all_ids())} done]"
                          f"{'   BORDER UNCERTAIN' if self.uncertain else ''}\n"
                          "drag/click=trace  right=undo  enter=save  u=uncertain  n=no clear lesion  b=back  q=quit",
                          fontsize=9)
        self.fig.canvas.draw_idle()

    def log(self, status):
        row = pd.DataFrame([{"isic_id": self.isic_id, "status": status, "border_uncertain": self.uncertain,
                             "n_vertices": len(self.points()), "annotator": self.annotator,
                             "time": dt.datetime.now().isoformat(timespec="seconds")}])
        path = GOLD / "annotation_log.csv"
        row.to_csv(path, mode="a", header=not path.exists(), index=False)

    # --- events ------------------------------------------------------------------------------------
    def toolbar_busy(self):
        tb = self.fig.canvas.toolbar
        return tb is not None and str(tb.mode) != ""

    def press(self, ev):
        if ev.inaxes != self.ax or self.toolbar_busy():
            return
        if ev.button == 1:
            self.strokes.append([(ev.xdata, ev.ydata)])
            self.drawing = True
        elif ev.button == 3 and self.strokes:
            self.strokes.pop()
        self.redraw()

    def motion(self, ev):
        if not self.drawing or ev.inaxes != self.ax:
            return
        x0, y0 = self.strokes[-1][-1]
        if (ev.xdata - x0) ** 2 + (ev.ydata - y0) ** 2 > 9:
            self.strokes[-1].append((ev.xdata, ev.ydata))
            self.redraw()

    def release(self, ev):
        self.drawing = False

    def key(self, ev):
        if ev.key == "enter":
            if len(self.points()) < 3:
                print("Need at least 3 points.")
                return
            polygon_mask(self.points(), self.img.shape).save(GOLD / "masks" / f"{self.isic_id}.png")
            self.log("annotated")
            self.k += 1
            self.load()
        elif ev.key == "u":
            self.uncertain = not self.uncertain
            self.redraw()
        elif ev.key == "n":
            self.log("no_clear_lesion")
            self.k += 1
            self.load()
        elif ev.key == "r":
            self.strokes = []
            self.redraw()
        elif ev.key == "b" and self.k > 0:
            self.k -= 1
            self.load()
        elif ev.key == "q":
            plt.close(self.fig)


def all_ids():
    return pd.read_csv(GOLD / "gold_set.csv").isic_id.tolist()


def skipped_ids():
    path = GOLD / "annotation_log.csv"
    if not path.exists():
        return set()
    log = pd.read_csv(path).drop_duplicates("isic_id", keep="last")
    return set(log[log.status == "no_clear_lesion"].isic_id)


def main(opts):
    (GOLD / "masks").mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in (GOLD / "masks").glob("*.png")} | skipped_ids()
    todo = [i for i in all_ids() if i not in done]
    print(f"{len(todo)} images to annotate ({len(done)} already done).")
    if todo:
        Annotator(todo, opts.annotator)
        plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotator", default="MM")
    main(ap.parse_args())
