import math
import cv2
import numpy as np
from config import CELL_W, CELL_H, TITLE_H, GAP, BG, TITLE_BG, TITLE_FG, FONT_SCALE, FONT_THICK

FONT = cv2.FONT_HERSHEY_SIMPLEX


def make_grid(panels: list[tuple[str, np.ndarray]]) -> np.ndarray:
    n = len(panels)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    cell_outer_w = CELL_W + GAP
    cell_outer_h = CELL_H + TITLE_H + GAP

    grid_w = cols * cell_outer_w + GAP
    grid_h = rows * cell_outer_h + GAP
    grid = np.full((grid_h, grid_w, 3), BG, dtype=np.uint8)

    for i, (title, frame) in enumerate(panels):
        r, c = divmod(i, cols)
        x = GAP + c * cell_outer_w
        y = GAP + r * cell_outer_h

        cv2.rectangle(grid, (x, y), (x + CELL_W, y + TITLE_H), TITLE_BG, -1)
        cv2.putText(grid, title, (x + 6, y + TITLE_H - 7),
                    FONT, FONT_SCALE, TITLE_FG, FONT_THICK, cv2.LINE_AA)

        cell = cv2.resize(frame, (CELL_W, CELL_H))
        grid[y + TITLE_H : y + TITLE_H + CELL_H, x : x + CELL_W] = cell

    return grid
