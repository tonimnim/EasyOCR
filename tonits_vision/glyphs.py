"""Find score digits by their shape, next to the logo.

CRAFT often misses a lone digit in a box beside the round logo ("0" and "1" especially), even when the
banner is re-detected at a higher zoom. A score is a short run of high-contrast glyphs right beside the
logo, so it can be found directly: threshold the area beside the centre and keep components that are
enclosed (don't touch the edge of the area, unlike the background or the logo) and glyph-sized. Both
polarities are kept, so dark digits on the yellow boxes and light digits on dark backgrounds both work.
"""

import cv2


def find_glyphs(grey, region, line_h):
    """Boxes of enclosed glyph-sized components inside region (x0, y0, x1, y1)."""
    H, W = grey.shape[:2]
    x0, y0 = int(round(max(0, region[0]))), int(round(max(0, region[1])))
    x1, y1 = int(round(min(W, region[2]))), int(round(min(H, region[3])))
    crop = grey[y0:y1, x0:x1]
    if crop.size == 0 or int(crop.max()) - int(crop.min()) < 40:
        return []
    _, bw = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out = []
    for mask in (bw, 255 - bw):
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            bx, by, w, h, area = (int(v) for v in stats[i])
            if bx == 0 or by == 0 or bx + w >= crop.shape[1] or by + h >= crop.shape[0]:
                continue
            if not (0.35 * line_h <= h <= 1.05 * line_h and 0.1 <= w / h <= 1.0 and area >= 0.15 * w * h):
                continue
            out.append((bx + x0, by + y0, bx + w + x0, by + h + y0))
    return out


def score_box(grey, cx, band, line_h, side):
    """The 1-2 digit score box on one side (-1 left, +1 right) of the centre line, or None."""
    y0, y1 = band
    near, far = 0.5 * line_h, 3.2 * line_h
    if side < 0:
        region = (cx - far, y0 - 0.4 * line_h, cx - near, y1 + 0.4 * line_h)
    else:
        region = (cx + near, y0 - 0.4 * line_h, cx + far, y1 + 0.4 * line_h)
    glyphs = sorted(find_glyphs(grey, region, line_h), key=lambda b: abs((b[0] + b[2]) / 2 - cx))
    if not glyphs:
        return None
    first = glyphs[0]
    inner = first[2] if side < 0 else first[0]
    if abs(inner - cx) > 2.2 * line_h:  # too far out: that's the team name, not a score
        return None
    group = [first]
    mid = (first[1] + first[3]) / 2
    for b in glyphs[1:]:
        last = group[-1]
        gap = (last[0] - b[2]) if side < 0 else (b[0] - last[2])
        if 0 <= gap < 0.35 * line_h and abs((b[1] + b[3]) / 2 - mid) < 0.25 * line_h:
            group.append(b)
            if len(group) == 2:
                break
    return (min(b[0] for b in group), min(b[1] for b in group), max(b[2] for b in group), max(b[3] for b in group))
