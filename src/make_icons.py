"""휴대폰 홈 화면용 앱 아이콘 만들기 (한 번만 실행하면 됨): python src/make_icons.py"""
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "docs"
BG, BAR, TOP = (36, 72, 199), (255, 255, 255), (79, 211, 160)


def icon(size: int, safe: float) -> Image.Image:
    """safe: 그림을 그릴 가운데 영역 비율 (안드로이드 마스크 아이콘은 가장자리가 잘림)"""
    s = size * 4                                   # 크게 그린 뒤 줄여서 테두리를 부드럽게
    im = Image.new("RGB", (s, s), BG)
    d = ImageDraw.Draw(im)
    box = s * safe
    x0, y1 = (s - box) / 2, (s + box) / 2
    w, gap = box * 0.2, box * 0.1
    left = x0 + (box - (3 * w + 2 * gap)) / 2
    for i, h in enumerate((0.38, 0.6, 0.86)):
        bx = left + i * (w + gap)
        d.rounded_rectangle([bx, y1 - box * h, bx + w, y1], radius=w * 0.28, fill=TOP if i == 2 else BAR)
    return im.resize((size, size), Image.LANCZOS)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    icon(192, 0.62).save(OUT / "icon-192.png")
    icon(512, 0.62).save(OUT / "icon-512.png")
    icon(512, 0.5).save(OUT / "icon-maskable-512.png")
    icon(180, 0.62).save(OUT / "apple-touch-icon.png")
    print("아이콘 저장:", OUT)
