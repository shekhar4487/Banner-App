import io
import math
import textwrap
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


# ================== CONFIG ==================

ASSETS_DIR = Path("assets")
LOGO_PATH = ASSETS_DIR / "Bz_Logo.jpeg"
BACKGROUND_PATH = ASSETS_DIR / "background.jpg"   # optional

MAX_PRODUCTS_PER_BANNER = 12
FOOTER_TEXT = "To order Now Call 7876400500"


# ================== IMAGE HELPERS ==================

def safe_open_image_from_url(url, size):
    w, h = size
    placeholder = Image.new("RGBA", size, (240, 240, 240, 255))

    if not url:
        return placeholder

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
        img.thumbnail(size, Image.LANCZOS)

        canvas = Image.new("RGBA", size, (0, 0, 0, 0))
        x = (size[0] - img.width) // 2
        y = (size[1] - img.height) // 2
        canvas.paste(img, (x, y), img)
        return canvas
    except:
        return placeholder


def get_font(size, bold=False):
    try:
        return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", size)
    except:
        return ImageFont.load_default()


def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# ================== COLUMN MAPPING ==================

def normalize(h):
    return str(h).lower().replace(" ", "").replace("_", "")


def map_columns(df):
    col_map = {
        "productname": ["productname", "product name"],
        "mrp": ["mrp"],
        "sellingprice": ["sp", "sellingprice"],
        "imageurl": ["imagelink", "image link", "imageurl"],
        "buylink": ["productlink", "product link"]
    }

    norm_cols = {normalize(c): c for c in df.columns}

    def find(target):
        for alias in col_map[target]:
            if normalize(alias) in norm_cols:
                return norm_cols[normalize(alias)]
        return None

    mapped = {
        "ProductName": find("productname"),
        "MRP": find("mrp"),
        "SellingPrice": find("sellingprice"),
        "ImageURL": find("imageurl"),
        "BuyLink": find("buylink")
    }

    missing = [k for k, v in mapped.items() if v is None]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    return pd.DataFrame({
        "ProductName": df[mapped["ProductName"]].astype(str),
        "MRP": df[mapped["MRP"]],
        "SellingPrice": df[mapped["SellingPrice"]],
        "ImageURL": df[mapped["ImageURL"]].astype(str),
        "BuyLink": df[mapped["BuyLink"]].astype(str)
    })


# ================== DRAWING FUNCTIONS ==================

def draw_centered(draw, text, y, W, font, color):
    tw, _ = draw.textsize(text, font=font)
    draw.text(((W - tw) // 2, y), text, font=font, fill=color)


def calculate_savings(mrp, sp):
    try:
        m = float(mrp); s = float(sp)
        if m > s:
            return round((m - s) * 100 / m)
    except:
        return None


def compute_grid(n):
    """Decide optimal grid for n products (max 12)."""
    if n <= 5:
        return (1, 5 if n == 5 else n)  # 1 row
    elif n <= 8:
        return (2, 4)  # 2 rows × 4 columns
    else:
        return (2, 6)  # max layout for 12 products: 2 rows × 6 columns


def create_banner_image(products, heading, size, logo_img=None, bg_img=None):

    if size == "1440x720":
        W, H = 1440, 720
    else:
        W, H = 960, 720

    # Background
    if bg_img:
        bg = bg_img.resize((W, H), Image.LANCZOS).convert("RGBA")
    else:
        bg = Image.new("RGBA", (W, H), (255, 255, 255, 255))

    draw = ImageDraw.Draw(bg)

    # Header
    header_h = 110
    footer_h = 70
    padding_x = 40

    # Logo
    if logo_img:
        lh = 90
        ratio = lh / logo_img.height
        lw = int(logo_img.width * ratio)
        logo_resized = logo_img.resize((lw, lh))
        bg.paste(logo_resized, (padding_x, 20), logo_resized)

    # Heading centered
    heading_font = get_font(42, bold=True)
    draw_centered(draw, heading, 50, W, heading_font, (11, 101, 0))

    # Grid layout
    n = len(products)
    rows, cols = compute_grid(n)

    grid_top = 150
    grid_bottom = H - footer_h - 20
    grid_h = grid_bottom - grid_top
    grid_w = W - 2 * padding_x

    row_h = grid_h // rows
    col_w = grid_w // cols

    circle_diam = int(min(row_h * 0.6, col_w * 0.7))

    name_font = get_font(18, bold=True)
    price_font = get_font(20, bold=True)
    mrp_font = get_font(14)
    save_font = get_font(14, bold=True)

    for idx, p in enumerate(products):

        r = idx // cols
        c = idx % cols

        cx = padding_x + c * col_w + col_w // 2
        cy = grid_top + r * row_h + row_h // 2 - 20

        # Product image
        img = safe_open_image_from_url(p["ImageURL"], (circle_diam, circle_diam))

        mask = Image.new("L", (circle_diam, circle_diam), 0)
        md = ImageDraw.Draw(mask)
        md.ellipse((0, 0, circle_diam, circle_diam), fill=255)

        canvas = Image.new("RGBA", (circle_diam, circle_diam), (0, 0, 0, 0))
        canvas.paste(img, (0, 0), img)

        bg.paste(canvas, (cx - circle_diam // 2, cy - circle_diam // 2), mask)

        ty = cy + circle_diam // 2 + 5

        # Name
        for line in textwrap.wrap(p["ProductName"], 18)[:2]:
            tw, th = draw.textsize(line, font=name_font)
            draw.text((cx - tw // 2, ty), line, font=name_font, fill=(0, 0, 0))
            ty += th + 2

        # Pricing
        if str(p["MRP"]).strip():
            mrp = f"MRP: ₹{p['MRP']}"
            mw, mh = draw.textsize(mrp, font=mrp_font)
            draw.text((cx - mw // 2, ty), mrp, font=mrp_font, fill=(120, 120, 120))
            ty += mh + 2

        if str(p["SellingPrice"]).strip():
            sp = f"₹{p['SellingPrice']}"
            sw, sh = draw.textsize(sp, font=price_font)
            draw.text((cx - sw // 2, ty), sp, font=price_font, fill=(11, 101, 0))
            ty += sh

        # Savings
        save = calculate_savings(p["MRP"], p["SellingPrice"])
        if save:
            txt = f"Save {save}%"
            tw, th = draw.textsize(txt, font=save_font)
            bw, bh = tw + 16, th + 6
            bx = cx - bw // 2
            by = ty + 4
            draw.rounded_rectangle((bx, by, bx + bw, by + bh),
                                   radius=bh // 2, fill=(11, 101, 0))
            draw.text((cx - tw // 2, by + 3), txt, font=save_font, fill="white")

    # Footer
    footer_y = H - footer_h
    draw.rectangle((0, footer_y, W, H), fill=(11, 101, 0))
    footer_font = get_font(26, bold=True)
    draw_centered(draw, FOOTER_TEXT, footer_y + 20, W, footer_font, "white")

    return bg


# ================== STREAMLIT APP ==================

st.set_page_config(page_title="Behtar Zindagi Banner Generator", layout="wide")

st.title("Product Banner Generator (Supports up to 12 Products)")
st.write("Upload Excel/CSV → Generate PNG banners → Download directly.")

uploaded_file = st.file_uploader("Upload CSV / XLSX", type=["csv", "xlsx", "xls"])

col1, col2, col3 = st.columns(3)
with col1:
    banner_size = st.selectbox("Banner size", ["1440x720", "960x720"])
with col2:
    heading_text = st.text_input("Banner heading", "Behtar Zindagi Offers")
with col3:
    use_bg = st.checkbox("Use background image (optional)", False)

if st.button("Generate banners"):

    if not uploaded_file:
        st.error("Please upload a file.")
        st.stop()

    # Read file
    try:
        df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith("csv") else pd.read_excel(uploaded_file)
        df = map_columns(df_raw)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Logo
    logo_img = Image.open(LOGO_PATH).convert("RGBA") if LOGO_PATH.exists() else None

    # Background
    bg_img = Image.open(BACKGROUND_PATH).convert("RGBA") if use_bg and BACKGROUND_PATH.exists() else None

    banners = []

    # chunk into groups of up to 12
    for chunk in chunk_list(df.to_dict("records"), MAX_PRODUCTS_PER_BANNER):
        img = create_banner_image(chunk, heading_text, banner_size, logo_img, bg_img)
        banners.append(img)

    st.success(f"Generated {len(banners)} banner(s). Scroll down to download.")

    for idx, img in enumerate(banners, start=1):
        st.subheader(f"Banner {idx}")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        st.image(img, use_column_width=True)

        st.download_button(
            f"Download Banner {idx} (PNG)",
            data=buf.getvalue(),
            file_name=f"banner_{idx}.png",
            mime="image/png",
            key=f"banner_dl_{idx}"
        )
