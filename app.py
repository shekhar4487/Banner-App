import io
import textwrap
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


# ================== CONFIG ==================

ASSETS_DIR = Path("assets")
LOGO_PATH = ASSETS_DIR / "Bz_Logo.jpeg"          # logo file
BACKGROUND_PATH = ASSETS_DIR / "background.jpg"  # optional background

MAX_PRODUCTS_PER_BANNER = 12
FOOTER_TEXT = "To order Now Call 7876400500"


# ================== GENERIC HELPERS ==================

def safe_open_image_from_url(url, size):
    """
    Download image from URL and resize; return PIL Image.
    On error, returns a gray placeholder.
    """
    w, h = size
    placeholder = Image.new("RGBA", size, (240, 240, 240, 255))
    if not url or str(url).strip() == "":
        return placeholder

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
        img.thumbnail(size, Image.LANCZOS)

        canvas = Image.new("RGBA", size, (0, 0, 0, 0))
        x = (w - img.width) // 2
        y = (h - img.height) // 2
        canvas.paste(img, (x, y), img)
        return canvas
    except Exception:
        return placeholder


def get_font(size: int, bold: bool = False):
    """Load a font; fall back to default if TTF not available."""
    try:
        return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def chunk_list(lst, n):
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont):
    """
    Get text size that works on both old and new Pillow versions.
    Streamlit Cloud uses a newer Pillow where draw.textsize() is removed,
    so we prefer draw.textbbox() and fall back to textsize().
    """
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return w, h
    except Exception:
        return draw.textsize(text, font=font)


# ================== DATA MAPPING ==================

def normalize_header(h: str) -> str:
    return str(h).lower().replace(" ", "").replace("_", "")


def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map user's columns to a standard schema:
    ProductName, MRP, SellingPrice, ImageURL, BuyLink

    Expected headers (case/spacing isn’t strict):
      Product Name, SP, MRP, Image Link, Product Link
    """
    col_map = {
        "productname": ["productname", "product name"],
        "mrp": ["mrp"],
        "sellingprice": ["sp", "sellingprice", "selling price"],
        "imageurl": ["imagelink", "image link", "imageurl", "image url"],
        "buylink": ["productlink", "product link", "buylink", "buy link"],
    }

    norm_cols = {normalize_header(c): c for c in df.columns}

    def find_col(key):
        for alias in col_map[key]:
            n = normalize_header(alias)
            if n in norm_cols:
                return norm_cols[n]
        return None

    mapped = {
        "ProductName": find_col("productname"),
        "MRP": find_col("mrp"),
        "SellingPrice": find_col("sellingprice"),
        "ImageURL": find_col("imageurl"),
        "BuyLink": find_col("buylink"),
    }

    missing = [k for k, v in mapped.items() if v is None]
    if missing:
        raise ValueError(
            "Missing required columns in sheet: "
            + ", ".join(missing)
            + ". Expected something like: "
            "Product Name, SP, MRP, Image Link, Product Link"
        )

    return pd.DataFrame(
        {
            "ProductName": df[mapped["ProductName"]].astype(str),
            "MRP": df[mapped["MRP"]],
            "SellingPrice": df[mapped["SellingPrice"]],
            "ImageURL": df[mapped["ImageURL"]].astype(str),
            "BuyLink": df[mapped["BuyLink"]].astype(str),
        }
    )


# ================== DRAWING FUNCTIONS ==================

def draw_centered(draw, text, y, W, font, color):
    tw, th = get_text_size(draw, text, font)
    draw.text(((W - tw) // 2, y), text, font=font, fill=color)


def calculate_savings(mrp, sp):
    try:
        m = float(mrp)
        s = float(sp)
        if m > s:
            return round((m - s) * 100 / m)
    except Exception:
        pass
    return None


def compute_grid(n: int):
    """Decide optimal grid for n products (max 12)."""
    if n <= 5:
        return 1, n  # 1 row
    elif n <= 8:
        return 2, 4  # 2x4
    else:
        return 2, 6  # 2x6 (up to 12)


def create_banner_image(products, heading, size, logo_img=None, bg_img=None):
    """
    Create one banner image (PIL.Image) for up to 12 products.
    Layout:
      - Logo at top-left
      - Heading centered
      - Products in circular frames (multi-row grid)
      - Footer green bar with phone number
    """
    if size == "1440x720":
        W, H = 1440, 720
    else:
        W, H = 960, 720

    # Background
    if bg_img is not None:
        bg = bg_img.resize((W, H), Image.LANCZOS).convert("RGBA")
    else:
        bg = Image.new("RGBA", (W, H), (255, 255, 255, 255))

    draw = ImageDraw.Draw(bg)

    header_h = 110
    footer_h = 70
    padding_x = 40

    # Logo
    if logo_img is not None:
        lh = 90
        ratio = lh / logo_img.height
        lw = int(logo_img.width * ratio)
        logo_resized = logo_img.resize((lw, lh), Image.LANCZOS)
        bg.paste(logo_resized, (padding_x, 20), logo_resized)

    # Heading
    heading_font = get_font(42, bold=True)
    draw_centered(draw, heading, 50, W, heading_font, (11, 101, 0))

    # Grid area
    n = len(products)
    if n == 0:
        return bg

    rows, cols = compute_grid(n)

    grid_top = 150
    grid_bottom = H - footer_h - 20
    grid_h = grid_bottom - grid_top
    grid_w = W - 2 * padding_x

    row_h = grid_h / rows
    col_w = grid_w / cols

    circle_diam = int(min(row_h * 0.6, col_w * 0.7))

    name_font = get_font(18, bold=True)
    price_font = get_font(20, bold=True)
    mrp_font = get_font(14, bold=False)
    save_font = get_font(14, bold=True)

    for idx, p in enumerate(products):
        r = idx // cols
        c = idx % cols

        cx = int(padding_x + c * col_w + col_w / 2)
        cy = int(grid_top + r * row_h + row_h / 2 - 20)

        # Product image (circle)
        img = safe_open_image_from_url(p["ImageURL"], (circle_diam, circle_diam))

        mask = Image.new("L", (circle_diam, circle_diam), 0)
        md = ImageDraw.Draw(mask)
        md.ellipse((0, 0, circle_diam, circle_diam), fill=255)

        canvas = Image.new("RGBA", (circle_diam, circle_diam), (0, 0, 0, 0))
        canvas.paste(img, (0, 0), img)

        bg.paste(canvas, (cx - circle_diam // 2, cy - circle_diam // 2), mask)

        ty = cy + circle_diam // 2 + 5

        # Product name (max 2 lines)
        for line in textwrap.wrap(str(p["ProductName"]), 18)[:2]:
            tw, th = get_text_size(draw, line, name_font)
            draw.text((cx - tw // 2, ty), line, font=name_font, fill=(0, 0, 0))
            ty += th + 2

        # MRP
        if str(p["MRP"]).strip():
            mrp_text = f"MRP: ₹{p['MRP']}"
            mw, mh = get_text_size(draw, mrp_text, mrp_font)
            draw.text((cx - mw // 2, ty), mrp_text, font=mrp_font, fill=(120, 120, 120))
            ty += mh + 2

        # Selling price
        if str(p["SellingPrice"]).strip():
            sp_text = f"₹{p['SellingPrice']}"
            sw, sh = get_text_size(draw, sp_text, price_font)
            draw.text((cx - sw // 2, ty), sp_text, font=price_font, fill=(11, 101, 0))
            ty += sh

        # Savings badge
        save = calculate_savings(p["MRP"], p["SellingPrice"])
        if save:
            txt = f"Save {save}%"
            tw, th = get_text_size(draw, txt, save_font)
            bw, bh = tw + 16, th + 6
            bx = cx - bw // 2
            by = ty + 4
            draw.rounded_rectangle(
                (bx, by, bx + bw, by + bh),
                radius=bh // 2,
                fill=(11, 101, 0),
            )
            draw.text((cx - tw // 2, by + 3), txt, font=save_font, fill="white")

    # Footer
    footer_y = H - footer_h
    draw.rectangle((0, footer_y, W, H), fill=(11, 101, 0))
    footer_font = get_font(26, bold=True)
    draw_centered(draw, FOOTER_TEXT, footer_y + 20, W, footer_font, "white")

    return bg


# ================== STREAMLIT APP ==================

st.set_page_config(page_title="Behtar Zindagi Banner Generator", layout="wide")

st.title("Product Banner Generator (up to 12 products per banner)")
st.write(
    "Upload your Excel/CSV and generate **PNG banners** "
    "(1440×720 or 960×720) with logo, centred heading, up to 12 products, "
    "and footer text: **To order Now Call 7876400500**."
)

uploaded_file = st.file_uploader("Upload CSV / XLSX", type=["csv", "xlsx", "xls"])

c1, c2, c3 = st.columns(3)
with c1:
    banner_size = st.selectbox("Banner size", ["1440x720", "960x720"])
with c2:
    heading_text = st.text_input("Banner heading", "Behtar Zindagi Offers")
with c3:
    use_bg = st.checkbox("Use background image (assets/background.jpg)", value=False)

if st.button("Generate banners"):
    if uploaded_file is None:
        st.error("Please upload a file first.")
        st.stop()

    # Read file
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # Map columns
    try:
        df = map_columns(df_raw)
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.success(f"Loaded {len(df)} products.")

    # Logo
    logo_img = None
    if LOGO_PATH.exists():
        try:
            logo_img = Image.open(LOGO_PATH).convert("RGBA")
        except Exception:
            st.warning("Could not load logo image; continuing without logo.")
    else:
        st.warning(f"Logo file not found at {LOGO_PATH}; continuing without logo.")

    # Background
    bg_img = None
    if use_bg and BACKGROUND_PATH.exists():
        try:
            bg_img = Image.open(BACKGROUND_PATH).convert("RGBA")
        except Exception:
            st.warning("Could not load background image; using plain white.")

    # Build banners
    banners = []
    for chunk in chunk_list(df.to_dict("records"), MAX_PRODUCTS_PER_BANNER):
        img = create_banner_image(chunk, heading_text, banner_size, logo_img, bg_img)
        banners.append(img)

    st.write(f"Generated **{len(banners)}** banner(s). Scroll down to preview & download.")

    for idx, img in enumerate(banners, start=1):
        st.subheader(f"Banner {idx}")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        st.image(img, use_column_width=True)
        st.download_button(
            label=f"Download Banner {idx} (PNG)",
            data=buf.getvalue(),
            file_name=f"banner_{idx}.png",
            mime="image/png",
            key=f"banner_dl_{idx}",
        )
