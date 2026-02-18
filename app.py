import os
import re
import pandas as pd
import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline

# =========================
# CONFIG
# =========================
DB_DIR = "physics_db"
MAPPING_FILE = os.path.join(DB_DIR, "mapping.xlsx")
PAGES_ARE_1_BASED = True
MODEL_NAME = "google/flan-t5-small"

# As you specified:
# m25 = Feb–March 2025
# s25 = May–June 2025
# w25 = Oct–Nov 2025
SERIES_MAP = {
    "m": "Feb–March",
    "s": "May–June",
    "w": "Oct–Nov",
}
SERIES_ORDER = {"m": 0, "s": 1, "w": 2}


# =========================
# HELPERS
# =========================
def page_to_index(page_num: int) -> int:
    return page_num - 1 if PAGES_ARE_1_BASED else page_num


def parse_pages(val) -> list[int]:
    """
    Accepts:
      - "2,3,4"
      - "2-4"
      - 2
    Returns list of pages in mapping numbering (1-based).
    """
    if val is None:
        return []
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return []

    pages: list[int] = []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = a.strip(), b.strip()
            if a.isdigit() and b.isdigit():
                pages.extend(list(range(int(a), int(b) + 1)))
        else:
            if part.isdigit():
                pages.append(int(part))

    seen, out = set(), []
    for p in pages:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def split_tags(s) -> list[str]:
    """Split topics/keywords by ; or ,"""
    if s is None:
        return []
    s = str(s).strip()
    if not s or s.lower() == "nan":
        return []
    parts = re.split(r"[;,]", s)
    return [p.strip() for p in parts if p.strip()]


def qp_to_ms_code(qp_code: str) -> str:
    return qp_code.replace("_qp_", "_ms_")


def pdf_path_from_code(code: str) -> str:
    return os.path.join(DB_DIR, f"{code}.pdf")


def render_pdf_page(pdf_path: str, page_num_in_map: int, zoom: float = 2.0) -> bytes:
    doc = fitz.open(pdf_path)
    idx = page_to_index(int(page_num_in_map))
    if idx < 0 or idx >= doc.page_count:
        raise ValueError(f"Page {page_num_in_map} out of range for {os.path.basename(pdf_path)}")
    page = doc.load_page(idx)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    return pix.tobytes("png")


def extract_text_from_pdf_page(pdf_path: str, page_num_in_map: int) -> str:
    doc = fitz.open(pdf_path)
    idx = page_to_index(int(page_num_in_map))
    if idx < 0 or idx >= doc.page_count:
        return ""
    return doc.load_page(idx).get_text("text").strip()


@st.cache_resource
def get_explainer():
    return pipeline("text2text-generation", model=MODEL_NAME)


def make_explanation(ms_text: str, qp_text: str, keywords: str, topics: str, student_note: str) -> str:
    explainer = get_explainer()

    qp_text = (qp_text or "")[:1500]
    ms_text = (ms_text or "")[:1500]
    student_note = (student_note or "")[:300]
    keywords = (keywords or "")[:250]
    topics = (topics or "")[:250]

    prompt = f"""
You are a Physics exam tutor.

STRICT RULES:
- Do NOT change any mathematical symbols, units, or numbers.
- Do NOT invent equations or symbols.
- Any equations/symbols/numbers MUST come directly from the mark scheme text.
- You may only explain meaning in words.

Output format:
A) Mark scheme points (bullet points; copy wording as closely as possible)
B) Simple explanation in words (NO equations; simple English)
C) Common mistake (one sentence)

Keywords: {keywords}
Topics: {topics}

Question text:
{qp_text}

Mark scheme text:
{ms_text}

Student note (optional):
{student_note}
""".strip()

    return explainer(prompt, max_new_tokens=260, do_sample=False)[0]["generated_text"]


def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    s = str(x)
    return "" if s.lower() == "nan" else s


def sort_qid_key(qid: str):
    qid = safe_str(qid).strip()
    m = re.match(r"^[Qq](\d+)", qid)
    return int(m.group(1)) if m else 999999


def try_dataframe_select(df_show: pd.DataFrame) -> int | None:
    """Try click-to-select row (newer Streamlit)."""
    try:
        event = st.dataframe(
            df_show,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )
        if event and "selection" in event and event["selection"]["rows"]:
            return int(event["selection"]["rows"][0])
        return 0
    except Exception:
        return None


# ---- Parse paper_code metadata ----
# Example: 9702_m25_qp_22
_paper_re = re.compile(r"^(\d{4})_(?P<series>[msw])(?P<yy>\d{2})_qp_(?P<pv>\d{2})$", re.IGNORECASE)

def parse_paper_meta(paper_code: str):
    s = safe_str(paper_code).strip()
    m = _paper_re.match(s)
    if not m:
        return None
    series = m.group("series").lower()  # m/s/w
    yy = int(m.group("yy"))
    pv = m.group("pv")  # "22", "41", ...
    paper_num = int(pv[0])    # 2 or 4
    variant = int(pv[1])      # 1/2/3
    year = 2000 + yy
    return {"series": series, "year": year, "paper": paper_num, "variant": variant}


def add_meta_columns(df: pd.DataFrame) -> pd.DataFrame:
    metas = df["paper_code"].apply(parse_paper_meta)
    df2 = df.copy()
    df2["meta_ok"] = metas.notna()
    df2["paper"] = metas.apply(lambda x: x["paper"] if isinstance(x, dict) else None)
    df2["year"] = metas.apply(lambda x: x["year"] if isinstance(x, dict) else None)
    df2["series"] = metas.apply(lambda x: x["series"] if isinstance(x, dict) else None)
    df2["series_name"] = df2["series"].apply(lambda c: SERIES_MAP.get(c, "") if isinstance(c, str) else "")
    return df2


def build_unique_tags(df: pd.DataFrame) -> list[str]:
    """Union of keyword + topic tags"""
    s = set()
    for cell in df["keywords"].tolist():
        for t in split_tags(cell):
            s.add(t)
    for cell in df["topics"].tolist():
        for t in split_tags(cell):
            s.add(t)
    return sorted(s)


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(
    page_title="TAT Physics Past Paper Bot",
    layout="wide"
)

st.markdown(
    """
    <h1 style='text-align: center;'>
    📘 Physics Past Paper Bot
    </h1>
    <h4 style='text-align: center; color: grey;'>
    Designed & Developed by TAT
    </h4>
    """,
    unsafe_allow_html=True
)

st.markdown("---")


if not os.path.exists(DB_DIR):
    st.error(f"Không thấy folder `{DB_DIR}`. Hãy đặt `physics_db` cùng cấp với app.py.")
    st.stop()

if not os.path.exists(MAPPING_FILE):
    st.error(f"Không thấy file `{MAPPING_FILE}`.")
    st.stop()

try:
    df = pd.read_excel(MAPPING_FILE)
except Exception as e:
    st.error(f"Không đọc được mapping.xlsx: {e}")
    st.stop()

df.columns = [c.strip().lower() for c in df.columns]
required_cols = {"paper_code", "qid", "qp_pages", "ms_pages", "marks", "keywords", "topics"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"mapping.xlsx thiếu cột: {', '.join(sorted(missing))}")
    st.stop()

df["paper_code"] = df["paper_code"].astype(str).str.strip()
df["qid"] = df["qid"].astype(str).str.strip()

df = add_meta_columns(df)
df = df[df["meta_ok"] == True].copy()

if df.empty:
    st.error("Không có paper_code parse được theo format: 9702_m25_qp_22. Hãy kiểm tra paper_code trong mapping.xlsx.")
    st.stop()

# ONLY 2 options
main_mode = st.radio("Options", ["Chọn theo paper_code", "Tìm theo keyword/topic"], horizontal=True)

paper_choice = st.radio("Chọn Paper", ["Paper 2 (AS)", "Paper 4 (A Level)"], horizontal=True)
paper_num = 2 if paper_choice.startswith("Paper 2") else 4

df_p = df[df["paper"] == paper_num].copy()
if df_p.empty:
    st.warning(f"Không có dữ liệu cho {paper_choice}.")
    st.stop()

selected_row = None
paper_code = None

# =========================
# OPTION 1: Choose by paper_code with Year -> Series(code) -> paper_code
# =========================
if main_mode == "Chọn theo paper_code":
    years = sorted(df_p["year"].dropna().unique().tolist())
    year = st.selectbox("Chọn Year", years, index=len(years) - 1)

    df_y = df_p[df_p["year"] == year].copy()

    series_codes = sorted(
        df_y["series"].dropna().unique().tolist(),
        key=lambda x: SERIES_ORDER.get(x, 99)
    )

    # ✅ FIX: select code directly; display name via format_func
    series_code = st.selectbox(
        "Chọn Series",
        series_codes,
        index=0,
        format_func=lambda c: SERIES_MAP.get(c, c)
    )

    df_s = df_y[df_y["series"] == series_code].copy()

    paper_codes = sorted(df_s["paper_code"].unique().tolist())
    if not paper_codes:
        st.warning("Không có paper_code phù hợp với bộ lọc này.")
        st.stop()

    paper_code = st.selectbox("Chọn mã đề (paper_code)", paper_codes, index=0)

    df_paper = df_s[df_s["paper_code"] == paper_code].copy()
    df_paper = df_paper.sort_values(by="qid", key=lambda s: s.map(sort_qid_key))

    qid_list = df_paper["qid"].tolist()
    qid = st.selectbox("Chọn câu hỏi (Q1, Q2, ...)", qid_list, index=0)

    selected_row = df_paper[df_paper["qid"] == qid].iloc[0]

# =========================
# OPTION 2: Search keyword/topic within selected paper (2 or 4)
# =========================
else:
    st.markdown("### Tick tags (keywords + topics)")

    tags = build_unique_tags(df_p)
    if not tags:
        st.warning("Không có tags trong dataset (keywords/topics trống).")
        st.stop()

    chosen_tags = st.multiselect("Chọn tags", tags, default=[])
    match_mode = st.radio("Match mode (dùng chung)", ["Match ANY (OR)", "Match ALL (AND)"], horizontal=True)

    if not chosen_tags:
        st.info("Hãy tick ít nhất 1 tag.")
        st.stop()

    chosen_set = set(chosen_tags)

    def match_in_cell(cell) -> bool:
        cell_set = set(split_tags(cell))
        if match_mode.startswith("Match ALL"):
            return chosen_set.issubset(cell_set)
        return len(chosen_set.intersection(cell_set)) > 0

    df_result = df_p[
        df_p["keywords"].apply(match_in_cell) | df_p["topics"].apply(match_in_cell)
    ].copy()

    if df_result.empty:
        st.warning("Không có câu hỏi nào khớp tags bạn chọn.")
        st.stop()

    st.markdown("### Kết quả — click 1 dòng để mở QP/MS bên dưới")
    limit_on = st.checkbox("Chỉ hiển thị 200 dòng đầu (đỡ lag)", value=True)
    df_show = df_result.head(200) if (limit_on and len(df_result) > 200) else df_result

    show_cols = ["paper_code", "year", "series_name", "qid", "marks", "keywords", "topics", "qp_pages", "ms_pages"]
    df_show = df_show[show_cols].reset_index(drop=True)

    sel_idx = try_dataframe_select(df_show)
    if sel_idx is None:
        # fallback
        df_show["pick_label"] = df_show["paper_code"].astype(str) + " — " + df_show["qid"].astype(str)
        pick = st.selectbox("Chọn 1 câu để mở QP/MS", df_show["pick_label"].tolist(), index=0)
        selected_row = df_show[df_show["pick_label"] == pick].iloc[0]
    else:
        selected_row = df_show.iloc[int(sel_idx)]

    paper_code = safe_str(selected_row["paper_code"]).strip()

# =========================
# Render QP/MS
# =========================
if selected_row is None:
    st.stop()

paper_code = paper_code or safe_str(selected_row["paper_code"]).strip()

qp_pdf = pdf_path_from_code(paper_code)
ms_code = qp_to_ms_code(paper_code)
ms_pdf = pdf_path_from_code(ms_code)

if not os.path.exists(qp_pdf):
    st.error(f"Không thấy file QP: {os.path.basename(qp_pdf)} trong {DB_DIR}")
    st.stop()

if not os.path.exists(ms_pdf):
    st.error(f"Không thấy file MS: {os.path.basename(ms_pdf)} trong {DB_DIR}")
    st.stop()

qid = safe_str(selected_row["qid"]).strip()
qp_pages = parse_pages(selected_row["qp_pages"])
ms_pages = parse_pages(selected_row["ms_pages"])
marks = safe_str(selected_row["marks"])
keywords = safe_str(selected_row["keywords"])
topics = safe_str(selected_row["topics"])
year = safe_str(selected_row.get("year", ""))
series_name = safe_str(selected_row.get("series_name", ""))

st.caption(f"**{paper_code} — {qid}** | {series_name} {year} | Marks: {marks} | Keywords: {keywords} | Topics: {topics}")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Question Paper (QP)")
    if not qp_pages:
        st.warning("qp_pages trống hoặc sai format. Dùng dạng: 2,3,4")
        qp_text = ""
    else:
        for p in qp_pages:
            st.image(render_pdf_page(qp_pdf, p), caption=f"{paper_code} — {qid} — QP page {p}")
        qp_text = "\n\n".join(extract_text_from_pdf_page(qp_pdf, p) for p in qp_pages)

    with st.expander("QP text (optional)"):
        st.text(qp_text if qp_text else "(Không trích được text rõ.)")

with col2:
    st.subheader("✅ Mark Scheme (MS)")
    if not ms_pages:
        st.warning("ms_pages trống hoặc sai format. Dùng dạng: 2,3,4")
        ms_text = ""
    else:
        for p in ms_pages:
            st.image(render_pdf_page(ms_pdf, p), caption=f"{ms_code} — {qid} — MS page {p}")
        ms_text = "\n\n".join(extract_text_from_pdf_page(ms_pdf, p) for p in ms_pages)

    with st.expander("MS text (source)"):
        st.text(ms_text if ms_text else "(Không trích được text rõ.)")

st.divider()
st.subheader("🧠 Explain mark scheme for students (free CPU model)")

student_note = st.text_area("Ghi chú thêm (tuỳ chọn)", height=90, placeholder="Ví dụ: Explain in very simple English...")

if st.button("Generate explanation"):
    if not ms_text:
        st.error("Không có MS text để giải thích. Nếu MS là scan ảnh thì cần OCR.")
        st.stop()
    with st.spinner("Generating... (may be slower on CPU)"):
        ans = make_explanation(ms_text, qp_text, keywords, topics, student_note)
    st.markdown(ans)
