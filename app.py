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


# =========================
# HELPERS
# =========================
def page_to_index(page_num: int) -> int:
    return page_num - 1 if PAGES_ARE_1_BASED else page_num


def parse_pages(val) -> list[int]:
    """
    Accepts:
      - "2,3,4"
      - "2-4" (optional)
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
    # de-duplicate keep order
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


def build_all_unique(df: pd.DataFrame, col: str) -> list[str]:
    s = set()
    for cell in df[col].tolist():
        for t in split_tags(cell):
            s.add(t)
    return sorted(s)


def sort_qid_key(qid: str):
    qid = safe_str(qid).strip()
    m = re.match(r"^[Qq](\d+)", qid)
    return int(m.group(1)) if m else 999999


def try_dataframe_select(df_show: pd.DataFrame) -> int | None:
    """
    Try click-to-select row (newer Streamlit).
    Returns selected row index, or None if not supported.
    """
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


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Physics Bot (QP → MS)", layout="wide")
st.title("📘 Physics Past Paper Bot (QP → MS → Explain)")

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

mode = st.radio(
    "Chế độ tìm kiếm",
    ["Chọn theo paper_code → qid", "Tìm theo Topic (toàn DB)", "Tìm theo Keyword/Topic (toàn DB)"],
    horizontal=True
)

selected_row = None
paper_code = None

# -------------------------
# MODE 1: paper_code -> qid
# -------------------------
if mode == "Chọn theo paper_code → qid":
    paper_codes = sorted(df["paper_code"].unique().tolist())
    paper_code = st.selectbox("Chọn mã đề (paper_code)", paper_codes, index=0)

    df_paper = df[df["paper_code"] == paper_code].copy()
    df_paper = df_paper.sort_values(by="qid", key=lambda s: s.map(sort_qid_key))

    qid_list = df_paper["qid"].tolist()
    qid = st.selectbox("Chọn câu hỏi (Q1, Q2, ...)", qid_list, index=0)

    selected_row = df_paper[df_paper["qid"] == qid].iloc[0]

# -------------------------
# MODE 2: Topic across DB
# -------------------------
elif mode == "Tìm theo Topic (toàn DB)":
    all_topics = build_all_unique(df, "topics")
    if not all_topics:
        st.warning("Database chưa có topics trong mapping.")
        st.stop()

    chosen_topic = st.selectbox("Chọn topic", all_topics)

    def has_topic(cell) -> bool:
        return chosen_topic in split_tags(cell)

    df_result = df[df["topics"].apply(has_topic)].copy()
    if df_result.empty:
        st.warning("Không có câu hỏi nào thuộc topic này.")
        st.stop()

    st.markdown("### Kết quả theo Topic (toàn DB) — click 1 dòng để mở bên dưới")
    limit_on = st.checkbox("Chỉ hiển thị 200 dòng đầu (đỡ lag)", value=True)
    df_show = df_result.head(200) if (limit_on and len(df_result) > 200) else df_result

    show_cols = ["paper_code", "qid", "marks", "keywords", "topics", "qp_pages", "ms_pages"]
    df_show = df_show[show_cols].reset_index(drop=True)

    sel_idx = try_dataframe_select(df_show)
    if sel_idx is None:
        df_show["pick_label"] = df_show["paper_code"].astype(str) + " — " + df_show["qid"].astype(str)
        pick = st.selectbox("Chọn 1 câu để mở QP/MS", df_show["pick_label"].tolist(), index=0)
        selected_row = df_show[df_show["pick_label"] == pick].iloc[0]
    else:
        selected_row = df_show.iloc[int(sel_idx)]

    paper_code = safe_str(selected_row["paper_code"]).strip()

# -------------------------
# MODE 3: One box filters (tags) across DB
# -------------------------
else:
    all_keywords = build_all_unique(df, "keywords")
    all_topics = build_all_unique(df, "topics")

    all_tags = sorted(set(all_keywords) | set(all_topics))
    if not all_tags:
        st.warning("Database chưa có keywords/topics trong mapping.")
        st.stop()

    st.markdown("### Tick tags (keywords + topics) trong 1 ô")
    chosen_tags = st.multiselect("Chọn tags", all_tags, default=[])

    match_mode = st.radio("Match mode (dùng chung)", ["Match ANY (OR)", "Match ALL (AND)"], horizontal=True)

    if not chosen_tags:
        st.info("Hãy tick ít nhất 1 tag.")
        st.stop()

    # Determine which chosen tags are keywords / topics (if overlap, it belongs to both)
    chosen_kw = set([t for t in chosen_tags if t in set(all_keywords)])
    chosen_tp = set([t for t in chosen_tags if t in set(all_topics)])

    # If user selected tag that appears in neither list (shouldn't happen), ignore it.
    if not chosen_kw and not chosen_tp:
        st.info("Các tag bạn tick không khớp keyword/topic trong mapping.")
        st.stop()

    def match_cell(cell, chosen_set: set[str]) -> bool:
        if not chosen_set:
            return True
        cell_set = set(split_tags(cell))
        if match_mode.startswith("Match ALL"):
            return chosen_set.issubset(cell_set)
        return len(chosen_set.intersection(cell_set)) > 0

    df_result = df.copy()
    df_result = df_result[df_result["keywords"].apply(lambda x: match_cell(x, chosen_kw))]
    df_result = df_result[df_result["topics"].apply(lambda x: match_cell(x, chosen_tp))]

    if df_result.empty:
        st.warning("Không có câu hỏi nào khớp tags bạn chọn.")
        st.stop()

    st.markdown("### Kết quả — click 1 dòng để mở QP/MS bên dưới")
    limit_on = st.checkbox("Chỉ hiển thị 200 dòng đầu (đỡ lag)", value=True)
    df_show = df_result.head(200) if (limit_on and len(df_result) > 200) else df_result

    show_cols = ["paper_code", "qid", "marks", "keywords", "topics", "qp_pages", "ms_pages"]
    df_show = df_show[show_cols].reset_index(drop=True)

    sel_idx = try_dataframe_select(df_show)
    if sel_idx is None:
        df_show["pick_label"] = df_show["paper_code"].astype(str) + " — " + df_show["qid"].astype(str)
        pick = st.selectbox("Chọn 1 câu để mở QP/MS", df_show["pick_label"].tolist(), index=0)
        selected_row = df_show[df_show["pick_label"] == pick].iloc[0]
    else:
        selected_row = df_show.iloc[int(sel_idx)]

    paper_code = safe_str(selected_row["paper_code"]).strip()

# If paper_code not set (mode 1)
if paper_code is None:
    paper_code = safe_str(selected_row["paper_code"]).strip()

# Build pdf paths
qp_pdf = pdf_path_from_code(paper_code)
ms_code = qp_to_ms_code(paper_code)
ms_pdf = pdf_path_from_code(ms_code)

if not os.path.exists(qp_pdf):
    st.error(f"Không thấy file QP: {os.path.basename(qp_pdf)} trong {DB_DIR}")
    st.stop()

if not os.path.exists(ms_pdf):
    st.error(f"Không thấy file MS: {os.path.basename(ms_pdf)} trong {DB_DIR}")
    st.stop()

# Read row fields
qid = safe_str(selected_row["qid"]).strip()
qp_pages = parse_pages(selected_row["qp_pages"])
ms_pages = parse_pages(selected_row["ms_pages"])
marks = safe_str(selected_row["marks"])
keywords = safe_str(selected_row["keywords"])
topics = safe_str(selected_row["topics"])

st.caption(f"**{paper_code} — {qid}** | Marks: {marks} | Keywords: {keywords} | Topics: {topics}")

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
