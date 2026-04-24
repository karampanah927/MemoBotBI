import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from bot import (
    load_system_index,
    load_icon_references,
    get_display_name,
    get_icon_key,
    make_rag,
    ICONS_BASE_DIR,
)

CYAN_LIGHT = "#00C8DC"
CYAN_MID = "#00B4C8"
CYAN_DARK = "#0097A8"
YELLOW = "#FFD600"
DARK_TEXT = "#0D1B2A"

HERE = Path(__file__).parent


def _img_b64(path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    mime = {"png": "png", "jpg": "jpeg", "jpeg": "jpeg"}.get(
        p.suffix.lower().lstrip("."), "png"
    )
    data = base64.b64encode(p.read_bytes()).decode()
    return f"data:image/{mime};base64,{data}"


SYSTEM_INDEX = load_system_index()
ICON_REFS = load_icon_references()

SYSTEMS: list[dict] = []
for _entry in SYSTEM_INDEX:
    _system_key = _entry["system"]
    _icon_key = get_icon_key(_system_key)

    # WaWi and MeWi use the Memodo logo icon
    if _system_key.lower() in ("wawi", "mewi"):
        _icon_rel = "memodo.png"
    else:
        _icon_rel = ICON_REFS.get(_icon_key)

    _icon_path = (ICONS_BASE_DIR / _icon_rel) if _icon_rel else None

    SYSTEMS.append({
        "entry": _entry,
        "display": get_display_name(_system_key),
        "icon_b64": _img_b64(_icon_path) if _icon_path else None,
    })


LOGO_SRC = (
    _img_b64(HERE / "MemodoImage.png")
    or _img_b64(HERE / "Memodologo.png")
    or _img_b64(HERE / "memodo.png")
)

_logo_tag = (
    f'<img src="{LOGO_SRC}" style="height:44px;width:auto;border-radius:8px;object-fit:contain;" />'
    if LOGO_SRC else
    '<div style="background:#FFD600;border-radius:12px;width:48px;height:48px;min-width:48px;'
    'display:flex;align-items:center;justify-content:center;font-size:26px;'
    'box-shadow:0 2px 8px rgba(255,214,0,0.35);">💡</div>'
)


@st.cache_resource
def get_rag(system_name: str):
    entry = next((s["entry"] for s in SYSTEMS if s["entry"]["system"] == system_name), None)
    if entry is None:
        return None
    rag = make_rag(entry)
    rag.build()
    return rag


def say_hi(system_display: str) -> str:
    is_powerbi = system_display == "Power BI"
    nav_lines = (
        "- 📊 **Dashboard navigation** — which Power BI report & page to open\n"
        "- 🔗 **Power BI deep links** — open the exact report page directly\n"
        if is_powerbi else ""
    )
    return (
        f"👋 Welcome! I'm the **Memodo BI Bot**.\n\n"
        f"Active system: **{system_display}**\n\n"
        "I can help you with:\n"
        "- 📖 **KPI definitions** — formulas, date logic, and business rules\n"
        + nav_lines
        + "\nAsk me about any KPI in this system!"
    )


def render_result(result: dict) -> str:
    parts = []

    answer = result.get("answer", "")
    if answer:
        parts.append(answer)

    kpi_defs = result.get("kpi_definitions") or []
    if kpi_defs:
        parts.append("---\n### KPI Definitions")
        for d in kpi_defs:
            block = [f"#### {d.get('name') or ''}"]
            if d.get("definition"):
                block.append(f"**Definition:** {d['definition']}")
            if d.get("formula"):
                block.append(f"**Formula:** `{d['formula']}`")
            if d.get("assigned_date"):
                block.append(f"**Date assignment:** {d['assigned_date']}")
            notes = d.get("notes") or []
            if notes:
                block.append("**Notes:**")
                for n in notes:
                    block.append(f"- {n}")
            parts.append("\n".join(block))

    best_matches = result.get("best_matches") or []
    if best_matches:
        parts.append("---\n### Dashboard Navigation")
        for i, m in enumerate(best_matches, 1):
            report = m.get("report_name", "")
            page = m.get("page_name", "")
            url = m.get("report_url", "")
            why = m.get("why", "")
            score = m.get("score")

            header = f"**{i}. {report}** → *{page}*"
            if score is not None:
                score_str = f"{score:.3f}" if isinstance(score, float) else str(score)
                header += f"  `score: {score_str}`"

            lines = [header]
            if url and url.lower() not in ("", "no url", "null", "none"):
                lines.append(f"[Open in Power BI ↗]({url})")
            if why:
                lines.append(f"*{why}*")
            parts.append("\n".join(lines))

    note = result.get("note")
    if note:
        parts.append(f"ℹ️ *{note}*")

    missing = result.get("missing_info")
    if missing:
        parts.append(f"❓ **Missing info:** {missing}")

    raw = result.get("raw_output")
    if raw:
        parts.append("**Raw model output:**\n```\n" + raw + "\n```")

    return "\n\n".join(parts).strip()


CUSTOM_CSS = f"""
<style>
.stApp {{
    background: #87CEEB !important;
    min-height: 100vh;
}}

#MainMenu, footer, header {{ visibility: hidden; }}

[data-testid="block-container"] {{
    padding-top: 0.75rem !important;
    padding-bottom: 0.5rem !important;
    max-width: 840px !important;
}}

[data-testid="stChatMessage"] {{
    background: rgba(255, 255, 255, 0.91) !important;
    border-radius: 14px !important;
    margin: 5px 0 !important;
    box-shadow: 0 2px 14px rgba(0, 60, 90, 0.11) !important;
    backdrop-filter: blur(10px);
}}

[data-testid="stChatInputContainer"] {{
    background: rgba(255, 255, 255, 0.96) !important;
    border: 2px solid {YELLOW} !important;
    border-radius: 28px !important;
    padding: 2px 6px !important;
    box-shadow: 0 2px 12px rgba(0, 60, 90, 0.14) !important;
}}

[data-testid="stChatInputContainer"] textarea {{
    background: transparent !important;
    color: {DARK_TEXT} !important;
    font-size: 15px !important;
}}

[data-testid="stChatInputContainer"] button {{ color: {YELLOW} !important; }}

.stSpinner > div {{ border-top-color: {YELLOW} !important; }}

[data-testid="stChatMessage"] a {{
    color: #0077A8 !important;
    font-weight: 600;
    text-decoration: none;
}}

[data-testid="stChatMessage"] a:hover {{ text-decoration: underline; }}

::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: rgba(0,0,0,0.18); border-radius: 3px; }}

[data-testid="stSidebar"] {{
    background: {YELLOW} !important;
    border-right: 1px solid rgba(0,0,0,0.12) !important;
}}

[data-testid="stSidebar"] > div:first-child {{ padding: 0 !important; }}

[data-testid="stSidebar"] [data-testid="stHtml"] {{
    padding: 0 !important;
    margin: 0 !important;
}}

[data-testid="stSidebar"] .stSelectbox {{
    padding: 0 16px 4px !important;
}}

[data-testid="stSidebar"] .stSelectbox label {{
    color: #000000 !important;
    font-size: 11px !important;
    font-weight: 900 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    font-family: 'Segoe UI', Arial, sans-serif !important;
}}

[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div:first-child {{
    background: rgba(255,255,255,0.45) !important;
    border-color: rgba(0,27,42,0.25) !important;
    border-radius: 10px !important;
}}

[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span {{
    color: {DARK_TEXT} !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}}

[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] svg {{
    fill: {DARK_TEXT} !important;
}}
</style>
"""


HEADER_HTML = f"""
<div style="
    background: white;
    border-radius: 18px;
    padding: 14px 22px;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 14px;
    box-shadow: 0 4px 20px rgba(0,60,90,0.13);
    font-family: 'Segoe UI', Arial, sans-serif;
">
    {_logo_tag}
    <div>
        <div style="font-weight:800;font-size:19px;color:{DARK_TEXT};letter-spacing:.2px;line-height:1.2;">
            memodo BI Bot
        </div>
        <div style="font-size:12.5px;color:#555;margin-top:2px;">
            Multi-system KPI assistant · definitions · formulas · navigation
        </div>
    </div>
    <div style="
        margin-left: auto;
        background: {YELLOW};
        border-radius: 10px;
        padding: 6px 14px;
        font-weight: 800;
        font-size: 14px;
        color: {DARK_TEXT};
        letter-spacing: .5px;
        white-space: nowrap;
    ">memodo ◆</div>
</div>
"""


st.set_page_config(page_title="Memodo BI Bot", page_icon="💡", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


with st.sidebar:
    _sidebar_logo = (
        f'<img src="{LOGO_SRC}" style="height:33vh;width:auto;max-width:100%;display:block;'
        f'margin-bottom:12px;object-fit:contain;" />'
        if LOGO_SRC else
        '<div style="font-size:28px;margin-bottom:6px;">💡</div>'
    )

    st.html(
        '<div style="padding:20px 16px 12px;font-family:\'Segoe UI\',Arial,sans-serif;">'
        + _sidebar_logo
        + f'<div style="font-size:20px;font-weight:800;color:{DARK_TEXT};'
        f'letter-spacing:.3px;line-height:1.15;margin-bottom:2px;">BI Bot</div>'
        + f'<div style="font-size:11px;color:{CYAN_DARK};margin-bottom:0;'
        f'letter-spacing:.2px;">Multi-System KPI Assistant</div>'
        + '</div>'
    )

    selected_idx = st.selectbox(
        "Select your system",
        range(len(SYSTEMS)),
        format_func=lambda i: SYSTEMS[i]["display"],
        key="system_selector_idx",
    )

    selected_sys = SYSTEMS[selected_idx]
    active_system_key = selected_sys["entry"]["system"]

    if st.session_state.get("active_system") != active_system_key:
        st.session_state.active_system = active_system_key
        st.session_state.messages = [
            {"role": "assistant", "content": say_hi(selected_sys["display"])}
        ]

    _has_nav = selected_sys["entry"].get("has_urls", False)
    _icon_html = (
        f'<div style="padding:8px 16px 12px;text-align:center;'
        f'border-bottom:1px solid rgba(0,0,0,0.1);">'
    )

    if selected_sys["icon_b64"]:
        _icon_html += (
            f'<img src="{selected_sys["icon_b64"]}" '
            f'style="height:52px;width:auto;object-fit:contain;margin-bottom:6px;" /><br/>'
        )
    else:
        _icon_html += '<div style="font-size:36px;margin-bottom:6px;">📊</div>'

    _icon_html += (
        f'<div style="color:{DARK_TEXT};font-size:13px;font-weight:700;letter-spacing:.3px;">'
        f'{selected_sys["display"]}</div>'
    )

    if _has_nav:
        _icon_html += (
            f'<div style="color:{CYAN_DARK};font-size:10px;margin-top:3px;">'
            f'Power BI navigation enabled</div>'
        )

    _icon_html += '</div>'
    st.html(_icon_html)

    _caps = [
        ("📖", "rgba(255,214,0,0.2)", "KPI Definitions", "Formulas · date logic · rules"),
        ("🧠", "rgba(0,200,220,0.2)", "AI-Powered Answers", "Context-aware RAG"),
    ]

    if _has_nav:
        _caps.insert(1, ("📊", "rgba(0,200,220,0.2)", "Dashboard Navigation", "Find the right report"))
        _caps.insert(2, ("🔗", "rgba(255,214,0,0.2)", "Power BI Deep Links", "Open pages directly"))

    _caps_html = (
        '<div style="padding:14px 16px 10px;font-family:\'Segoe UI\',Arial,sans-serif;">'
        '<div style="font-size:9.5px;font-weight:700;color:rgba(0,27,42,0.5);'
        'letter-spacing:1.1px;text-transform:uppercase;margin-bottom:10px;">Capabilities</div>'
    )

    for _icon, _bg, _title, _sub in _caps:
        _caps_html += (
            '<div style="display:flex;align-items:center;gap:9px;margin-bottom:9px;">'
            f'<div style="width:28px;height:28px;min-width:28px;border-radius:8px;'
            f'background:rgba(255,255,255,0.45);display:flex;align-items:center;justify-content:center;'
            f'font-size:14px;">{_icon}</div>'
            f'<div>'
            f'<div style="font-size:12px;font-weight:700;color:{DARK_TEXT};">{_title}</div>'
            f'<div style="font-size:10px;color:rgba(0,27,42,0.5);">{_sub}</div>'
            f'</div></div>'
        )

    _caps_html += '</div>'
    st.html(_caps_html)

    st.html(
        '<div style="padding:10px 16px 16px;border-top:1px solid rgba(0,0,0,0.1);'
        'text-align:center;color:rgba(0,27,42,0.4);font-size:10px;letter-spacing:.4px;">'
        'Powered by memodo</div>'
    )


components.html(HEADER_HTML, height=90, scrolling=False)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": say_hi(selected_sys["display"])}
    ]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_text = st.chat_input("Ask about a KPI definition or dashboard…")

if user_text:
    rag = get_rag(active_system_key)

    st.session_state.messages.append({"role": "user", "content": user_text})

    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            if rag is None:
                md = "⚠️ Could not load the knowledge base for this system."
            else:
                result = rag.answer(user_text, top_k=4)
                md = render_result(result)

            st.markdown(md)

    st.session_state.messages.append({"role": "assistant", "content": md})