import json
import streamlit as st

from bot2 import DashboardRAG, KB_JSON_PATH

@st.cache_resource
def get_rag():
    rag = DashboardRAG(KB_JSON_PATH)
    rag.build()
    return rag

def render_result(result: dict) -> str:
    """
    Convert your JSON dict into nice Markdown with clickable links.
    """
    lines = []
    lines.append(result.get("answer", ""))
    lines.append("")

    matches = result.get("best_matches") or []
    if matches:
        lines.append("### Best matches")
        for i, m in enumerate(matches, 1):
            report = m.get("report_name", "")
            page = m.get("page_name", "")
            url = m.get("report_url", "")
            why = m.get("why", "")
            score = m.get("score", None)

            title = f"{i}) **{report}** → *{page}*"
            if score is not None:
                title += f" (score: {score})"

            lines.append(title)
            if url:
                lines.append(f"- Link: {url}")
            if why:
                lines.append(f"- Why: {why}")
            # optional: screenshot info if present
            screenshot = m.get("screenshot")
            if screenshot:
                lines.append(f"- Screenshot: `{json.dumps(screenshot, ensure_ascii=False)}`")
            lines.append("")

    note = result.get("note")
    if note:
        lines.append(f"ℹ️ {note}")

    missing = result.get("missing_info")
    if missing:
        lines.append(f"❓ Missing info: {missing}")

    raw = result.get("raw_output")
    if raw:
        lines.append("### Raw model output")
        lines.append(f"```text\n{raw}\n```")

    return "\n".join(lines).strip()

st.set_page_config(page_title="Dashboard Bot", page_icon="🤖", layout="centered")
st.title("🤖 Memodo BI Bot")
st.caption("Ask about KPIs / charts and get the best Power BI pages + links.")

rag = get_rag()

def say_hi():
    return "👋 Hi! I'm Memodo BI Bot. Ask me anything about your KPIs, charts, or Power BI pages!"

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": say_hi()}]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"], unsafe_allow_html=False)

user_text = st.chat_input("Ask a KPI/chart question…")

if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        with st.spinner("Searching…"):
            result = rag.answer(user_text, top_k=5)
            md = render_result(result)
            st.markdown(md)

    st.session_state.messages.append({"role": "assistant", "content": md})
