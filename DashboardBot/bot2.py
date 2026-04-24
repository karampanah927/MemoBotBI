import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ----------------------------
# Config
# ----------------------------
KB_JSON_PATH  = Path(__file__).parent / "dashboardBot.json"
KPI_JSON_PATH = Path(__file__).parent / "dashboardBotKPIDefinitios.json"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# ----------------------------
# Data models
# ----------------------------
@dataclass
class PageRecord:
    report_id: str
    report_name: str
    report_url: str
    page_id: str
    page_name: str
    purpose: str
    filters_slicers: List[str]
    kpis: List[Dict[str, Any]]
    visuals: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]
    typical_questions: List[str]
    screenshot: Dict[str, Any]
    extra: Dict[str, Any]
    text_for_embedding: str

@dataclass
class KPIRecord:
    kpi_id: str
    name: str
    synonyms: List[str]
    definition: Optional[str]
    formula: Optional[str]
    assigned_date: Optional[str]
    notes: List[str]
    text_for_embedding: str

# ----------------------------
# Text builders
# ----------------------------
def _safe_list(x):
    return x if isinstance(x, list) else []

def build_page_text(d: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"REPORT: {d.get('report_name', '')}")
    lines.append(f"REPORT_URL: {d.get('report_url', '')}")
    lines.append(f"PAGE: {d.get('page_name', '')}")
    if d.get("purpose"):
        lines.append(f"PURPOSE: {d['purpose']}")
    if d.get("filters_slicers"):
        lines.append("FILTERS: " + ", ".join(d["filters_slicers"]))
    if d.get("kpis"):
        lines.append("KPIS:")
        for k in d["kpis"]:
            name = k.get("name", "")
            syn  = _safe_list(k.get("synonyms"))
            lines.append(f"- {name}" + (f" | synonyms: {', '.join(syn)}" if syn else ""))
    if d.get("visuals"):
        lines.append("VISUALS:")
        for v in d["visuals"]:
            row = f"- {v.get('title', '')}"
            if v.get("use_for"):
                row += f" | use_for: {v['use_for']}"
            syn = _safe_list(v.get("synonyms"))
            if syn:
                row += f" | synonyms: {', '.join(syn)}"
            lines.append(row)
    if d.get("tables"):
        lines.append("TABLES:")
        for t in d["tables"]:
            cols = _safe_list(t.get("columns_example"))
            lines.append(f"- {t.get('title', '')}" + (f" | columns: {', '.join(cols)}" if cols else ""))
    if d.get("typical_questions"):
        lines.append("TYPICAL_QUESTIONS:")
        for q in d["typical_questions"]:
            lines.append(f"- {q}")
    return "\n".join(lines).strip()

def build_kpi_text(k: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"KPI: {k.get('name', '')}")
    syns = k.get("synonyms") or []
    if syns:
        lines.append(f"SYNONYMS: {', '.join(syns)}")
    if k.get("definition"):
        lines.append(f"DEFINITION: {k['definition']}")
    if k.get("formula"):
        lines.append(f"FORMULA: {k['formula']}")
    if k.get("assigned_date"):
        lines.append(f"DATE_ASSIGNMENT: {k['assigned_date']}")
    notes = k.get("notes") or []
    if notes:
        lines.append("NOTES: " + " | ".join(notes))
    return "\n".join(lines).strip()

# ----------------------------
# Loaders
# ----------------------------
def load_kb_pages(kb_path) -> List[PageRecord]:
    with open(kb_path, "r", encoding="utf-8") as f:
        kb = json.load(f)

    pages: List[PageRecord] = []
    for dash in kb.get("dashboards", []):
        report_id   = dash.get("report_id", "")
        report_name = dash.get("report_name", "")
        report_url  = dash.get("report_url", "")
        for p in dash.get("pages", []):
            known = {"page_id","page_name","purpose","filters_slicers",
                     "kpis","visuals","tables","typical_questions","screenshot"}
            doc = {
                "report_id": report_id, "report_name": report_name,
                "report_url": report_url,
                "page_id":   p.get("page_id", ""),
                "page_name": p.get("page_name", ""),
                "purpose":   p.get("purpose", ""),
                "filters_slicers":   _safe_list(p.get("filters_slicers")),
                "kpis":      _safe_list(p.get("kpis")),
                "visuals":   _safe_list(p.get("visuals")),
                "tables":    _safe_list(p.get("tables")),
                "typical_questions": _safe_list(p.get("typical_questions")),
                "screenshot": p.get("screenshot") or {},
            }
            pages.append(PageRecord(
                **{k: doc[k] for k in doc},
                extra={k: v for k, v in p.items() if k not in known},
                text_for_embedding=build_page_text(doc),
            ))
    return pages

def load_kpi_defs(kpi_path) -> List[KPIRecord]:
    with open(kpi_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records: List[KPIRecord] = []
    for k in data.get("kpis", []):
        records.append(KPIRecord(
            kpi_id       = k.get("kpi_id", ""),
            name         = k.get("name", ""),
            synonyms     = k.get("synonyms") or [],
            definition   = k.get("definition"),
            formula      = k.get("formula"),
            assigned_date= k.get("assigned_date"),
            notes        = k.get("notes") or [],
            text_for_embedding=build_kpi_text(k),
        ))
    return records

# ----------------------------
# RAG
# ----------------------------
class DashboardRAG:
    def __init__(self, kb_json_path, kpi_json_path=None):
        self.pages    = load_kb_pages(kb_json_path)
        self.kpi_defs = load_kpi_defs(kpi_json_path) if kpi_json_path and Path(kpi_json_path).exists() else []
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
        self.page_embeddings: Optional[np.ndarray] = None
        self.kpi_embeddings:  Optional[np.ndarray] = None

        self.client = None
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = OpenAI(api_key=api_key)

    def build(self) -> None:
        page_texts = [p.text_for_embedding for p in self.pages]
        self.page_embeddings = np.asarray(
            self.embedder.encode(page_texts, normalize_embeddings=True), dtype=np.float32
        )
        if self.kpi_defs:
            kpi_texts = [k.text_for_embedding for k in self.kpi_defs]
            self.kpi_embeddings = np.asarray(
                self.embedder.encode(kpi_texts, normalize_embeddings=True), dtype=np.float32
            )

    def _retrieve_pages(self, query: str, top_k: int) -> List[Tuple[PageRecord, float]]:
        q = np.asarray(self.embedder.encode([query], normalize_embeddings=True), dtype=np.float32)[0]
        scores = self.page_embeddings @ q
        idx = np.argsort(scores)[::-1][:top_k]
        return [(self.pages[int(i)], float(scores[int(i)])) for i in idx]

    def _retrieve_kpis(self, query: str, top_k: int) -> List[Tuple[KPIRecord, float]]:
        if self.kpi_embeddings is None:
            return []
        q = np.asarray(self.embedder.encode([query], normalize_embeddings=True), dtype=np.float32)[0]
        scores = self.kpi_embeddings @ q
        idx = np.argsort(scores)[::-1][:top_k]
        return [(self.kpi_defs[int(i)], float(scores[int(i)])) for i in idx]

    def answer(self, user_question: str, top_k: int = 4) -> Dict[str, Any]:
        page_hits = self._retrieve_pages(user_question, top_k=top_k)
        kpi_hits  = self._retrieve_kpis(user_question, top_k=3)

        # ── Retrieval-only (no API key) ──────────────────────────────────────
        if not self.client:
            return {
                "answer": "Here are the most relevant results. Set OPENAI_API_KEY for a natural-language explanation.",
                "answer_type": "both",
                "kpi_definitions": [
                    {
                        "name": k.name,
                        "definition": k.definition or "Definition not yet available.",
                        "formula": k.formula,
                        "assigned_date": k.assigned_date,
                        "notes": k.notes,
                        "score": round(s, 4),
                    }
                    for k, s in kpi_hits
                ],
                "best_matches": [
                    {
                        "score": round(s, 4),
                        "report_name": p.report_name,
                        "report_url":  p.report_url,
                        "page_name":   p.page_name,
                        "page_id":     p.page_id,
                        "filters_slicers": p.filters_slicers,
                        "screenshot":  p.screenshot,
                    }
                    for p, s in page_hits
                ],
            }

        # ── LLM-assisted ─────────────────────────────────────────────────────
        page_ctx = "\n\n---\n\n".join(
            f"[PAGE score={s:.3f}]\nREPORT: {p.report_name}\nPAGE: {p.page_name}\nURL: {p.report_url}\n{p.text_for_embedding}"
            for p, s in page_hits
        )
        kpi_ctx = "\n\n---\n\n".join(
            f"[KPI score={s:.3f}]\n{k.text_for_embedding}"
            for k, s in kpi_hits
        )

        system = (
            "You are Memodo BI Bot — an expert assistant for Power BI dashboards and KPI definitions.\n"
            "You help users with TWO things:\n"
            "  1. Dashboard navigation: point them to the right Power BI report and page.\n"
            "  2. KPI definitions: explain what a KPI means, its formula, and date assignment logic.\n\n"
            "Rules:\n"
            "- Use ONLY the provided context. Never invent dashboards or KPI definitions.\n"
            "- If the question is about KPI meaning/formula/calculation → populate kpi_definitions.\n"
            "- If the question is about finding a dashboard/chart/visual → populate best_matches.\n"
            "- If both apply, populate both arrays.\n\n"
            "Return valid JSON with exactly these fields:\n"
            "  answer        : string – clear, helpful response (2-4 sentences)\n"
            "  answer_type   : 'kpi_definition' | 'dashboard_navigation' | 'both'\n"
            "  kpi_definitions: array – each item: { name, definition, formula, assigned_date, notes }\n"
            "  best_matches  : array – each item: { report_name, report_url, page_name, page_id, why }\n"
            "  missing_info  : null or string\n"
        )

        user_msg = (
            f"User question: {user_question}\n\n"
            f"== DASHBOARD PAGES CONTEXT ==\n{page_ctx}\n\n"
            f"== KPI DEFINITIONS CONTEXT ==\n{kpi_ctx}\n\n"
            "Return JSON only."
        )

        resp = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.2,
        )
        txt = resp.choices[0].message.content.strip()
        # strip possible markdown code fences
        if txt.startswith("```"):
            txt = "\n".join(txt.split("\n")[1:])
        if txt.endswith("```"):
            txt = txt[: txt.rfind("```")]

        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            return {
                "answer": "I found relevant info but had a formatting issue. Here are the raw results.",
                "answer_type": "both",
                "kpi_definitions": [
                    {"name": k.name, "definition": k.definition,
                     "formula": k.formula, "assigned_date": k.assigned_date, "notes": k.notes}
                    for k, _ in kpi_hits[:2]
                ],
                "best_matches": [
                    {"report_name": p.report_name, "report_url": p.report_url,
                     "page_name": p.page_name, "page_id": p.page_id, "why": ""}
                    for p, _ in page_hits[:3]
                ],
                "missing_info": None,
                "raw_output": txt,
            }


if __name__ == "__main__":
    rag = DashboardRAG(KB_JSON_PATH, KPI_JSON_PATH)
    rag.build()
    print(f"Loaded {len(rag.pages)} pages | {len(rag.kpi_defs)} KPI definitions")

    while True:
        q = input("\nAsk a question (or 'exit'): ").strip()
        if q.lower() == "exit":
            break
        print(json.dumps(rag.answer(q, top_k=4), indent=2, ensure_ascii=False))
