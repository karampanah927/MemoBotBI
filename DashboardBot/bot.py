"""Multi-system KPI assistant backend (RAG)."""
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE              = Path(__file__).parent
KPI_DIR           = HERE / "memodo_kpi_json_files"
POWERBI_NAV_PATH  = KPI_DIR / "dashboardBot_powerbi.json"
SYSTEM_INDEX_PATH = KPI_DIR / "kpi_system_file_index.json"
ICONS_BASE_DIR    = HERE / "source_system_icons_and_json"
ICON_REFS_PATH    = ICONS_BASE_DIR / "kpi_system_icon_references.json"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# ── System metadata ────────────────────────────────────────────────────────────
_DISPLAY_NAMES: Dict[str, str] = {
    "PowerBI":            "Power BI",
    "WaWi":               "WaWi",
    "MeWi":               "MeWi",
    "DATEV":              "DATEV",
    "Prolag":             "Prolag",
    "Salesforce":         "Salesforce",
    "Shopware6":          "Shopware 6",
    "GA4_for_Shop":       "GA4 for Shop",
    "GC-Definitions":     "GC Definitions",
    "Sales":              "Sales",
    "Purchasing":         "Purchasing",
    "Finance":            "Finance",
    "Overall_MGMT_Board": "Overall MGMT Board",
}

_ICON_KEYS: Dict[str, str] = {
    "PowerBI":      "powerbi",
    "WaWi":         "wawi",
    "MeWi":         "mewi",
    "DATEV":        "datev",
    "Prolag":       "prolag",
    "Salesforce":   "salesforce",
    "Shopware6":    "shopware6",
    "GA4_for_Shop": "ga4",
}


def get_display_name(system_name: str) -> str:
    return _DISPLAY_NAMES.get(system_name, system_name.replace("_", " ").replace("-", " "))


def get_icon_key(system_name: str) -> str:
    return _ICON_KEYS.get(system_name, system_name.lower().replace("_", "").replace("-", ""))


def load_system_index() -> List[Dict[str, Any]]:
    with open(SYSTEM_INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f).get("systems", [])


def load_icon_references() -> Dict[str, str]:
    if ICON_REFS_PATH.exists():
        with open(ICON_REFS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ── Data models ───────────────────────────────────────────────────────────────
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


# ── Text builders ─────────────────────────────────────────────────────────────
def _safe_list(x):
    return x if isinstance(x, list) else []


def build_page_text(d: Dict[str, Any]) -> str:
    lines = [
        f"REPORT: {d.get('report_name', '')}",
        f"REPORT_URL: {d.get('report_url', '')}",
        f"PAGE: {d.get('page_name', '')}",
    ]
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
            lines.append(
                f"- {t.get('title', '')}" + (f" | columns: {', '.join(cols)}" if cols else "")
            )
    if d.get("typical_questions"):
        lines.append("TYPICAL_QUESTIONS:")
        for q in d["typical_questions"]:
            lines.append(f"- {q}")
    return "\n".join(lines).strip()


def build_kpi_text(k: Dict[str, Any]) -> str:
    lines = [f"KPI: {k.get('name', '')}"]
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


# ── Loaders ───────────────────────────────────────────────────────────────────
def _load_nav_pages(nav_path: Path) -> List[PageRecord]:
    with open(nav_path, "r", encoding="utf-8") as f:
        kb = json.load(f)
    pages: List[PageRecord] = []
    for dash in kb.get("dashboards", []):
        report_id   = dash.get("report_id", "")
        report_name = dash.get("report_name", "")
        report_url  = dash.get("report_url", "")
        for p in dash.get("pages", []):
            known = {
                "page_id", "page_name", "purpose", "filters_slicers",
                "kpis", "visuals", "tables", "typical_questions", "screenshot",
            }
            doc = {
                "report_id":         report_id,
                "report_name":       report_name,
                "report_url":        report_url,
                "page_id":           p.get("page_id", ""),
                "page_name":         p.get("page_name", ""),
                "purpose":           p.get("purpose", ""),
                "filters_slicers":   _safe_list(p.get("filters_slicers")),
                "kpis":              _safe_list(p.get("kpis")),
                "visuals":           _safe_list(p.get("visuals")),
                "tables":            _safe_list(p.get("tables")),
                "typical_questions": _safe_list(p.get("typical_questions")),
                "screenshot":        p.get("screenshot") or {},
            }
            pages.append(PageRecord(
                **{k: doc[k] for k in doc},
                extra={k: v for k, v in p.items() if k not in known},
                text_for_embedding=build_page_text(doc),
            ))
    return pages


def _load_kpi_defs(kpi_path: Path) -> List[KPIRecord]:
    with open(kpi_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records: List[KPIRecord] = []
    for k in data.get("kpis", []):
        records.append(KPIRecord(
            kpi_id        = k.get("kpi_id", ""),
            name          = k.get("name", ""),
            synonyms      = k.get("synonyms") or [],
            definition    = k.get("definition"),
            formula       = k.get("formula"),
            assigned_date = k.get("assigned_date"),
            notes         = k.get("notes") or [],
            text_for_embedding=build_kpi_text(k),
        ))
    return records


# ── RAG ───────────────────────────────────────────────────────────────────────
class MultiSystemRAG:
    def __init__(
        self,
        kpi_path: Optional[Path],
        nav_path: Optional[Path],
        system_name: str,
    ):
        self.system_name    = system_name
        self.has_navigation = nav_path is not None and nav_path.exists()
        self.kpi_defs       = _load_kpi_defs(kpi_path) if kpi_path and kpi_path.exists() else []
        self.pages          = _load_nav_pages(nav_path) if self.has_navigation else []
        self.embedder       = SentenceTransformer(EMBED_MODEL_NAME)
        self.page_embeddings: Optional[np.ndarray] = None
        self.kpi_embeddings:  Optional[np.ndarray] = None
        self.client: Optional[OpenAI] = None
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = OpenAI(api_key=api_key)

    def build(self) -> None:
        if self.kpi_defs:
            texts = [k.text_for_embedding for k in self.kpi_defs]
            self.kpi_embeddings = np.asarray(
                self.embedder.encode(texts, normalize_embeddings=True), dtype=np.float32
            )
        if self.pages:
            texts = [p.text_for_embedding for p in self.pages]
            self.page_embeddings = np.asarray(
                self.embedder.encode(texts, normalize_embeddings=True), dtype=np.float32
            )

    def _retrieve_kpis(self, query: str, top_k: int) -> List[Tuple[KPIRecord, float]]:
        if self.kpi_embeddings is None:
            return []
        q = np.asarray(
            self.embedder.encode([query], normalize_embeddings=True), dtype=np.float32
        )[0]
        scores = self.kpi_embeddings @ q
        idx = np.argsort(scores)[::-1][:top_k]
        return [(self.kpi_defs[int(i)], float(scores[int(i)])) for i in idx]

    def _retrieve_pages(self, query: str, top_k: int) -> List[Tuple[PageRecord, float]]:
        if self.page_embeddings is None:
            return []
        q = np.asarray(
            self.embedder.encode([query], normalize_embeddings=True), dtype=np.float32
        )[0]
        scores = self.page_embeddings @ q
        idx = np.argsort(scores)[::-1][:top_k]
        return [(self.pages[int(i)], float(scores[int(i)])) for i in idx]

    def answer(self, user_question: str, top_k: int = 4) -> Dict[str, Any]:
        kpi_hits  = self._retrieve_kpis(user_question, top_k=3)
        page_hits = self._retrieve_pages(user_question, top_k=top_k) if self.has_navigation else []

        # ── Retrieval-only (no API key) ─────────────────────────────────────
        if not self.client:
            result: Dict[str, Any] = {
                "answer": (
                    "Here are the most relevant results. "
                    "Set OPENAI_API_KEY for a natural-language explanation."
                ),
                "answer_type":     "both",
                "kpi_definitions": [
                    {
                        "name":          k.name,
                        "definition":    k.definition or "Definition not yet available.",
                        "formula":       k.formula,
                        "assigned_date": k.assigned_date,
                        "notes":         k.notes,
                        "score":         round(s, 4),
                    }
                    for k, s in kpi_hits
                ],
                "best_matches": [
                    {
                        "score":           round(s, 4),
                        "report_name":     p.report_name,
                        "report_url":      p.report_url,
                        "page_name":       p.page_name,
                        "page_id":         p.page_id,
                        "filters_slicers": p.filters_slicers,
                        "screenshot":      p.screenshot,
                    }
                    for p, s in page_hits
                ],
            }
            if not self.has_navigation:
                result["note"] = f"No dashboard navigation is available for {self.system_name}."
            return result

        # ── LLM-assisted ────────────────────────────────────────────────────
        kpi_ctx = "\n\n---\n\n".join(
            f"[KPI score={s:.3f}]\n{k.text_for_embedding}" for k, s in kpi_hits
        )
        page_ctx = "\n\n---\n\n".join(
            f"[PAGE score={s:.3f}]\nREPORT: {p.report_name}\n"
            f"PAGE: {p.page_name}\nURL: {p.report_url}\n{p.text_for_embedding}"
            for p, s in page_hits
        ) if page_hits else ""

        nav_rule = (
            "- For 'where to find' questions → populate best_matches with "
            "report_name, report_url, page_name, page_id, why."
            if self.has_navigation else
            f"- This system ({self.system_name}) has no Power BI navigation. "
            "If asked where to find a KPI, state: 'No dashboard URL is available for this system.' "
            "Set best_matches to []."
        )

        system_prompt = (
            f"You are Memodo BI Bot — expert assistant for KPI definitions and analytics.\n"
            f"Selected source system: {self.system_name}\n\n"
            "Rules:\n"
            "- Use ONLY the provided context. Never invent KPI definitions.\n"
            "- For definition/formula/calculation questions → populate kpi_definitions.\n"
            + nav_rule + "\n"
            "- Populate both arrays when both apply.\n\n"
            "Return valid JSON with exactly these fields:\n"
            "  answer         : string (2–4 sentences)\n"
            "  answer_type    : 'kpi_definition' | 'dashboard_navigation' | 'both'\n"
            "  kpi_definitions: array [{ name, definition, formula, assigned_date, notes }]\n"
            "  best_matches   : array [{ report_name, report_url, page_name, page_id, why }]\n"
            "  missing_info   : null | string\n"
        )

        ctx_parts = [
            f"User question: {user_question}",
            f"== KPI DEFINITIONS ==\n{kpi_ctx}",
        ]
        if page_ctx:
            ctx_parts.append(f"== DASHBOARD PAGES ==\n{page_ctx}")
        ctx_parts.append("Return JSON only.")

        resp = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": "\n\n".join(ctx_parts)},
            ],
            temperature=0.2,
        )
        txt = resp.choices[0].message.content.strip()
        if txt.startswith("```"):
            txt = "\n".join(txt.split("\n")[1:])
        if txt.endswith("```"):
            txt = txt[: txt.rfind("```")]

        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            return {
                "answer":          "Found relevant info but had a formatting issue.",
                "answer_type":     "both",
                "kpi_definitions": [
                    {
                        "name": k.name, "definition": k.definition,
                        "formula": k.formula, "assigned_date": k.assigned_date, "notes": k.notes,
                    }
                    for k, _ in kpi_hits[:2]
                ],
                "best_matches": [
                    {
                        "report_name": p.report_name, "report_url": p.report_url,
                        "page_name": p.page_name, "page_id": p.page_id, "why": "",
                    }
                    for p, _ in page_hits[:3]
                ],
                "missing_info": None,
                "raw_output":   txt,
            }


def make_rag(system_entry: Dict[str, Any]) -> MultiSystemRAG:
    """Factory: build a MultiSystemRAG from a kpi_system_file_index entry."""
    kpi_file = system_entry.get("kpi_definition_file", "")
    kpi_path = KPI_DIR / kpi_file if kpi_file else None

    # Fallback: if config references a non-existent "_updated" file, try without it
    if kpi_path and not kpi_path.exists() and "_updated" in kpi_file:
        fallback = KPI_DIR / kpi_file.replace("_updated", "")
        if fallback.exists():
            kpi_path = fallback

    nav_path = POWERBI_NAV_PATH if system_entry.get("has_urls") else None
    return MultiSystemRAG(
        kpi_path=kpi_path,
        nav_path=nav_path,
        system_name=system_entry["system"],
    )
