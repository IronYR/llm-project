import re
import json
import logging
from pathlib import Path
from typing import Any

import openpyxl

logger = logging.getLogger(__name__)

_PII_PATTERNS: list[tuple[str, str]] = [
    (r"\b\d{5}-\d{7}-\d\b", "[CNIC REDACTED]"),                              # CNIC: 12345-1234567-1
    (r"\bPK\d{2}[A-Z]{4}\d{16}\b", "[IBAN REDACTED]"),                       # IBAN
    (r"\b(\+92|0)\s?\d{3}[-\s]?\d{7}\b", "[PHONE REDACTED]"),               # PK mobile
    (r"\b\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]?\d{0,4}\b", "[ACCOUNT# REDACTED]"), # card/account numbers
    (r"\b[A-Za-z0-9._%+-]+@(?!NUSTbank)[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
     "[EMAIL REDACTED]"),  # emails (keep bank's official email)
]

PRODUCT_SHEET_MAP: dict[str, str] = {
    "LCA":          "Little Champs Account",
    "NAA":          "NUST Asaan Account",
    "NWA":          "NUST Waqaar Account",
    "PWRA":         "PakWatan Remittance Account",
    "RDA":          "Roshan Digital Account",
    "VPCA":         "Value Plus Current Account (Individual)",
    "VP-BA":        "Value Plus Business Account",
    "VPBA":         "Value Premium Business Account",
    "NSDA":         "NUST Special Deposit Account",
    "PLS":          "Profit and Loss Sharing (PLS) Account",
    "CDA":          "Current Deposit Account",
    "NMA":          "NUST Maximiser Account",
    "NADA":         "NUST Asaan Digital Account",
    "NADRA":        "NUST Asaan Digital Remittance Account",
    "NUST4Car":     "NUST Auto Finance (NUST4Car)",
    "ESFCA":        "Exporters Special Foreign Currency Account",
    "NFDA":         "NUST Freelancer Digital Account",
    "NSA":          "NUST Sahar Account (Women)",
    "PF":           "NUST Personal Finance",
    "NMC":          "NUST Mastercard Credit Card",
    "NMF":          "NUST Mortgage Finance",
    "NSF":          "NUST Sahar Finance (SME Women)",
    "NIF":          "NUST Imarat Finance (SME)",
    "NUF":          "NUST Ujala Finance (Renewable Energy)",
    "NFMF":         "NUST Flour Mill Finance",
    "NFBF":         "NUST Fauri Business Finance",
    "PMYB &ALS":    "PM Youth Business & Agriculture Loan Scheme",
    "NRF":          "NUST Rice Finance",
    "NHF":          "NUST Hunarmand Finance",
    "Nust Life":    "NUST Life Bancassurance",
    "EFU Life":     "EFU Life Bancassurance",
    "Jubilee Life ":"Jubilee Life Bancassurance",
    "HOME REMITTANCE": "Home Remittance Service",
}

# Sheets that are not product Q&A (skip them)
_SKIP_SHEETS = {"Main", "Rate Sheet July 1 2024", "Sheet1"}



def _fix_encoding(text: str) -> str:
    """Repair common mojibake patterns and normalise whitespace."""
    if not isinstance(text, str):
        return ""
    table = {
        "â€œ": '"', "â€\x9d": '"', "â€™": "'", "â€˜": "'",
        "\xe2\x80\x93": "\u2013", "\xe2\x80\x94": "\u2014", "\xc2\xb7": "\u00b7", "\xc2": "",
        "\xa0": " ", "\u200b": "", "\u2013": "–", "\u2014": "—",
    }
    for bad, good in table.items():
        text = text.replace(bad, good)
    # Collapse multiple spaces / newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _anonymize(text: str) -> str:
    """Replace PII patterns with safe labels."""
    for pattern, replacement in _PII_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _is_question(text: str) -> bool:
    """
    Return True if the text looks like a question.
    Handles numbered questions like '1.What is...?' and
    non-standard questions like 'I would like to inquire...?'
    """
    t = text.strip()
    # Strip leading number+dot prefix  e.g. "1.What" → "What"
    t = re.sub(r"^\d+\.\s*", "", t)
    return t.endswith("?")


def _clean_cell(value: Any) -> str:
    """Convert an XLSX cell value to a clean string, dropping formulas and pure numbers."""
    if value is None:
        return ""
    s = str(value).strip()
    if s.startswith("="):          # Excel formula reference → skip
        return ""
    if re.fullmatch(r"\d+\.?\d*", s):  # bare profit-rate number → skip
        return ""
    return _fix_encoding(s)


def _row_to_text(row: tuple) -> str:
    """Merge all non-empty cells in a row into a single string."""
    parts = [_clean_cell(c) for c in row]
    parts = [p for p in parts if p and p.lower() not in ("main", "none")]
    # For multi-column rows (feature tables) join with "  |  "
    return "  |  ".join(parts) if parts else ""


def _parse_sheet(ws, product_name: str) -> list[dict]:
    """
    Extract Q&A pairs from a single product sheet.
    """
    rows = list(ws.iter_rows(values_only=True))

    pre_question_lines: list[str] = []
    current_question: str | None = None
    answer_parts: list[str] = []
    documents: list[dict] = []

    for row in rows:
        text = _row_to_text(row)
        if not text:
            continue

        # Skip the sheet title row if it matches the product name
        if text.strip().rstrip(".") == product_name.strip().rstrip("."):
            continue

        if _is_question(text):
            # Save previous Q&A
            if current_question and answer_parts:
                answer = "\n".join(answer_parts).strip()
                if answer:
                    documents.append({"question": current_question, "answer": answer})
            elif not current_question and pre_question_lines:
                # Emit preamble as an 'about' document
                about_text = "\n".join(pre_question_lines).strip()
                if about_text:
                    documents.append({
                        "question": f"What is {product_name}?",
                        "answer": about_text,
                    })
                pre_question_lines = []

            current_question = text
            answer_parts = []
        else:
            if current_question is None:
                pre_question_lines.append(text)
            else:
                answer_parts.append(text)

    # Flush last Q&A
    if current_question and answer_parts:
        answer = "\n".join(answer_parts).strip()
        if answer:
            documents.append({"question": current_question, "answer": answer})

    return documents


def parse_xlsx(filepath: Path) -> list[dict]:
    wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
    all_docs: list[dict] = []

    for sheet_name in wb.sheetnames:
        if sheet_name in _SKIP_SHEETS:
            continue

        product_name = PRODUCT_SHEET_MAP.get(sheet_name, sheet_name.strip())
        ws = wb[sheet_name]

        sheet_docs = _parse_sheet(ws, product_name)

        for doc in sheet_docs:
            all_docs.append({
                "category": product_name,
                "source": "NUST Bank-Product-Knowledge.xlsx",
                "source_sheet": sheet_name,
                **doc,
            })

        logger.info("Sheet %-20s (%s): %d Q&A pairs", sheet_name, product_name, len(sheet_docs))

    wb.close()
    logger.info("XLSX total: %d documents across %d sheets", len(all_docs), len(PRODUCT_SHEET_MAP))
    return all_docs



def parse_json_faq(filepath: Path) -> list[dict]:
    """Parse the funds-transfer & app-features FAQ JSON file."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    docs: list[dict] = []
    for cat_obj in data.get("categories", []):
        category = _fix_encoding(cat_obj.get("category", "General"))
        for qa in cat_obj.get("questions", []):
            question = _fix_encoding(qa.get("question", ""))
            answer   = _fix_encoding(qa.get("answer", ""))
            if question and answer:
                docs.append({
                    "category": category,
                    "source": "funds_transfer_app_features_faq.json",
                    "source_sheet": category,
                    "question": question,
                    "answer": answer,
                })

    logger.info("JSON FAQ: %d Q&A pairs extracted", len(docs))
    return docs



def _build_text(doc: dict) -> str:
    """Full text representation used for embedding."""
    q = doc["question"]
    a = doc["answer"]
    cat = doc["category"]
    if q:
        return f"Product: {cat}\nQuestion: {q}\nAnswer: {a}"
    return f"Product: {cat}\n{a}"


def _process_documents(raw_docs: list[dict]) -> list[dict]:
    """Clean, anonymize, deduplicate, and assign IDs to all documents."""
    seen: set[str] = set()
    processed: list[dict] = []

    for i, doc in enumerate(raw_docs):
        question = _anonymize(doc.get("question", ""))
        answer   = _anonymize(doc.get("answer", ""))

        if not answer.strip():
            continue

        # Deduplication based on (category, question)
        key = f"{doc['category']}::{question.lower().strip()}"
        if key in seen:
            continue
        seen.add(key)

        text = _build_text({**doc, "question": question, "answer": answer})

        processed.append({
            "id":           f"doc_{len(processed):04d}",
            "category":     doc["category"],
            "source":       doc["source"],
            "source_sheet": doc["source_sheet"],
            "question":     question,
            "answer":       answer,
            "text":         text,
        })

    return processed



def run_ingestion(config: dict) -> list[dict]:
    raw_dir       = Path(config["data"]["raw_dir"])
    processed_dir = Path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_docs: list[dict] = []

    # --- XLSX ---
    xlsx_path = raw_dir / "NUST Bank-Product-Knowledge.xlsx"
    if xlsx_path.exists():
        logger.info("Parsing XLSX: %s", xlsx_path)
        raw_docs.extend(parse_xlsx(xlsx_path))
    else:
        logger.warning("XLSX not found: %s", xlsx_path)

    # --- JSON ---
    json_path = raw_dir / "funds_transfer_app_features_faq.json"
    if json_path.exists():
        logger.info("Parsing JSON FAQ: %s", json_path)
        raw_docs.extend(parse_json_faq(json_path))
    else:
        logger.warning("JSON FAQ not found: %s", json_path)

    # --- Process ---
    logger.info("Post-processing %d raw documents...", len(raw_docs))
    processed = _process_documents(raw_docs)
    logger.info("Final document count after dedup: %d", len(processed))

    # --- Save ---
    kb_path = processed_dir / config["data"]["knowledge_base_file"]
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump({"documents": processed, "total": len(processed)}, f,
                  indent=2, ensure_ascii=False)
    logger.info("Saved knowledge base → %s", kb_path)

    return processed


def ingest_new_document(text: str, category: str, config: dict) -> dict:
    processed_dir = Path(config["data"]["processed_dir"])
    kb_path = processed_dir / config["data"]["knowledge_base_file"]

    # Load existing KB
    if kb_path.exists():
        with open(kb_path, encoding="utf-8") as f:
            kb = json.load(f)
        docs = kb["documents"]
    else:
        docs = []

    cleaned_text = _anonymize(_fix_encoding(text))
    new_id = f"doc_{len(docs):04d}"

    new_doc: dict = {
        "id":           new_id,
        "category":     category,
        "source":       "user_upload",
        "source_sheet": category,
        "question":     "",
        "answer":       cleaned_text,
        "text":         f"Product: {category}\n{cleaned_text}",
    }

    docs.append(new_doc)
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump({"documents": docs, "total": len(docs)}, f, indent=2, ensure_ascii=False)

    logger.info("Ingested new document (id=%s, category=%s)", new_id, category)
    return new_doc


def load_knowledge_base(config: dict) -> list[dict]:
    """Load the processed knowledge base from disk."""
    kb_path = Path(config["data"]["processed_dir"]) / config["data"]["knowledge_base_file"]
    if not kb_path.exists():
        raise FileNotFoundError(
            f"Knowledge base not found at {kb_path}. "
            "Run 'python ingest.py' first to build it."
        )
    with open(kb_path, encoding="utf-8") as f:
        kb = json.load(f)
    return kb["documents"]
