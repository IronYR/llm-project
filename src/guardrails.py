import re
from dataclasses import dataclass

_JAILBREAK_PATTERNS: list[str] = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"forget\s+(you\s+are|your\s+(previous|prior|system|original))",
    r"you\s+are\s+now\s+(a|an|the)",
    r"pretend\s+(you\s+are|to\s+be)",
    r"act\s+as\s+(if\s+you\s+(are|were)|a\s+)",
    r"roleplay\s+as",
    r"DAN\s+mode",
    r"developer\s+mode",
    r"jailbreak",
    r"bypass\s+your\s+(restrictions?|filters?|guidelines?|rules?)",
    r"override\s+your\s+(restrictions?|instructions?|guidelines?|programming)",
    r"disregard\s+(your\s+)?(training|guidelines?|instructions?|restrictions?)",
    r"ignore\s+(your\s+)?(training|guidelines?|restrictions?|filters?)",
    r"reveal\s+(your\s+)?(system\s+prompt|instructions?|prompt|configuration)",
    r"show\s+(me\s+)?(your\s+)?(system\s+prompt|internal\s+instructions?)",
    r"what\s+(is|are)\s+your\s+(system\s+)?prompt",
    r"print\s+your\s+instructions",
    r"repeat\s+(the\s+)?(above|previous|system)",
    r"prompt\s+injection",
    r"admin\s+(access|mode|override|password)",
    r"unlock\s+(restricted|hidden|secret)",
    r"enable\s+unrestricted\s+mode",
    r"sudo\s+",
    r"<\s*/?system\s*>",          # XML/HTML tags injecting fake system messages
    r"\[INST\]",                   # Llama instruction injection
    r"###\s*(system|instruction)", # Markdown heading injection
    r"you\s+have\s+no\s+(restrictions?|rules?|limits?)",
    r"confidential\s+information\s+override",
    r"ignore\s+the\s+context",
    r"do\s+not\s+(use\s+the\s+)?context",
]

_JAILBREAK_RE = re.compile("|".join(_JAILBREAK_PATTERNS), re.IGNORECASE)

_INPUT_PII_RE = re.compile(
    r"\b\d{5}-\d{7}-\d\b"                        # CNIC
    r"|\bPK\d{2}[A-Z]{4}\d{16}\b"                # IBAN
    r"|\b(\+92|0)\s?\d{3}[-\s]?\d{7}\b"          # PK phone
    r"|\b\d{4}[-\s]\d{4}[-\s]\d{4}\b",           # card/account
    re.IGNORECASE,
)

_OUTPUT_PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b\d{5}-\d{7}-\d\b"),                     "[CNIC REDACTED]"),
    (re.compile(r"\bPK\d{2}[A-Z]{4}\d{16}\b"),              "[IBAN REDACTED]"),
    (re.compile(r"\b(\+92|0)\s?\d{3}[-\s]?\d{7}\b"),        "[PHONE REDACTED]"),
    (re.compile(r"\b\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]?\d{0,4}\b"), "[ACCOUNT# REDACTED]"),
]

=
@dataclass
class GuardrailResult:
    allowed:      bool
    reason:       str = ""           # Human-readable explanation
    safe_message: str = ""           # Pre-built refusal message (if not allowed)
    warning:      str = ""           # Non-blocking advisory (e.g. PII detected in input)



def check_input(query: str, config: dict) -> GuardrailResult:
    gr_cfg = config.get("guardrails", {})
    max_len = gr_cfg.get("max_input_length", 1000)

    # 1. Empty input
    if not query or not query.strip():
        return GuardrailResult(
            allowed=False,
            reason="empty_input",
            safe_message="Please enter a question so I can help you.",
        )

    # 2. Length check
    if len(query) > max_len:
        return GuardrailResult(
            allowed=False,
            reason="input_too_long",
            safe_message=(
                f"Your message is too long (limit: {max_len} characters). "
                "Please shorten your question."
            ),
        )

    # 3. Jailbreak / prompt injection
    if _JAILBREAK_RE.search(query):
        return GuardrailResult(
            allowed=False,
            reason="jailbreak_attempt",
            safe_message=(
                "I'm sorry, but I'm unable to process that request. "
                "I'm a NUST Bank customer service assistant and I can only "
                "help with banking-related questions. "
                "Please ask me about NUST Bank products or services."
            ),
        )

    # 4. Blocked topics
    blocked = gr_cfg.get("blocked_topics", [])
    query_lower = query.lower()
    for topic in blocked:
        if topic.lower() in query_lower:
            return GuardrailResult(
                allowed=False,
                reason=f"blocked_topic:{topic}",
                safe_message=(
                    "I'm sorry, I can't assist with that topic. "
                    "I'm here to help with NUST Bank products, accounts, "
                    "loans, and other banking services. How can I help you today?"
                ),
            )

    # 5. PII in input (warn, don't block)
    warning = ""
    if _INPUT_PII_RE.search(query):
        warning = (
            "⚠️ Your message appears to contain sensitive personal information "
            "(CNIC, account number, or phone number). "
            "Please avoid sharing sensitive data in chat for your security."
        )

    return GuardrailResult(allowed=True, warning=warning)


def is_banking_related(query: str, config: dict) -> bool:
    keywords = config.get("guardrails", {}).get("banking_keywords", [])
    q_lower = query.lower()
    return any(kw.lower() in q_lower for kw in keywords)



def sanitize_output(text: str) -> str:
    for pattern, replacement in _OUTPUT_PII_PATTERNS:
        text = re.sub(pattern, replacement, text)

    # Remove any LLM "thinking" artefacts or injected tags
    text = re.sub(r"<\s*(system|assistant|user|inst)\s*>.*?</\s*\1\s*>", "",
                  text, flags=re.IGNORECASE | re.DOTALL)

    return text.strip()


def check_output(text: str) -> GuardrailResult:
    cleaned = sanitize_output(text)
    if cleaned != text:
        return GuardrailResult(
            allowed=True,
            reason="pii_sanitized",
            safe_message=cleaned,
        )
    return GuardrailResult(allowed=True, safe_message=cleaned)
