"""
07_prompting.py
---------------
Pipeline Stage 7: PROMPTING

MarketPulse AI
AI-powered Marketing Intelligence

Production V7

Main goals:
- Python Analytics is the single source of truth for rankings.
- Compact prompts to stay under Groq TPM limits.
- Retrieved documents remain available for UI evidence.
- LLM receives only the evidence needed for reasoning.
- No unsupported rankings.
- No unsupported causal claims.
- No hallucinated sponsors, brands, platforms, or audience facts.
- Safe fallback when the LLM request is too large.
- Compatible with the existing Streamlit application.
"""

import importlib.util
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# DYNAMIC IMPORT
# ============================================================

def _import(
    module_filename,
    module_name,
):
    """
    Dynamically import a Python module
    from the same project directory.
    """

    module_path = os.path.join(
        BASE_DIR,
        module_filename,
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        module_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load module: {module_filename}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


# ============================================================
# IMPORT RETRIEVAL PIPELINE
# ============================================================

_retrieve = _import(
    "06_retrieve_context.py",
    "retrieve_mod",
)

load_vectorstore = (
    _retrieve.load_vectorstore
)

retrieve_context = (
    _retrieve.retrieve_context
)

format_context = (
    _retrieve.format_context
)


# ============================================================
# IMPORT ANALYTICS PIPELINE
# ============================================================

_analytics = _import(
    "08_analytics.py",
    "analytics_mod",
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

DEFAULT_MODEL = (
    "llama-3.1-8b-instant"
)

# Keep generated output controlled
# to reduce Groq TPM usage.
MAX_OUTPUT_TOKENS = 1200


# ============================================================
# PROMPT
# ============================================================

PROMPT_TEMPLATE = """
You are MarketPulse AI, a senior marketing intelligence analyst.

Analyze the user's question using ONLY:

1. AUTHORITATIVE PYTHON ANALYTICS
2. COMPACT RETRIEVED EVIDENCE

============================================================
CORE RULES
============================================================

Python Analytics is the single source of truth for:

- top performer
- engagement ranking
- engagement score
- averages
- metric leaders
- platform summaries
- category summaries

Never replace a Python ranking with your own ranking.

If Python Analytics identifies a top-performing content item,
you MUST use that exact item.

Do NOT calculate a new Engagement Score.

Do NOT rank content manually using:
- Likes
- Comments
- Shares
- Views

when Python Analytics already provides an Engagement Score ranking.

Do NOT invent:
- numbers
- rankings
- brands
- sponsors
- influencers
- audience demographics
- causes
- trends
- facts not supported by the evidence

Sponsorship is descriptive only.

Never claim that sponsorship caused higher engagement.

A platform or category with only one retrieved item
must NOT be described as definitively outperforming another group.

Small or unbalanced samples must be acknowledged.

Every factual statement based on retrieved evidence
must use an existing source citation such as [Source 1].

Do not invent source citations.

Possible explanations must be clearly labeled as:

- interpretation
- hypothesis
- possible explanation

Never present correlation as causation.

If evidence is insufficient, explicitly say so.

============================================================
REQUIRED OUTPUT
============================================================

Return ONLY these sections:

### 🎯 Key Insight

Give the single most important answer.

Include the authoritative top performer if available:

- Content ID
- Engagement Score
- Platform
- Category
- Source citation when available

Do not create a new ranking.

---

### 📊 Supporting Evidence

List the strongest factual evidence.

Use source citations such as:

[Source 1]

[Source 2]

Do not create a new ranking.

---

### 🔥 Engagement Drivers

Observed Pattern

State only what the data directly shows.

Possible Interpretation

Give cautious possible explanations.

Explicitly state that interpretation
is not proven causation.

---

### 💡 Content Opportunities

Give 2 to 3 practical opportunities.

Use cautious language such as:

- "This suggests an opportunity to..."
- "A potential direction is..."
- "The team could test..."

Do not present opportunities as guaranteed results.

---

### 🚀 Recommended Actions

Give exactly 3 actions:

1. Validate
2. Test
3. Scale

Do not aggressively recommend scaling
when evidence is limited.

---

### 📌 Decision Signal

Return exactly one:

STRONG SIGNAL

MODERATE SIGNAL

WEAK SIGNAL

Then briefly explain why.

Consider:

- sample size
- metric consistency
- missing metrics
- platform balance
- category balance
- causal limitations

---

### 🎯 Recommended Next Step

Give exactly ONE highest-priority executable action.

---

### ⚠️ Data Limitations

List only relevant limitations.

============================================================
AUTHORITATIVE PYTHON ANALYTICS
============================================================

{analytics}

============================================================
COMPACT RETRIEVED EVIDENCE
============================================================

{context}

============================================================
USER QUESTION
============================================================

{question}

============================================================
FINAL RULE
============================================================

Return ONLY the requested structured analysis.

Do not add greetings.

Do not add introductions.

Do not add extra sections.

Do not calculate new rankings.

Do not override Python Analytics.

Do not invent unsupported facts.

Do not claim causation from correlation.
"""


# ============================================================
# PROMPT BUILDER
# ============================================================

def get_prompt():
    """
    Return the compact structured prompt.
    """

    return ChatPromptTemplate.from_template(
        PROMPT_TEMPLATE
    )


# ============================================================
# LLM FACTORY
# ============================================================

def get_llm(
    api_key=None,
    model=DEFAULT_MODEL,
):
    """
    Create the Groq LLM.

    Lookup order:
    1. Explicit API key
    2. Environment variable
    3. Streamlit Secrets
    """

    # --------------------------------------------------------
    # Explicit key
    # --------------------------------------------------------

    api_key = (
        api_key
        or os.environ.get(
            "GROQ_API_KEY"
        )
    )

    # --------------------------------------------------------
    # Streamlit Secrets
    # --------------------------------------------------------

    if not api_key:

        try:

            import streamlit as st

            api_key = st.secrets.get(
                "GROQ_API_KEY"
            )

        except Exception:

            api_key = None

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not api_key:

        raise ValueError(
            "No GROQ_API_KEY found.\n"
            "Add GROQ_API_KEY to "
            ".streamlit/secrets.toml "
            "or set it as an environment variable."
        )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    return ChatGroq(
        model=model,
        groq_api_key=api_key,
        temperature=0,
        max_tokens=MAX_OUTPUT_TOKENS,
    )


# ============================================================
# SAFE TEXT
# ============================================================

def _safe_text(
    value,
):
    """
    Convert any value to safe compact text.
    """

    if value is None:
        return ""

    try:

        return str(
            value
        ).strip()

    except Exception:

        return ""


# ============================================================
# ANALYTICS FORMATTER
# ============================================================

def format_analytics_for_prompt(
    analytics_result,
):
    """
    Convert authoritative Python analytics
    into a compact, reliable prompt block.

    Important:

    - Python remains the source of truth.
    - Raw document items are not duplicated.
    - Critical analytics are preserved.
    - Large analytics payloads are truncated safely.
    """

    if not analytics_result:

        return (
            "No authoritative Python Analytics "
            "results are available.\n"
            "Do not create a ranking."
        )

    # --------------------------------------------------------
    # String analytics
    # --------------------------------------------------------

    if isinstance(
        analytics_result,
        str,
    ):

        text = (
            analytics_result
            .strip()
        )

        if len(text) > 6000:

            text = (
                text[:6000]
                + "\n[Analytics truncated]"
            )

        return text

    # --------------------------------------------------------
    # Dictionary analytics
    # --------------------------------------------------------

    if not isinstance(
        analytics_result,
        dict,
    ):

        result = _safe_text(
            analytics_result
        )

        if len(result) > 6000:

            result = (
                result[:6000]
                + "\n[Analytics truncated]"
            )

        return result

    parts = []

    # ========================================================
    # AUTHORITATIVE ANALYTICS ONLY
    # ========================================================

    important_sections = [

        "top_content",

        "ranked_items",

        "metric_leaders",

        "averages",

        "platform_summary",

        "category_summary",

        "limitations",

    ]

    for key in important_sections:

        if key not in analytics_result:

            continue

        value = analytics_result.get(
            key
        )

        if value is None:

            continue

        value_text = _safe_text(
            value
        )

        if not value_text:

            continue

        # ----------------------------------------------------
        # Section-specific limits
        # ----------------------------------------------------

        if key == "ranked_items":

            max_section_chars = 2500

        elif key == "metric_leaders":

            max_section_chars = 1200

        elif key in (
            "platform_summary",
            "category_summary",
        ):

            max_section_chars = 1500

        elif key == "limitations":

            max_section_chars = 1000

        else:

            max_section_chars = 1200

        if len(value_text) > max_section_chars:

            value_text = (
                value_text[
                    :max_section_chars
                ]
                + "\n[Section truncated]"
            )

        parts.append(

            f"{key.upper()}:\n"
            f"{value_text}"

        )

    # --------------------------------------------------------
    # No useful sections
    # --------------------------------------------------------

    if not parts:

        return (
            "No authoritative Python Analytics "
            "results are available.\n"
            "Do not create a ranking."
        )

    # --------------------------------------------------------
    # Final compact result
    # --------------------------------------------------------

    result = "\n\n".join(
        parts
    )

    if len(result) > 6000:

        result = (
            result[:6000]
            + "\n[Analytics truncated]"
        )

    return result


# ============================================================
# ANALYTICS RUNNER
# ============================================================

def run_python_analytics(
    docs,
):
    """
    Run the available Python analytics function.
    """

    if hasattr(
        _analytics,
        "analyze_documents",
    ):

        return _analytics.analyze_documents(
            docs
        )

    if hasattr(
        _analytics,
        "analyze_docs",
    ):

        return _analytics.analyze_docs(
            docs
        )

    if hasattr(
        _analytics,
        "run_analytics",
    ):

        return _analytics.run_analytics(
            docs
        )

    if hasattr(
        _analytics,
        "calculate_analytics",
    ):

        return _analytics.calculate_analytics(
            docs
        )

    if hasattr(
        _analytics,
        "analyze_retrieved_documents",
    ):

        return (
            _analytics
            .analyze_retrieved_documents(
                docs
            )
        )

    return (
        "Python analytics functions were not found "
        "in 08_analytics.py. "
        "Do not create unsupported rankings."
    )


# ============================================================
# COMPACT DOCUMENT EXTRACTOR
# ============================================================

def _extract_document_text(
    doc,
):
    """
    Safely extract page content
    from a LangChain Document.
    """

    if doc is None:

        return ""

    if hasattr(
        doc,
        "page_content",
    ):

        return _safe_text(
            doc.page_content
        )

    if isinstance(
        doc,
        dict,
    ):

        return _safe_text(

            doc.get(
                "page_content"
            )

            or doc.get(
                "content"
            )

            or doc.get(
                "text"
            )

        )

    return _safe_text(
        doc
    )


# ============================================================
# SOURCE METADATA
# ============================================================

def _get_metadata(
    doc,
):
    """
    Safely retrieve document metadata.
    """

    if doc is None:

        return {}

    if hasattr(
        doc,
        "metadata",
    ):

        metadata = (
            doc.metadata
        )

        if isinstance(
            metadata,
            dict,
        ):

            return metadata

    if isinstance(
        doc,
        dict,
    ):

        metadata = doc.get(
            "metadata",
            {},
        )

        if isinstance(
            metadata,
            dict,
        ):

            return metadata

    return {}


# ============================================================
# COMPACT EVIDENCE BUILDER
# ============================================================

def build_compact_evidence(
    docs,
    max_chars=6500,
):
    """
    Build a compact evidence layer.

    The UI still receives complete docs,
    but the LLM receives a compact version.

    This helps prevent Groq 413 / TPM errors.
    """

    if not docs:

        return (
            "No retrieved evidence is available."
        )

    chunks = []

    total_chars = 0

    for index, doc in enumerate(
        docs,
        start=1,
    ):

        metadata = _get_metadata(
            doc
        )

        content = (
            _extract_document_text(
                doc
            )
        )

        # ----------------------------------------------------
        # Extract useful metadata
        # ----------------------------------------------------

        platform = (
            metadata.get(
                "platform"
            )
            or "Unknown"
        )

        category = (
            metadata.get(
                "category"
            )
            or "Unknown"
        )

        content_id = (

            metadata.get(
                "content_id"
            )

            or metadata.get(
                "id"
            )

            or "Unknown"

        )

        # ----------------------------------------------------
        # Compact content
        # ----------------------------------------------------

        if len(content) > 1000:

            content = (
                content[:1000]
                + "..."
            )

        block = (

            f"[Source {index}]\n"
            f"Content ID: {content_id}\n"
            f"Platform: {platform}\n"
            f"Category: {category}\n"
            f"Evidence: {content}"

        )

        remaining = (
            max_chars
            - total_chars
        )

        if remaining <= 0:

            break

        if len(block) > remaining:

            block = block[
                :remaining
            ]

        chunks.append(
            block
        )

        total_chars += len(
            block
        )

        if total_chars >= max_chars:

            break

    return "\n\n".join(
        chunks
    )


# ============================================================
# SOURCE COUNT
# ============================================================

def get_source_count(
    docs,
):
    """
    Safely return source count.
    """

    if docs is None:

        return 0

    try:

        return len(
            docs
        )

    except Exception:

        return 0


# ============================================================
# API ERROR DETECTION
# ============================================================

def is_request_too_large_error(
    error,
):
    """
    Detect Groq 413 / TPM request-too-large errors.
    """

    message = str(
        error
    ).lower()

    return (

        "413" in message

        or "request too large"
        in message

        or "tokens per minute"
        in message

        or "rate_limit_exceeded"
        in message

    )


# ============================================================
# SAFE FALLBACK ANSWER
# ============================================================

def build_fallback_answer(
    analytics_result,
    docs,
):
    """
    Return a safe answer if Groq cannot process the request.

    This prevents Streamlit from crashing.
    """

    source_count = (
        get_source_count(
            docs
        )
    )

    analytics_text = (
        format_analytics_for_prompt(
            analytics_result
        )
    )

    # --------------------------------------------------------
    # Extract top performer if available
    # --------------------------------------------------------

    top_content = None

    if isinstance(
        analytics_result,
        dict,
    ):

        top_content = (
            analytics_result.get(
                "top_content"
            )
        )

    # --------------------------------------------------------
    # Build key insight
    # --------------------------------------------------------

    if top_content:

        top_content_id = (
            top_content.get(
                "content_id",
                "Unknown",
            )
        )

        top_score = (
            top_content.get(
                "engagement_score",
                "Unknown",
            )
        )

        top_platform = (
            top_content.get(
                "platform",
                "Unknown",
            )
        )

        top_category = (
            top_content.get(
                "category",
                "Unknown",
            )
        )

        top_source = (
            top_content.get(
                "source_label",
                "",
            )
        )

        key_insight = (

            f"The authoritative Python Analytics "
            f"identify {top_content_id} as the "
            f"top-performing content by Engagement Score "
            f"({top_score}). Platform: {top_platform}. "
            f"Category: {top_category}. "
            f"{top_source}"

        )

    else:

        key_insight = (

            "The authoritative Python Analytics "
            "results are available, but a top-performing "
            "content item could not be determined."

        )

    return (

        "### 🎯 Key Insight\n\n"

        f"{key_insight}\n\n"

        "---\n\n"

        "### 📊 Supporting Evidence\n\n"

        "Python Analytics results were generated "
        "successfully. "

        f"{source_count} retrieved sources are "
        "available in the Evidence Layer.\n\n"

        "---\n\n"

        "### 🔥 Engagement Drivers\n\n"

        "Observed Pattern\n\n"

        "The available pattern should be interpreted "
        "directly from the authoritative Python "
        "Analytics results.\n\n"

        "Possible Interpretation\n\n"

        "No additional AI interpretation was generated "
        "because the model request exceeded the available "
        "token limit. Any interpretation would require "
        "further evidence and is not a causal conclusion.\n\n"

        "---\n\n"

        "### 💡 Content Opportunities\n\n"

        "The team could validate the strongest "
        "analytics pattern with additional data.\n\n"

        "A potential direction is to test similar "
        "content patterns with a larger sample.\n\n"

        "The team could compare results across "
        "additional platforms or categories.\n\n"

        "---\n\n"

        "### 🚀 Recommended Actions\n\n"

        "1. Validate: Review the authoritative "
        "Python Analytics results.\n"

        "2. Test: Validate the strongest pattern "
        "with additional data.\n"

        "3. Scale: Scale only after validation "
        "confirms the pattern.\n\n"

        "---\n\n"

        "### 📌 Decision Signal\n\n"

        "WEAK SIGNAL\n\n"

        "The AI interpretation was not generated "
        "because the model request exceeded the "
        "available token limit. The Python Analytics "
        "remain authoritative.\n\n"

        "---\n\n"

        "### 🎯 Recommended Next Step\n\n"

        "Reduce the retrieved evidence size and "
        "rerun the analysis using the authoritative "
        "Python Analytics results.\n\n"

        "---\n\n"

        "### ⚠️ Data Limitations\n\n"

        f"- Retrieved sources: {source_count}\n"

        "- AI interpretation was blocked by the "
        "model token limit.\n"

        "- Python Analytics remains authoritative.\n"

        f"- Analytics summary available: "
        f"{bool(analytics_text)}\n"

    )


# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_answer(
    query,
    vectorstore,
    llm,
    k=5,
):
    """
    Complete RAG + Analytics + Prompting pipeline.

    Architecture:

    Full Documents
        |
        +--> UI Evidence Layer
        |
        +--> Python Analytics
        |
        +--> Compact Analytics
        |
        +--> Compact Evidence
        |
        +--> LLM

    This prevents large prompts while preserving
    complete evidence for the UI.
    """

    # --------------------------------------------------------
    # Validate query
    # --------------------------------------------------------

    if not query or not query.strip():

        return {

            "answer":
                "Please enter a marketing question.",

            "sources":
                [],

            "analytics":
                None,

            "source_count":
                0,

        }

    # --------------------------------------------------------
    # Retrieve documents
    # --------------------------------------------------------

    docs = retrieve_context(

        vectorstore,

        query,

        k=k,

    )

    # --------------------------------------------------------
    # Run Python Analytics
    # --------------------------------------------------------

    analytics_result = (
        run_python_analytics(
            docs
        )
    )

    # --------------------------------------------------------
    # Compact Analytics
    # --------------------------------------------------------

    analytics_text = (
        format_analytics_for_prompt(
            analytics_result
        )
    )

    # --------------------------------------------------------
    # Compact Evidence
    # --------------------------------------------------------

    compact_context = (
        build_compact_evidence(
            docs,
            max_chars=6500,
        )
    )

    # --------------------------------------------------------
    # Build Prompt
    # --------------------------------------------------------

    prompt = get_prompt()

    chain = (

        prompt

        | llm

        | StrOutputParser()

    )

    # --------------------------------------------------------
    # Generate Answer
    # --------------------------------------------------------

    try:

        answer = chain.invoke(

            {

                "analytics":
                    analytics_text,

                "context":
                    compact_context,

                "question":
                    query.strip(),

            }

        )

    except Exception as error:

        # ----------------------------------------------------
        # Handle Groq request-too-large error
        # ----------------------------------------------------

        if is_request_too_large_error(
            error
        ):

            # ------------------------------------------------
            # Retry with smaller evidence
            # ------------------------------------------------

            smaller_context = (
                build_compact_evidence(
                    docs,
                    max_chars=2500,
                )
            )

            smaller_analytics = (
                analytics_text[:3000]
            )

            try:

                answer = chain.invoke(

                    {

                        "analytics":
                            smaller_analytics,

                        "context":
                            smaller_context,

                        "question":
                            query.strip(),

                    }

                )

            except Exception:

                answer = (
                    build_fallback_answer(
                        analytics_result,
                        docs,
                    )
                )

        else:

            # ------------------------------------------------
            # Unexpected error
            # ------------------------------------------------

            answer = (

                "### ❌ Analysis Error\n\n"

                "The analysis could not be generated.\n\n"

                f"Technical details: {str(error)}"

            )

    # --------------------------------------------------------
    # Return complete pipeline result
    # --------------------------------------------------------

    return {

        "answer":
            answer,

        "sources":
            docs,

        "analytics":
            analytics_result,

        "source_count":
            get_source_count(
                docs
            ),

    }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    print(

        "\n"
        + "=" * 60

    )

    print(
        "MARKETPULSE AI - PROMPTING PIPELINE TEST"
    )

    print(

        "=" * 60

    )

    print()

    # --------------------------------------------------------
    # Load Vector Store
    # --------------------------------------------------------

    vectorstore = (
        load_vectorstore()
    )

    # --------------------------------------------------------
    # Load LLM
    # --------------------------------------------------------

    llm = (
        get_llm()
    )

    # --------------------------------------------------------
    # Test Query
    # --------------------------------------------------------

    query = (

        "Which content performs best "
        "based on engagement?"

    )

    # --------------------------------------------------------
    # Generate Answer
    # --------------------------------------------------------

    result = generate_answer(

        query,

        vectorstore,

        llm,

        k=5,

    )

    # --------------------------------------------------------
    # Print Query
    # --------------------------------------------------------

    print(

        f"Query:\n{query}\n"

    )

    # --------------------------------------------------------
    # Print Analytics
    # --------------------------------------------------------

    print(

        "\n"
        + "=" * 60

    )

    print(
        "PYTHON ANALYTICS"
    )

    print(

        "=" * 60

    )

    print()

    print(

        format_analytics_for_prompt(

            result.get(
                "analytics"
            )

        )

    )

    # --------------------------------------------------------
    # Print AI Answer
    # --------------------------------------------------------

    print(

        "\n"
        + "=" * 60

    )

    print(
        "AI ANSWER"
    )

    print(

        "=" * 60

    )

    print()

    print(

        result.get(
            "answer",
            "",
        )

    )

    # --------------------------------------------------------
    # Print Sources
    # --------------------------------------------------------

    print(

        "\n"
        + "=" * 60

    )

    print(

        f"Sources used: "
        f"{result.get('source_count', 0)}"

    )

    print(

        "=" * 60

    )

    print()