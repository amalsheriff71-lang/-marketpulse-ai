"""
07_prompting.py
-----------------
Pipeline stage 7: PROMPTING + DETERMINISTIC ANALYTICS

MarketPulse AI

Architecture:
1. Retrieve relevant documents.
2. Parse measurable engagement metrics with Python.
3. Calculate deterministic engagement scores.
4. Rank content using Python.
5. Build an analytics summary.
6. Send verified calculations + retrieved evidence to the LLM.
7. Let the LLM interpret the evidence and generate recommendations.

Important:
- Python is responsible for calculations and rankings.
- The LLM is responsible for interpretation and recommendations.
- The LLM must never recalculate or override Python results.
- All factual claims must remain grounded in retrieved sources.
- Raw sources are returned to the Streamlit UI.
"""

import importlib.util
import os
import re
from typing import Any, Dict, List, Optional


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
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
    module_filename: str,
    module_name: str,
):
    """
    Dynamically import another pipeline module.
    """

    path = os.path.join(
        BASE_DIR,
        module_filename,
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load module: {module_filename}"
        )

    mod = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        mod
    )

    return mod


# ============================================================
# LOAD RETRIEVAL MODULE
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
# MODEL CONFIG
# ============================================================

DEFAULT_MODEL = (
    "llama-3.1-8b-instant"
)


# ============================================================
# ENGAGEMENT METRIC CONFIG
# ============================================================

ENGAGEMENT_FIELDS = [
    "likes",
    "comments",
    "shares",
]


# ============================================================
# METRIC EXTRACTION
# ============================================================

def extract_engagement_metrics(
    text: str,
) -> Dict[str, Optional[int]]:
    """
    Extract engagement metrics from document text.

    Expected source format:

    Engagement:
    1551 Likes,
    199 Comments,
    310 Shares,
    10106 Views

    Returns:

    {
        "likes": 1551,
        "comments": 199,
        "shares": 310,
        "views": 10106,
    }

    Missing values are returned as None.
    """

    if not text:
        return {
            "likes": None,
            "comments": None,
            "shares": None,
            "views": None,
        }

    text = str(text)

    patterns = {
        "likes": r"([\d,]+)\s+Likes?",
        "comments": r"([\d,]+)\s+Comments?",
        "shares": r"([\d,]+)\s+Shares?",
        "views": r"([\d,]+)\s+Views?",
    }

    metrics = {}

    for metric_name, pattern in patterns.items():

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:

            raw_value = (
                match.group(1)
                .replace(",", "")
                .strip()
            )

            try:

                metrics[metric_name] = int(
                    raw_value
                )

            except ValueError:

                metrics[metric_name] = None

        else:

            metrics[metric_name] = None

    return metrics


# ============================================================
# ENGAGEMENT SCORE
# ============================================================

def calculate_engagement_score(
    metrics: Dict[str, Optional[int]],
) -> Optional[int]:
    """
    Calculate a deterministic engagement score.

    Engagement Score =
    Likes + Comments + Shares

    Views are intentionally excluded from the score
    because views represent reach/exposure rather than
    direct engagement actions.
    """

    values = []

    for field in ENGAGEMENT_FIELDS:

        value = metrics.get(
            field
        )

        if value is not None:

            values.append(
                value
            )

    if not values:

        return None

    return sum(
        values
    )


# ============================================================
# DOCUMENT ANALYTICS
# ============================================================

def analyze_documents(
    docs: list,
) -> List[Dict[str, Any]]:
    """
    Parse retrieved documents and calculate
    deterministic engagement metrics.

    Python performs all calculations here.

    The LLM does NOT calculate rankings.
    """

    analyzed = []

    for index, doc in enumerate(
        docs,
        start=1,
    ):

        metadata = (
            getattr(
                doc,
                "metadata",
                {},
            )
            or {}
        )

        page_content = getattr(
            doc,
            "page_content",
            "",
        )

        metrics = extract_engagement_metrics(
            page_content
        )

        engagement_score = (
            calculate_engagement_score(
                metrics
            )
        )

        platform = metadata.get(
            "platform",
            "Unknown",
        )

        category = metadata.get(
            "category",
            "Unknown",
        )

        content_id = metadata.get(
            "content_id",
            f"Source_{index}",
        )

        analyzed.append(
            {
                "source_number": index,
                "content_id": content_id,
                "platform": platform,
                "category": category,
                "likes": metrics.get(
                    "likes"
                ),
                "comments": metrics.get(
                    "comments"
                ),
                "shares": metrics.get(
                    "shares"
                ),
                "views": metrics.get(
                    "views"
                ),
                "engagement_score": (
                    engagement_score
                ),
            }
        )

    return analyzed


# ============================================================
# SORT CONTENT
# ============================================================

def rank_content(
    analyzed_docs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Rank content deterministically by engagement score.

    Secondary ranking:
    - Likes
    - Comments
    - Shares

    Content with missing engagement score
    is placed at the bottom.
    """

    return sorted(
        analyzed_docs,
        key=lambda item: (
            item.get(
                "engagement_score"
            )
            if item.get(
                "engagement_score"
            )
            is not None
            else -1,

            item.get(
                "likes"
            )
            if item.get(
                "likes"
            )
            is not None
            else -1,

            item.get(
                "comments"
            )
            if item.get(
                "comments"
            )
            is not None
            else -1,

            item.get(
                "shares"
            )
            if item.get(
                "shares"
            )
            is not None
            else -1,
        ),
        reverse=True,
    )


# ============================================================
# BUILD ANALYTICS SUMMARY
# ============================================================

def build_analytics_summary(
    analyzed_docs: List[Dict[str, Any]],
) -> str:
    """
    Build a deterministic analytics summary for the LLM.

    All rankings and numerical conclusions are calculated
    by Python before the prompt reaches the LLM.
    """

    if not analyzed_docs:

        return (
            "No measurable engagement data was found "
            "in the retrieved sources."
        )

    ranked = rank_content(
        analyzed_docs
    )

    lines = []

    lines.append(
        "=== VERIFIED PYTHON ANALYTICS ==="
    )

    lines.append(
        "The following calculations were performed "
        "deterministically by Python."
    )

    lines.append(
        "The LLM MUST NOT recalculate, change, "
        "or override these rankings."
    )

    lines.append("")

    # --------------------------------------------------------
    # TOP CONTENT
    # --------------------------------------------------------

    ranked_with_scores = [
        item
        for item in ranked
        if item.get(
            "engagement_score"
        )
        is not None
    ]

    if ranked_with_scores:

        top = ranked_with_scores[0]

        lines.append(
            "TOP PERFORMING CONTENT "
            "(by Engagement Score):"
        )

        lines.append(
            f"- Content ID: "
            f"{top['content_id']}"
        )

        lines.append(
            f"- Source: "
            f"[Source {top['source_number']}]"
        )

        lines.append(
            f"- Platform: "
            f"{top['platform']}"
        )

        lines.append(
            f"- Category: "
            f"{top['category']}"
        )

        lines.append(
            f"- Likes: "
            f"{top['likes']}"
        )

        lines.append(
            f"- Comments: "
            f"{top['comments']}"
        )

        lines.append(
            f"- Shares: "
            f"{top['shares']}"
        )

        lines.append(
            f"- Engagement Score: "
            f"{top['engagement_score']}"
        )

        if top.get(
            "views"
        ) is not None:

            lines.append(
                f"- Views: "
                f"{top['views']}"
            )

        lines.append("")

    # --------------------------------------------------------
    # RANKING TABLE
    # --------------------------------------------------------

    lines.append(
        "CONTENT RANKING:"
    )

    for rank, item in enumerate(
        ranked,
        start=1,
    ):

        score = item.get(
            "engagement_score"
        )

        score_text = (
            str(score)
            if score is not None
            else "N/A"
        )

        lines.append(
            f"{rank}. "
            f"{item['content_id']} | "
            f"{item['platform']} | "
            f"{item['category']} | "
            f"Score: {score_text} | "
            f"Likes: {item['likes']} | "
            f"Comments: {item['comments']} | "
            f"Shares: {item['shares']} | "
            f"Views: {item['views']} | "
            f"[Source {item['source_number']}]"
        )

    lines.append("")

    # --------------------------------------------------------
    # PLATFORM SUMMARY
    # --------------------------------------------------------

    platform_groups = {}

    for item in analyzed_docs:

        score = item.get(
            "engagement_score"
        )

        platform = item.get(
            "platform",
            "Unknown",
        )

        if score is None:
            continue

        platform_groups.setdefault(
            platform,
            [],
        ).append(
            score
        )

    if platform_groups:

        lines.append(
            "PLATFORM ENGAGEMENT SUMMARY:"
        )

        platform_averages = []

        for platform, scores in (
            platform_groups.items()
        ):

            average_score = (
                sum(scores)
                / len(scores)
            )

            platform_averages.append(
                (
                    platform,
                    average_score,
                    len(scores),
                )
            )

        platform_averages.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        for (
            platform,
            average_score,
            count,
        ) in platform_averages:

            lines.append(
                f"- {platform}: "
                f"Average Engagement Score = "
                f"{average_score:.2f} "
                f"across {count} content item(s)."
            )

        lines.append("")

    # --------------------------------------------------------
    # CATEGORY SUMMARY
    # --------------------------------------------------------

    category_groups = {}

    for item in analyzed_docs:

        score = item.get(
            "engagement_score"
        )

        category = item.get(
            "category",
            "Unknown",
        )

        if score is None:
            continue

        category_groups.setdefault(
            category,
            [],
        ).append(
            score
        )

    if category_groups:

        lines.append(
            "CATEGORY ENGAGEMENT SUMMARY:"
        )

        category_averages = []

        for category, scores in (
            category_groups.items()
        ):

            average_score = (
                sum(scores)
                / len(scores)
            )

            category_averages.append(
                (
                    category,
                    average_score,
                    len(scores),
                )
            )

        category_averages.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        for (
            category,
            average_score,
            count,
        ) in category_averages:

            lines.append(
                f"- {category}: "
                f"Average Engagement Score = "
                f"{average_score:.2f} "
                f"across {count} content item(s)."
            )

        lines.append("")

    # --------------------------------------------------------
    # METRIC LEADERS
    # --------------------------------------------------------

    lines.append(
        "METRIC LEADERS:"
    )

    for metric in [
        "likes",
        "comments",
        "shares",
        "views",
    ]:

        available = [
            item
            for item in analyzed_docs
            if item.get(
                metric
            )
            is not None
        ]

        if not available:
            continue

        leader = max(
            available,
            key=lambda item: item.get(
                metric
            ),
        )

        lines.append(
            f"- Highest {metric.title()}: "
            f"{leader['content_id']} "
            f"with {leader[metric]} "
            f"[Source {leader['source_number']}]"
        )

    lines.append("")

    lines.append(
        "=== END VERIFIED PYTHON ANALYTICS ==="
    )

    return "\n".join(
        lines
    )


# ============================================================
# PROMPT TEMPLATE
# ============================================================

PROMPT_TEMPLATE = """
You are MarketPulse AI, an expert Senior Marketing Intelligence Analyst.

Your job is to transform retrieved social media marketing data
into clear, professional, decision-oriented business intelligence.

You are working inside a RAG system.

IMPORTANT:

Python has already performed all numerical calculations,
engagement scoring, rankings, platform summaries,
category summaries, and metric leader calculations.

You MUST trust the VERIFIED PYTHON ANALYTICS section.

You MUST NOT:
- Recalculate engagement scores.
- Re-rank content.
- Change the top-performing content.
- Invent missing metrics.
- Treat views as engagement.
- Claim causation without evidence.
- Use information outside the retrieved context.

You MAY:
- Interpret the verified patterns.
- Explain possible reasons.
- Identify opportunities.
- Recommend actions.
- Highlight limitations.
- Suggest additional validation.

============================================================
SOURCE RULES
============================================================

1. Use ONLY retrieved sources and verified Python analytics.
2. Never invent statistics, percentages, trends, or facts.
3. Every factual claim must include a source citation.
4. Use citations exactly like [Source 1], [Source 2].
5. When discussing the Python-calculated ranking,
   cite the source attached to the relevant content.
6. Clearly distinguish:
   - Verified fact
   - Observed pattern
   - Possible interpretation
   - Recommendation
7. Never claim that correlation proves causation.
8. If the evidence is insufficient, say so clearly.
9. Do not pretend that a small retrieved sample represents
   the entire dataset.
10. Do not introduce external marketing knowledge as if it
    came from the retrieved data.

============================================================
ENGAGEMENT SCORE DEFINITION
============================================================

Engagement Score is defined by Python as:

Likes + Comments + Shares

Views are NOT included in Engagement Score.

Views represent exposure/reach and should be discussed
separately from direct engagement.

============================================================
REQUIRED RESPONSE STRUCTURE
============================================================

### 🎯 Key Insight

State the single most important verified finding
that directly answers the user's question.

If the question asks which content performs best based
on engagement, use the Python-ranked top content.

Do not choose a different winner.

### 📊 Supporting Evidence

List the strongest factual evidence.

Every factual bullet must include a source citation.

Use the verified Python analytics when discussing rankings
and calculated engagement scores.

### 🔥 Engagement Drivers

Separate the answer into:

**Observed Pattern**

Describe only patterns directly supported by the data.

**Possible Interpretation**

Explain plausible interpretations,
but clearly label them as interpretations,
not proven causal facts.

### 💡 Content Opportunities

Suggest practical opportunities based on
the observed evidence.

Clearly distinguish opportunities
from proven facts.

### 🚀 Recommended Actions

Provide 3 to 5 specific actions.

Actions should be:
- Practical
- Specific
- Derived from the evidence
- Suitable for a marketer

### 📌 Decision Signal

Choose one:

**STRONG SIGNAL**
Use only when the retrieved evidence is consistent
and sufficiently large.

**MODERATE SIGNAL**
Use when the evidence suggests a meaningful pattern
but additional validation is needed.

**WEAK SIGNAL**
Use when evidence is limited or inconsistent.

Briefly explain why.

### 🎯 Recommended Next Step

Give ONE highest-priority next step.

It must directly follow from the evidence.

### ⚠️ Data Limitations

Mention limitations such as:
- Small retrieved sample
- Missing metrics
- Missing audience data
- Missing platform coverage
- Missing category coverage
- Lack of causal evidence

Only mention limitations that actually apply.

============================================================
VERIFIED PYTHON ANALYTICS
============================================================

{analytics}

============================================================
RETRIEVED CONTEXT
============================================================

{context}

============================================================
USER QUESTION
============================================================

{question}

Return ONLY the structured analysis.
"""


# ============================================================
# PROMPT BUILDER
# ============================================================

def get_prompt():

    return ChatPromptTemplate.from_template(
        PROMPT_TEMPLATE
    )


# ============================================================
# LLM
# ============================================================

def get_llm(
    api_key: str = None,
    model: str = DEFAULT_MODEL,
) -> ChatGroq:
    """
    Initialize Groq LLM.
    """

    api_key = (
        api_key
        or os.environ.get(
            "GROQ_API_KEY"
        )
    )

    if not api_key:

        raise ValueError(
            "No GROQ_API_KEY found."
        )

    return ChatGroq(
        model=model,
        groq_api_key=api_key,
        temperature=0,
    )


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    query: str,
    vectorstore,
    llm,
    k: int = 5,
) -> dict:
    """
    Main MarketPulse AI pipeline.

    Flow:

    User Query
        ↓
    Retrieve Documents
        ↓
    Parse Metrics
        ↓
    Python Analytics
        ↓
    Verified Analytics Summary
        ↓
    LLM Interpretation
        ↓
    Structured Answer
    """

    if not query or not query.strip():

        return {
            "answer": (
                "Please enter a marketing question."
            ),
            "sources": [],
        }

    # --------------------------------------------------------
    # STEP 1: RETRIEVE
    # --------------------------------------------------------

    docs = retrieve_context(
        vectorstore,
        query,
        k=k,
    )

    # --------------------------------------------------------
    # STEP 2: ANALYZE WITH PYTHON
    # --------------------------------------------------------

    analyzed_docs = analyze_documents(
        docs
    )

    # --------------------------------------------------------
    # STEP 3: BUILD VERIFIED ANALYTICS
    # --------------------------------------------------------

    analytics = build_analytics_summary(
        analyzed_docs
    )

    # --------------------------------------------------------
    # STEP 4: FORMAT ORIGINAL CONTEXT
    # --------------------------------------------------------

    context = format_context(
        docs
    )

    # --------------------------------------------------------
    # STEP 5: BUILD LLM CHAIN
    # --------------------------------------------------------

    chain = (
        {
            "analytics": lambda _: analytics,
            "context": lambda _: context,
            "question": RunnablePassthrough(),
        }
        | get_prompt()
        | llm
        | StrOutputParser()
    )

    # --------------------------------------------------------
    # STEP 6: GENERATE INTERPRETATION
    # --------------------------------------------------------

    answer = chain.invoke(
        query
    )

    # --------------------------------------------------------
    # STEP 7: RETURN RESULT
    # --------------------------------------------------------

    return {
        "answer": answer,
        "sources": docs,
        "analytics": analyzed_docs,
    }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    vectorstore = load_vectorstore()

    llm = get_llm()

    query = (
        "Which content performs best "
        "based on engagement?"
    )

    result = generate_answer(
        query,
        vectorstore,
        llm,
    )

    print(
        f"Query: {query}\n"
    )

    print(
        "========================================"
    )

    print(
        "VERIFIED ANALYTICS"
    )

    print(
        "========================================"
    )

    print(
        build_analytics_summary(
            result.get(
                "analytics",
                [],
            )
        )
    )

    print(
        "\n========================================"
    )

    print(
        "AI ANSWER"
    )

    print(
        "========================================"
    )

    print(
        result["answer"]
    )

    print(
        "\n========================================"
    )

    print(
        "SOURCES USED"
    )

    print(
        "========================================"
    )

    print(
        len(
            result["sources"]
        )
    )