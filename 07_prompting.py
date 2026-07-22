"""
07_prompting.py
-----------------
Pipeline stage 7: PROMPTING + VERIFIED ANALYTICS

MarketPulse AI
Premium AI-powered Marketing Intelligence

Pipeline:
1. Retrieve relevant documents from Chroma.
2. Extract measurable metrics using Python.
3. Calculate verified metric leaders.
4. Build a verified analytics summary.
5. Send verified evidence to the LLM.
6. Let the LLM interpret the evidence and generate recommendations.

Important:
- Python handles numerical comparisons.
- The LLM handles interpretation and marketing intelligence.
- No invented engagement scores.
- Views are separated from engagement metrics.
- Source attribution is preserved.
"""

import importlib.util
import os
import re
from statistics import mean

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
    module_filename,
    module_name,
):

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
# RETRIEVAL MODULE
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
# DEFAULT MODEL
# ============================================================

DEFAULT_MODEL = (
    "llama-3.1-8b-instant"
)


# ============================================================
# METRIC EXTRACTION HELPERS
# ============================================================

def _extract_number(
    text,
    label,
):
    """
    Extract a numeric metric from retrieved content.

    Example:
    Likes: 1551
    Views: 10106
    """

    if not text:

        return None

    pattern = (
        rf"{label}\s*:\s*"
        rf"([\d,]+)"
    )

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE,
    )

    if not match:

        return None

    value = (
        match.group(1)
        .replace(",", "")
    )

    try:

        return int(value)

    except ValueError:

        return None


# ============================================================
# EXTRACT DOCUMENT METRICS
# ============================================================

def extract_document_metrics(
    doc,
    source_number,
):
    """
    Extract structured marketing metrics
    from a retrieved LangChain document.
    """

    metadata = (
        getattr(
            doc,
            "metadata",
            {}
        )
        or {}
    )

    content = (
        getattr(
            doc,
            "page_content",
            ""
        )
        or ""
    )


    # --------------------------------------------------------
    # CONTENT ID
    # --------------------------------------------------------

    content_id = metadata.get(
        "content_id"
    )

    if not content:

        match = re.search(
            r"Content\s*ID\s*:\s*([A-Za-z0-9_-]+)",
            content,
            flags=re.IGNORECASE,
        )

        if match:

            content_id = (
                match.group(1)
            )

    if not content_id:

        content_id = (
            f"Unknown Content {source_number}"
        )


    # --------------------------------------------------------
    # PLATFORM
    # --------------------------------------------------------

    platform = metadata.get(
        "platform"
    )

    if not platform:

        match = re.search(
            r"Platform\s*:\s*([^\n]+)",
            content,
            flags=re.IGNORECASE,
        )

        if match:

            platform = (
                match.group(1)
                .strip()
            )

    if not platform:

        platform = "Unknown"


    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    category = metadata.get(
        "category"
    )

    if not category:

        match = re.search(
            r"Category\s*:\s*([^\n]+)",
            content,
            flags=re.IGNORECASE,
        )

        if match:

            category = (
                match.group(1)
                .strip()
            )

    if not category:

        category = "Unknown"


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    views = _extract_number(
        content,
        "Views",
    )

    likes = _extract_number(
        content,
        "Likes",
    )

    comments = _extract_number(
        content,
        "Comments",
    )

    shares = _extract_number(
        content,
        "Shares",
    )


    return {

        "source": source_number,

        "content_id": content_id,

        "platform": platform,

        "category": category,

        "views": views,

        "likes": likes,

        "comments": comments,

        "shares": shares,

    }


# ============================================================
# BUILD VERIFIED ANALYTICS
# ============================================================

def build_verified_analytics(
    docs,
):
    """
    Calculate verified rankings using Python.

    The LLM does NOT calculate these values.

    This prevents errors such as:
    claiming 10,106 Views is higher than 10,164 Views.
    """

    records = []

    for index, doc in enumerate(
        docs,
        start=1,
    ):

        record = extract_document_metrics(
            doc,
            index,
        )

        records.append(
            record
        )


    # --------------------------------------------------------
    # METRIC LEADER HELPER
    # --------------------------------------------------------

    def get_leader(
        metric_name,
    ):

        valid_records = [

            record

            for record in records

            if record.get(
                metric_name
            ) is not None

        ]

        if not valid_records:

            return None

        return max(
            valid_records,
            key=lambda x: x[
                metric_name
            ],
        )


    # --------------------------------------------------------
    # LEADERS
    # --------------------------------------------------------

    highest_views = get_leader(
        "views"
    )

    highest_likes = get_leader(
        "likes"
    )

    highest_comments = get_leader(
        "comments"
    )

    highest_shares = get_leader(
        "shares"
    )


    # --------------------------------------------------------
    # AVERAGES
    # --------------------------------------------------------

    def calculate_average(
        metric_name,
    ):

        values = [

            record[metric_name]

            for record in records

            if record.get(
                metric_name
            ) is not None

        ]

        if not values:

            return None

        return round(
            mean(values),
            2,
        )


    average_views = calculate_average(
        "views"
    )

    average_likes = calculate_average(
        "likes"
    )

    average_comments = calculate_average(
        "comments"
    )

    average_shares = calculate_average(
        "shares"
    )


    # --------------------------------------------------------
    # BUILD VERIFIED SUMMARY
    # --------------------------------------------------------

    lines = []

    lines.append(
        "VERIFIED ANALYTICS SUMMARY"
    )

    lines.append(
        "The following values were calculated "
        "programmatically from the retrieved documents."
    )

    lines.append(
        "The LLM must treat these values as verified "
        "for the retrieved sample."
    )

    lines.append("")


    # --------------------------------------------------------
    # DATASET SIZE
    # --------------------------------------------------------

    lines.append(
        f"Retrieved content items: {len(records)}"
    )

    lines.append("")


    # --------------------------------------------------------
    # HIGHEST VIEWS
    # --------------------------------------------------------

    if highest_views:

        lines.append(
            "Highest Views:"
        )

        lines.append(
            f"- {highest_views['content_id']} "
            f"with {highest_views['views']:,} Views "
            f"[Source {highest_views['source']}]"
        )

    else:

        lines.append(
            "Highest Views: Not available"
        )


    # --------------------------------------------------------
    # HIGHEST LIKES
    # --------------------------------------------------------

    if highest_likes:

        lines.append(
            "Highest Likes:"
        )

        lines.append(
            f"- {highest_likes['content_id']} "
            f"with {highest_likes['likes']:,} Likes "
            f"[Source {highest_likes['source']}]"
        )

    else:

        lines.append(
            "Highest Likes: Not available"
        )


    # --------------------------------------------------------
    # HIGHEST COMMENTS
    # --------------------------------------------------------

    if highest_comments:

        lines.append(
            "Highest Comments:"
        )

        lines.append(
            f"- {highest_comments['content_id']} "
            f"with {highest_comments['comments']:,} Comments "
            f"[Source {highest_comments['source']}]"
        )

    else:

        lines.append(
            "Highest Comments: Not available"
        )


    # --------------------------------------------------------
    # HIGHEST SHARES
    # --------------------------------------------------------

    if highest_shares:

        lines.append(
            "Highest Shares:"
        )

        lines.append(
            f"- {highest_shares['content_id']} "
            f"with {highest_shares['shares']:,} Shares "
            f"[Source {highest_shares['source']}]"
        )

    else:

        lines.append(
            "Highest Shares: Not available"
        )


    # --------------------------------------------------------
    # AVERAGES
    # --------------------------------------------------------

    lines.append("")

    lines.append(
        "Retrieved Sample Averages:"
    )


    if average_views is not None:

        lines.append(
            f"- Average Views: "
            f"{average_views:,}"
        )


    if average_likes is not None:

        lines.append(
            f"- Average Likes: "
            f"{average_likes:,}"
        )


    if average_comments is not None:

        lines.append(
            f"- Average Comments: "
            f"{average_comments:,}"
        )


    if average_shares is not None:

        lines.append(
            f"- Average Shares: "
            f"{average_shares:,}"
        )


    # --------------------------------------------------------
    # CONTENT RECORDS
    # --------------------------------------------------------

    lines.append("")

    lines.append(
        "Verified Content Records:"
    )


    for record in records:

        metrics = []

        if record["views"] is not None:

            metrics.append(
                f"Views={record['views']:,}"
            )

        if record["likes"] is not None:

            metrics.append(
                f"Likes={record['likes']:,}"
            )

        if record["comments"] is not None:

            metrics.append(
                f"Comments={record['comments']:,}"
            )

        if record["shares"] is not None:

            metrics.append(
                f"Shares={record['shares']:,}"
            )


        metric_text = (
            ", ".join(metrics)
            if metrics
            else "No measurable metrics"
        )


        lines.append(

            f"- {record['content_id']} | "
            f"Platform={record['platform']} | "
            f"Category={record['category']} | "
            f"{metric_text} | "
            f"[Source {record['source']}]"

        )


    return "\n".join(
        lines
    )


# ============================================================
# PROMPT TEMPLATE
# ============================================================

PROMPT_TEMPLATE = """
You are MarketPulse AI, an expert Senior Marketing Intelligence Analyst.

Your job is to transform verified marketing data into clear,
accurate, actionable business intelligence.

You are an evidence-first marketing intelligence system.

============================================================
CRITICAL RULE
============================================================

Python has already calculated the numerical rankings.

DO NOT recalculate or override the verified analytics.

Use the VERIFIED ANALYTICS SUMMARY as the authoritative source
for numerical rankings.

The verified analytics were calculated programmatically.

You must NOT contradict them.

============================================================
DATA RULES
============================================================

1. Use ONLY the retrieved context and verified analytics.

2. Never invent statistics, percentages, rankings, trends,
   audience characteristics, or business facts.

3. Every factual claim must include source citations.

4. Use citations exactly like:

   [Source 1]

   [Source 2]

5. If multiple sources support a claim, cite all relevant sources.

6. Never cite a source that does not support the claim.

7. If evidence is insufficient, explicitly say so.

8. Do not claim causation from correlation.

9. Distinguish clearly between:

   - Observed Fact
   - Observed Pattern
   - Possible Interpretation
   - Recommendation

============================================================
ENGAGEMENT RULES
============================================================

Do NOT automatically equate Views with Engagement.

Views represent reach or exposure.

Likes, Comments, and Shares are engagement signals.

When the user asks:

"Which content performs best based on engagement?"

Analyze the engagement signals separately.

Use:

- Highest Likes
- Highest Comments
- Highest Shares

Do NOT invent a combined engagement score.

If one content item leads multiple engagement metrics,
you may describe it as the strongest engagement leader
for the retrieved sample.

If different content items lead different metrics,
clearly explain the difference.

============================================================
REACH VS ENGAGEMENT
============================================================

Always distinguish:

Reach Leader:
Content with the highest Views.

Engagement Leader:
Content that leads the available interaction metrics.

Example:

"content_31417 has the highest Views, while content_29243 leads
the available Likes, Comments, and Shares metrics."

This distinction is important.

============================================================
PLATFORM RULES
============================================================

Only compare platforms when comparable evidence exists.

Do not claim one platform is better overall unless the retrieved
evidence supports a fair comparison.

============================================================
CATEGORY RULES
============================================================

Do not claim that an entire category performs better based on
one or two content items.

Use conservative language:

"The retrieved sample suggests..."

"The available evidence indicates..."

"The observed pattern may suggest..."

============================================================
RECOMMENDATION RULES
============================================================

Recommendations must be logically connected to the evidence.

Prefer recommendations such as:

- validate the observed pattern with more data
- analyze high-performing content
- compare engagement metrics
- test similar content formats
- monitor future performance
- build a data-backed content calendar

Do not recommend paid campaigns, influencer partnerships,
or budget increases unless the evidence supports them.

============================================================
DECISION SIGNAL
============================================================

Choose exactly one:

HIGH SIGNAL

MODERATE SIGNAL

LOW SIGNAL

Use HIGH SIGNAL only when the evidence is strong and consistent.

Use MODERATE SIGNAL when the evidence suggests a pattern
but more validation is needed.

Use LOW SIGNAL when evidence is limited, mixed, or insufficient.

============================================================
RESPONSE STRUCTURE
============================================================

Return ONLY the following structured analysis.

### 🎯 Key Insight

Give the most important answer to the user's question.

If the question is about engagement, clearly identify the
engagement leader based on the verified Likes, Comments,
and Shares metrics.

If Views tell a different story, mention the Reach Leader separately.

Every factual claim must include a citation.

---

### 📊 Supporting Evidence

List the strongest verified evidence.

Use concise bullet points.

Every factual bullet must include a citation.

Do not change any numerical value from the verified analytics.

---

### 🔥 Engagement Drivers

Use:

**Observed Pattern**

Describe what the data actually shows.

**Possible Interpretation**

Explain what the pattern MAY indicate.

Do not present interpretations as proven facts.

---

### 💡 Content Opportunities

Suggest 2 to 4 evidence-based opportunities.

Clearly distinguish opportunities from proven facts.

---

### 🚀 Recommended Actions

Provide 3 to 5 practical actions.

Prioritize validation, analysis, testing, and monitoring.

---

### 📌 Decision Signal

Choose exactly one:

HIGH SIGNAL

MODERATE SIGNAL

LOW SIGNAL

Then explain the signal in one concise sentence.

---

### 🎯 Recommended Next Step

Provide exactly ONE specific next action.

---

### ⚠️ Data Limitations

Mention the most relevant limitations.

Consider:

- sample size
- missing metrics
- missing audience data
- missing platform comparisons
- missing causal evidence

============================================================
FINAL QUALITY CHECK
============================================================

Before returning the answer verify:

1. Numerical rankings match the VERIFIED ANALYTICS SUMMARY.

2. The highest Views content is correctly identified.

3. The highest Likes content is correctly identified.

4. The highest Comments content is correctly identified.

5. The highest Shares content is correctly identified.

6. Views are not incorrectly treated as engagement.

7. No unsupported causal claims are made.

8. Every factual claim has a source citation.

9. Exactly ONE Recommended Next Step is provided.

============================================================
VERIFIED ANALYTICS
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

    # --------------------------------------------------------
    # VALIDATE QUERY
    # --------------------------------------------------------

    if not query or not query.strip():

        return {
            "answer": (
                "Please enter a marketing question."
            ),
            "sources": [],
            "analytics": "",
        }


    # --------------------------------------------------------
    # RETRIEVE DOCUMENTS
    # --------------------------------------------------------

    docs = retrieve_context(
        vectorstore,
        query,
        k=k,
    )


    # --------------------------------------------------------
    # EMPTY RETRIEVAL
    # --------------------------------------------------------

    if not docs:

        return {
            "answer": (
                "I could not find enough relevant evidence "
                "in the marketing knowledge base to answer "
                "this question reliably."
            ),
            "sources": [],
            "analytics": "",
        }


    # --------------------------------------------------------
    # FORMAT RAW CONTEXT
    # --------------------------------------------------------

    context = format_context(
        docs
    )


    # --------------------------------------------------------
    # CALCULATE VERIFIED ANALYTICS
    # --------------------------------------------------------

    analytics = build_verified_analytics(
        docs
    )


    # --------------------------------------------------------
    # BUILD RAG + ANALYTICS CHAIN
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
    # GENERATE ANSWER
    # --------------------------------------------------------

    answer = chain.invoke(
        query
    )


    # --------------------------------------------------------
    # RETURN RESULT
    # --------------------------------------------------------

    return {
        "answer": answer,
        "sources": docs,
        "analytics": analytics,
    }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    vectorstore = load_vectorstore()

    llm = get_llm()

    query = (
        "Which content performs best based on engagement?"
    )

    result = generate_answer(
        query,
        vectorstore,
        llm,
        k=5,
    )


    print(
        "=" * 70
    )

    print(
        "MARKETPULSE AI"
    )

    print(
        "=" * 70
    )


    print(
        f"\nQuery:\n{query}"
    )


    print(
        "\n"
        "VERIFIED ANALYTICS"
    )

    print(
        "-" * 70
    )

    print(
        result.get(
            "analytics",
            ""
        )
    )


    print(
        "\n"
        "AI ANALYSIS"
    )

    print(
        "-" * 70
    )

    print(
        result.get(
            "answer",
            ""
        )
    )


    print(
        "\n"
        "Sources used: "
        f"{len(result.get('sources', []))}"
    )