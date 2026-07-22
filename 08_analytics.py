"""
08_analytics.py
----------------
Pipeline stage 8: MARKETING ANALYTICS

Responsible for deterministic calculations before the LLM.

This module:
- Extracts engagement metrics from retrieved documents.
- Calculates Engagement Score.
- Ranks content by Engagement Score.
- Identifies top-performing content.
- Identifies metric leaders.
- Calculates averages.
- Groups results by platform and category.
- Produces structured computed facts for the LLM.

IMPORTANT:
The LLM should NOT calculate rankings or engagement scores.
Python is the source of truth for all numerical analysis.
"""

import importlib.util
import os
import re
from collections import defaultdict


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

    spec = importlib.util.spec_from_file_location(
        module_name,
        os.path.join(
            BASE_DIR,
            module_filename,
        ),
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


# ============================================================
# METRIC HELPERS
# ============================================================

METRIC_NAMES = (
    "likes",
    "comments",
    "shares",
    "views",
)


def _safe_int(value):
    """
    Convert a value into an integer safely.

    Handles:
    - integers
    - floats
    - strings
    - comma-separated numbers
    - missing values
    """

    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ):
        return None

    if isinstance(
        value,
        (int, float),
    ):
        return int(value)

    value = str(value).strip()

    if not value:
        return None

    value = value.replace(
        ",",
        "",
    )

    match = re.search(
        r"-?\d+(?:\.\d+)?",
        value,
    )

    if not match:
        return None

    try:
        return int(
            float(
                match.group(0)
            )
        )

    except (
        ValueError,
        TypeError,
    ):
        return None


def _extract_metric(
    text,
    metric_name,
):
    """
    Extract a metric from document text.

    Expected examples:

    Likes: 1551
    1551 Likes
    Likes = 1551
    Likes - 1551
    """

    if not text:
        return None

    patterns = [

        rf"{metric_name}\s*[:=|-]\s*([\d,]+)",

        rf"([\d,]+)\s+{metric_name}",

        rf"{metric_name}\s+([\d,]+)",

    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:

            return _safe_int(
                match.group(1)
            )

    return None


# ============================================================
# EXTRACT DOCUMENT DATA
# ============================================================

def extract_document_metrics(
    doc,
    source_index,
):
    """
    Extract structured marketing metrics
    from a LangChain Document.
    """

    metadata = (
        doc.metadata
        or {}
    )

    text = (
        doc.page_content
        or ""
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
        "Unknown",
    )

    likes = _extract_metric(
        text,
        "Likes",
    )

    comments = _extract_metric(
        text,
        "Comments",
    )

    shares = _extract_metric(
        text,
        "Shares",
    )

    views = _extract_metric(
        text,
        "Views",
    )

    followers = _extract_metric(
        text,
        "Followers",
    )

    sponsored = None

    if re.search(
        r"\bSponsored\b",
        text,
        flags=re.IGNORECASE,
    ):

        sponsored = True

    elif re.search(
        r"\bNot Sponsored\b",
        text,
        flags=re.IGNORECASE,
    ):

        sponsored = False

    # ========================================================
    # ENGAGEMENT SCORE
    # ========================================================

    engagement_values = [
        value
        for value in [
            likes,
            comments,
            shares,
        ]
        if value is not None
    ]

    if engagement_values:

        engagement_score = sum(
            engagement_values
        )

    else:

        engagement_score = None

    return {

        "source_index": source_index,

        "source_label": (
            f"[Source {source_index}]"
        ),

        "content_id": content_id,

        "platform": platform,

        "category": category,

        "likes": likes,

        "comments": comments,

        "shares": shares,

        "views": views,

        "followers": followers,

        "sponsored": sponsored,

        "engagement_score": (
            engagement_score
        ),

        "content": text,

    }


# ============================================================
# ANALYZE RETRIEVED DOCUMENTS
# ============================================================

def analyze_documents(
    docs,
):
    """
    Analyze retrieved documents using deterministic Python logic.

    Returns structured analytics.
    """

    if not docs:

        return {

            "items": [],

            "ranked_items": [],

            "top_content": None,

            "metric_leaders": {},

            "averages": {},

            "platform_summary": {},

            "category_summary": {},

            "limitations": [

                "No documents were retrieved."

            ],

        }

    items = []

    for index, doc in enumerate(
        docs,
        start=1,
    ):

        item = extract_document_metrics(
            doc,
            index,
        )

        items.append(
            item
        )

    # ========================================================
    # RANK BY ENGAGEMENT SCORE
    # ========================================================

    ranked_items = sorted(

        [

            item

            for item in items

            if item[
                "engagement_score"
            ] is not None

        ],

        key=lambda item:
            item[
                "engagement_score"
            ],

        reverse=True,

    )

    # ========================================================
    # ADD RANK
    # ========================================================

    for rank, item in enumerate(
        ranked_items,
        start=1,
    ):

        item[
            "engagement_rank"
        ] = rank

    # ========================================================
    # TOP CONTENT
    # ========================================================

    top_content = None

    if ranked_items:

        top_content = (
            ranked_items[0]
        )

    # ========================================================
    # METRIC LEADERS
    # ========================================================

    metric_leaders = {}

    for metric in [
        "likes",
        "comments",
        "shares",
        "views",
        "engagement_score",
    ]:

        valid_items = [

            item

            for item in items

            if item.get(
                metric
            ) is not None

        ]

        if valid_items:

            leader = max(

                valid_items,

                key=lambda item:
                    item[
                        metric
                    ],

            )

            metric_leaders[
                metric
            ] = {

                "content_id":
                    leader[
                        "content_id"
                    ],

                "value":
                    leader[
                        metric
                    ],

                "source_index":
                    leader[
                        "source_index"
                    ],

                "source_label":
                    leader[
                        "source_label"
                    ],

            }

    # ========================================================
    # AVERAGES
    # ========================================================

    averages = {}

    for metric in [
        "likes",
        "comments",
        "shares",
        "views",
        "engagement_score",
    ]:

        values = [

            item[
                metric
            ]

            for item in items

            if item.get(
                metric
            ) is not None

        ]

        if values:

            averages[
                metric
            ] = round(

                sum(values)
                / len(values),

                2,

            )

    # ========================================================
    # PLATFORM SUMMARY
    # ========================================================

    platform_groups = defaultdict(
        list
    )

    for item in items:

        platform_groups[
            item[
                "platform"
            ]
        ].append(
            item
        )

    platform_summary = {}

    for platform, group in (
        platform_groups.items()
    ):

        scores = [

            item[
                "engagement_score"
            ]

            for item in group

            if item.get(
                "engagement_score"
            ) is not None

        ]

        views = [

            item[
                "views"
            ]

            for item in group

            if item.get(
                "views"
            ) is not None

        ]

        platform_summary[
            platform
        ] = {

            "content_count":
                len(group),

            "average_engagement_score":

                round(

                    sum(scores)
                    / len(scores),

                    2,

                )

                if scores

                else None,

            "average_views":

                round(

                    sum(views)
                    / len(views),

                    2,

                )

                if views

                else None,

        }

    # ========================================================
    # CATEGORY SUMMARY
    # ========================================================

    category_groups = defaultdict(
        list
    )

    for item in items:

        category_groups[
            item[
                "category"
            ]
        ].append(
            item
        )

    category_summary = {}

    for category, group in (
        category_groups.items()
    ):

        scores = [

            item[
                "engagement_score"
            ]

            for item in group

            if item.get(
                "engagement_score"
            ) is not None

        ]

        category_summary[
            category
        ] = {

            "content_count":
                len(group),

            "average_engagement_score":

                round(

                    sum(scores)
                    / len(scores),

                    2,

                )

                if scores

                else None,

        }

    # ========================================================
    # LIMITATIONS
    # ========================================================

    limitations = []

    if len(items) < 10:

        limitations.append(

            "The retrieved sample contains "
            f"only {len(items)} content items."

        )

    if not all(
        item.get("views")
        is not None
        for item in items
    ):

        limitations.append(

            "Views are missing for some "
            "retrieved content items."

        )

    if not all(
        item.get("likes")
        is not None
        for item in items
    ):

        limitations.append(

            "Likes are missing for some "
            "retrieved content items."

        )

    if not all(
        item.get("comments")
        is not None
        for item in items
    ):

        limitations.append(

            "Comments are missing for some "
            "retrieved content items."

        )

    if not all(
        item.get("shares")
        is not None
        for item in items
    ):

        limitations.append(

            "Shares are missing for some "
            "retrieved content items."

        )

    if len(
        set(
            item[
                "platform"
            ]

            for item in items
        )
    ) < 2:

        limitations.append(

            "The retrieved sample contains "
            "only one platform, so platform "
            "comparison is limited."

        )

    return {

        "items":
            items,

        "ranked_items":
            ranked_items,

        "top_content":
            top_content,

        "metric_leaders":
            metric_leaders,

        "averages":
            averages,

        "platform_summary":
            platform_summary,

        "category_summary":
            category_summary,

        "limitations":
            limitations,

    }


# ============================================================
# FORMAT COMPUTED FACTS FOR LLM
# ============================================================

def format_computed_facts(
    analytics,
):
    """
    Convert deterministic Python analytics
    into a clean context block for the LLM.

    The LLM should interpret these facts,
    not recalculate them.
    """

    if not analytics:

        return (
            "No computed analytics available."
        )

    lines = []

    lines.append(
        "COMPUTED MARKETING ANALYTICS"
    )

    lines.append(
        "================================"
    )

    lines.append(
        "IMPORTANT: "
        "The following numerical results "
        "were calculated by Python."
    )

    lines.append(
        "Do not recalculate or change "
        "the rankings."
    )

    lines.append("")

    # ========================================================
    # TOP CONTENT
    # ========================================================

    top = analytics.get(
        "top_content"
    )

    if top:

        lines.append(
            "TOP PERFORMING CONTENT"
        )

        lines.append(
            f"Content ID: "
            f"{top['content_id']}"
        )

        lines.append(
            f"Platform: "
            f"{top['platform']}"
        )

        lines.append(
            f"Category: "
            f"{top['category']}"
        )

        lines.append(
            f"Engagement Score: "
            f"{top['engagement_score']}"
        )

        lines.append(
            f"Likes: "
            f"{top['likes']}"
        )

        lines.append(
            f"Comments: "
            f"{top['comments']}"
        )

        lines.append(
            f"Shares: "
            f"{top['shares']}"
        )

        lines.append(
            f"Views: "
            f"{top['views']}"
        )

        lines.append(
            f"Source: "
            f"{top['source_label']}"
        )

        lines.append("")

    # ========================================================
    # RANKING
    # ========================================================

    ranked_items = analytics.get(
        "ranked_items",
        [],
    )

    if ranked_items:

        lines.append(
            "ENGAGEMENT RANKING"
        )

        for item in ranked_items:

            lines.append(

                f"{item['engagement_rank']}. "
                f"{item['content_id']} | "
                f"Score: "
                f"{item['engagement_score']} | "
                f"Platform: "
                f"{item['platform']} | "
                f"Category: "
                f"{item['category']} | "
                f"Source: "
                f"{item['source_label']}"

            )

        lines.append("")

    # ========================================================
    # METRIC LEADERS
    # ========================================================

    leaders = analytics.get(
        "metric_leaders",
        {},
    )

    if leaders:

        lines.append(
            "METRIC LEADERS"
        )

        for metric, data in (
            leaders.items()
        ):

            lines.append(

                f"{metric.title()}: "
                f"{data['content_id']} "
                f"with {data['value']} "
                f"{data['source_label']}"

            )

        lines.append("")

    # ========================================================
    # AVERAGES
    # ========================================================

    averages = analytics.get(
        "averages",
        {},
    )

    if averages:

        lines.append(
            "AVERAGE METRICS"
        )

        for metric, value in (
            averages.items()
        ):

            lines.append(

                f"Average {metric}: "
                f"{value}"

            )

        lines.append("")

    # ========================================================
    # PLATFORM SUMMARY
    # ========================================================

    platform_summary = analytics.get(
        "platform_summary",
        {},
    )

    if platform_summary:

        lines.append(
            "PLATFORM SUMMARY"
        )

        for platform, data in (
            platform_summary.items()
        ):

            lines.append(

                f"{platform}: "
                f"{data['content_count']} "
                f"content items | "
                f"Average Engagement Score: "
                f"{data['average_engagement_score']} | "
                f"Average Views: "
                f"{data['average_views']}"

            )

        lines.append("")

    # ========================================================
    # CATEGORY SUMMARY
    # ========================================================

    category_summary = analytics.get(
        "category_summary",
        {},
    )

    if category_summary:

        lines.append(
            "CATEGORY SUMMARY"
        )

        for category, data in (
            category_summary.items()
        ):

            lines.append(

                f"{category}: "
                f"{data['content_count']} "
                f"content items | "
                f"Average Engagement Score: "
                f"{data['average_engagement_score']}"

            )

        lines.append("")

    # ========================================================
    # LIMITATIONS
    # ========================================================

    limitations = analytics.get(
        "limitations",
        [],
    )

    if limitations:

        lines.append(
            "ANALYTICAL LIMITATIONS"
        )

        for limitation in limitations:

            lines.append(
                f"- {limitation}"
            )

    return "\n".join(
        lines
    )


# ============================================================
# COMPLETE ANALYTICS PIPELINE
# ============================================================

def analyze_query(
    query: str,
    vectorstore,
    k: int = 5,
):
    """
    Retrieve documents and run deterministic
    Python analytics on them.
    """

    if not query or not query.strip():

        return {

            "documents": [],

            "analytics": analyze_documents(
                []
            ),

        }

    docs = retrieve_context(
        vectorstore,
        query,
        k=k,
    )

    analytics = analyze_documents(
        docs
    )

    return {

        "documents":
            docs,

        "analytics":
            analytics,

    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    vectorstore = load_vectorstore()

    query = (
        "Which content performs best "
        "based on engagement?"
    )

    result = analyze_query(
        query,
        vectorstore,
        k=5,
    )

    analytics = result[
        "analytics"
    ]

    print(
        "\n"
        + "=" * 60
    )

    print(
        "MARKETPULSE AI - PYTHON ANALYTICS"
    )

    print(
        "=" * 60
    )

    print(
        format_computed_facts(
            analytics
        )
    )

    print(
        "\n"
        + "=" * 60
    )