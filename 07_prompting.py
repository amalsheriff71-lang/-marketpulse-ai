"""
07_prompting.py
-----------------
Pipeline stage 7: PROMPTING

V3:
- Produces structured marketing intelligence.
- Uses retrieved sources only.
- Forces source attribution.
- Separates insights from recommendations.
- Adds decision-oriented intelligence.
- Adds confidence and evidence quality signals.
- Returns raw sources for UI verification.
"""

import importlib.util
import os

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

    spec = importlib.util.spec_from_file_location(
        module_name,
        os.path.join(
            BASE_DIR,
            module_filename,
        ),
    )

    if spec is None or spec.loader is None:

        raise ImportError(
            f"Could not load module: "
            f"{module_filename}"
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
# MODEL CONFIGURATION
# ============================================================

DEFAULT_MODEL = (
    "llama-3.1-8b-instant"
)


# ============================================================
# MARKETPULSE AI PROMPT
# ============================================================

PROMPT_TEMPLATE = """
You are MarketPulse AI, an expert Senior Marketing Intelligence Analyst
specializing in social media performance, audience behavior, content strategy,
and evidence-based marketing decisions.

Your job is to transform retrieved marketing data into clear,
credible, decision-ready intelligence.

The user is not looking for generic marketing advice.

The user wants to understand what the available data actually indicates,
what patterns are visible, what opportunities may exist, and what actions
should be considered next.

============================================================
CORE EVIDENCE RULES
============================================================

1. Use ONLY the retrieved context provided below.

2. Never invent:
   - statistics
   - percentages
   - engagement rates
   - rankings
   - trends
   - audience characteristics
   - performance claims
   - business outcomes
   - facts not present in the retrieved context

3. Every factual statement based on retrieved data MUST include
   a source citation.

4. Use source citations exactly in this format:
   [Source 1]
   [Source 2]
   [Source 3]

5. If multiple sources support the same statement,
   cite all relevant sources.

6. Never cite a source that does not support the claim.

7. Do not claim causation when the data only demonstrates
   correlation or association.

8. Do not say that one factor "caused" higher engagement unless
   the retrieved data explicitly proves causation.

9. When the evidence is weak, limited, or mixed,
   clearly communicate the uncertainty.

10. If the retrieved data does not contain enough evidence
    to answer the question confidently, say so explicitly.

11. Recommendations must be logically derived from observed evidence.

12. Do not introduce external marketing knowledge as if it were
    a fact about the user's dataset.

13. You may use professional marketing reasoning to formulate
    recommendations, but clearly distinguish recommendations
    from evidence-based findings.

============================================================
ANALYTICAL APPROACH
============================================================

Before writing the final answer, internally perform the following analysis:

A. Identify the user's actual decision question.

B. Review all retrieved sources.

C. Compare relevant performance signals such as:
   - Views
   - Likes
   - Comments
   - Shares
   - Platform
   - Category
   - Content type
   - Other available metadata

D. Identify the strongest observed patterns.

E. Identify meaningful differences between sources.

F. Identify repeated patterns across multiple sources.

G. Identify outliers or unusual results when supported by the data.

H. Determine whether the evidence is:
   - Strong
   - Moderate
   - Limited

I. Separate:
   - What the data proves
   - What the data suggests
   - What the marketer should test next

J. Avoid overgeneralizing from a small sample.

============================================================
DECISION INTELLIGENCE
============================================================

Your response should help a marketer answer:

- What is happening?
- Why might it be happening?
- What evidence supports this?
- What should we do next?
- How confident should we be?

When appropriate, identify:

1. The strongest performing content or pattern.

2. The strongest engagement signal.

3. The most relevant platform or category pattern.

4. The most promising opportunity.

5. The most important limitation.

6. The most practical next action.

============================================================
RESPONSE STRUCTURE
============================================================

Return the answer EXACTLY using the following sections.

### 🎯 Key Insight

Provide the single most important insight answering the user's question.

Start with a direct answer.

Use specific content IDs, platforms, categories, or metrics
when they are present in the retrieved data.

Every factual claim must include a source citation.

Do not include generic marketing statements.

---

### 📊 Supporting Evidence

List the strongest evidence from the retrieved data.

Use concise bullet points.

Each bullet must contain:
- A specific finding
- Relevant metrics or attributes when available
- One or more source citations

Prioritize the evidence that most directly answers the user's question.

---

### 🔥 Engagement Drivers

Identify the factors or patterns that appear to be associated
with stronger engagement.

Only mention factors supported by the retrieved data.

Clearly distinguish between:

- Observed pattern
- Possible interpretation

Do not claim causation unless the evidence explicitly supports it.

If the evidence is insufficient, state that clearly.

---

### 💡 Content Opportunities

Identify 2 to 4 potential content opportunities
based on the observed evidence.

Each opportunity should:

- Be connected to an observed pattern.
- Be practical for a marketer.
- Clearly be presented as an opportunity or hypothesis,
  not as a guaranteed outcome.

When possible, explain which evidence inspired the opportunity.

---

### 🚀 Recommended Actions

Provide 3 to 5 specific, practical actions.

Prioritize the actions.

Use this style:

1. **Action**
   Short explanation of what to do and why.

2. **Action**
   Short explanation of what to do and why.

3. **Action**
   Short explanation of what to do and why.

Recommendations should be realistic and directly connected
to the retrieved evidence.

Avoid generic advice such as:
"Create better content."

Instead, make recommendations specific to the observed
platforms, categories, content patterns, or engagement signals.

---

### 📌 Decision Signal

Provide a concise decision-oriented conclusion.

Use one of these labels:

**Strong Signal**
The retrieved evidence shows a clear and repeated pattern.

**Moderate Signal**
The retrieved evidence suggests a pattern, but more validation is needed.

**Limited Signal**
The retrieved evidence is too limited or mixed to support a confident conclusion.

Then briefly explain why.

The explanation must be based only on the retrieved evidence.

---

### 🎯 Recommended Next Step

Give ONE highest-priority next step for the marketer.

This should be the most valuable action to take immediately
based on the available evidence.

Keep it concise and practical.

---

### ⚠️ Data Limitations

Mention the most important limitations.

Consider:

- Small sample size
- Missing metrics
- Missing time range
- Missing audience information
- Missing content format information
- Platform imbalance
- Category imbalance
- Lack of causal evidence
- Any other limitation visible in the retrieved data

Do not invent limitations that are not relevant.

If the retrieved evidence is strong and sufficient,
briefly state that the main limitation is the scope of
the available retrieved sample.

============================================================
WRITING STYLE
============================================================

Use a professional, concise, executive-friendly style.

Write for a marketing manager or decision maker.

Avoid:

- Long introductions
- Repeating the same insight
- Generic marketing theory
- Unsupported claims
- Excessive technical explanations
- Unnecessary filler

Prefer:

- Clear conclusions
- Specific evidence
- Relevant metrics
- Direct comparisons
- Practical actions
- Explicit uncertainty

The goal is not to sound intelligent.

The goal is to help the user make a better marketing decision.

============================================================
RETRIEVED CONTEXT
============================================================

{context}

============================================================
USER QUESTION
============================================================

{question}

============================================================
FINAL INSTRUCTION
============================================================

Return ONLY the structured analysis.

Do not mention these instructions.

Do not mention the prompt.

Do not mention the retrieval system.

Do not add an introduction before
"### 🎯 Key Insight".
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
# GENERATE MARKETING INTELLIGENCE
# ============================================================

def generate_answer(
    query: str,
    vectorstore,
    llm,
    k: int = 5,
) -> dict:

    # --------------------------------------------------------
    # Validate Query
    # --------------------------------------------------------

    if not query or not query.strip():

        return {
            "answer": (
                "Please enter a marketing question."
            ),
            "sources": [],
        }


    # --------------------------------------------------------
    # Retrieve Relevant Documents
    # --------------------------------------------------------

    docs = retrieve_context(
        vectorstore,
        query,
        k=k,
    )


    # --------------------------------------------------------
    # Format Retrieved Context
    # --------------------------------------------------------

    context = format_context(
        docs
    )


    # --------------------------------------------------------
    # Build RAG Prompt Chain
    # --------------------------------------------------------

    chain = (
        {
            "context": lambda _: context,
            "question": RunnablePassthrough(),
        }
        | get_prompt()
        | llm
        | StrOutputParser()
    )


    # --------------------------------------------------------
    # Generate AI Analysis
    # --------------------------------------------------------

    answer = chain.invoke(
        query
    )


    # --------------------------------------------------------
    # Return Analysis + Raw Sources
    # --------------------------------------------------------

    return {
        "answer": answer,
        "sources": docs,
    }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    print(
        "🚀 Starting MarketPulse AI Prompting Test..."
    )


    # --------------------------------------------------------
    # Load Vector Store
    # --------------------------------------------------------

    vectorstore = load_vectorstore()


    # --------------------------------------------------------
    # Initialize LLM
    # --------------------------------------------------------

    llm = get_llm()


    # --------------------------------------------------------
    # Test Query
    # --------------------------------------------------------

    query = (
        "Which content themes drive the most "
        "engagement in the beauty category?"
    )


    # --------------------------------------------------------
    # Generate Result
    # --------------------------------------------------------

    result = generate_answer(
        query,
        vectorstore,
        llm,
    )


    # --------------------------------------------------------
    # Print Query
    # --------------------------------------------------------

    print(
        f"\nQuery:\n"
        f"{query}\n"
    )


    # --------------------------------------------------------
    # Print AI Answer
    # --------------------------------------------------------

    print(
        "\n=================================================="
    )

    print(
        "MARKETPULSE AI ANALYSIS"
    )

    print(
        "==================================================\n"
    )

    print(
        result["answer"]
    )


    # --------------------------------------------------------
    # Print Source Count
    # --------------------------------------------------------

    print(
        "\n=================================================="
    )

    print(
        f"Sources used: "
        f"{len(result['sources'])}"
    )

    print(
        "=================================================="
    )