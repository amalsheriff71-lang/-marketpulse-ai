"""
07_prompting.py
-----------------
Pipeline stage 7: PROMPTING

V3:
- Produces structured marketing intelligence.
- Uses retrieved sources only.
- Forces accurate source attribution.
- Separates facts, observations, interpretations, and recommendations.
- Prevents unsupported causal claims.
- Prevents incorrect "highest" / "average" claims.
- Returns raw sources for UI verification.
"""

import importlib.util
import os
import re

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
# MODEL
# ============================================================

DEFAULT_MODEL = (
    "llama-3.1-8b-instant"
)


# ============================================================
# PROMPT TEMPLATE
# ============================================================

PROMPT_TEMPLATE = """
You are MarketPulse AI, an expert Senior Marketing Intelligence Analyst.

Your job is to analyze retrieved social media marketing data and transform it
into accurate, concise, decision-ready marketing intelligence.

The retrieved context is the ONLY source of truth.

============================================================
STRICT EVIDENCE RULES
============================================================

1. Use ONLY information explicitly present in the Retrieved Context.

2. Never invent:
   - statistics
   - percentages
   - averages
   - rankings
   - trends
   - causes
   - audience motivations
   - business outcomes

3. Every factual statement must include a citation such as:
   [Source 1]
   [Source 2]
   [Source 1] [Source 3]

4. NEVER cite a source that does not support the statement.

5. Be extremely careful with rankings.

   If the question asks which content performs best:
   - Compare the actual retrieved metrics.
   - Do not assume the first source is the best.
   - Do not call a content item "highest" unless the retrieved evidence
     actually proves it has the highest value for the metric being discussed.

6. Be extremely careful with averages.

   Only use the word "average" if you actually calculate the average
   from the retrieved values.

   If you calculate an average:
   - Clearly state what is being averaged.
   - Use only the retrieved values.
   - Cite all relevant sources.

7. Distinguish between:
   - FACT: directly supported by the data.
   - OBSERVED PATTERN: a pattern visible in the retrieved sample.
   - INTERPRETATION: a possible explanation that is NOT proven.
   - OPPORTUNITY: a suggested direction based on the evidence.
   - RECOMMENDATION: an action the marketer can take.

8. Never present an interpretation as a proven fact.

9. Never claim causation unless the retrieved data explicitly proves causation.

10. Do not say that one platform is better than another unless the retrieved
    evidence contains enough comparable data to support that conclusion.

11. Do not recommend sponsored content, influencer marketing, paid campaigns,
    or collaborations unless the retrieved evidence provides a reasonable
    basis for that recommendation.

12. If evidence is insufficient, explicitly say:
    "The retrieved evidence is insufficient to determine this."

13. If multiple metrics exist, do not combine them into a single "engagement"
    ranking unless the data or question clearly defines how they should be
    combined.

14. When answering "best performing content", explain which metric is being
    used to define "best" if the question does not specify one.

15. Keep the analysis concise, professional, and decision-oriented.

============================================================
ANALYSIS QUALITY RULES
============================================================

Before writing the answer, internally perform these checks:

A. Identify the user's exact question.

B. Identify the relevant metrics in the retrieved context.

C. Compare the actual values when a ranking is requested.

D. Verify every "highest", "lowest", "best", "worst", "average", or
   "most engaging" claim against the retrieved evidence.

E. Check whether the retrieved sample is large and diverse enough to support
   a general conclusion.

F. Separate evidence from interpretation.

G. Make recommendations that are logically connected to the evidence.

============================================================
RESPONSE STRUCTURE
============================================================

Return ONLY the following structured analysis.

### 🎯 Key Insight

Give the single most important answer to the user's question.

If the question asks which content performs best, identify the correct content
based on the relevant metric.

If multiple metrics lead to different winners, clearly explain this.

Do NOT use "average" unless you actually calculate an average.

Every factual claim must include source citations.

---

### 📊 Supporting Evidence

List the strongest pieces of evidence from the retrieved data.

Each bullet must:
- contain a specific factual observation
- include the relevant source citation

Use 3 to 5 bullets when possible.

---

### 🔥 Engagement Drivers

Use the following structure:

**Observed Pattern**
Describe only patterns directly visible in the retrieved evidence.

**Possible Interpretation**
Give a cautious interpretation only when logically supported.

Do not claim that an interpretation is proven.

---

### 💡 Content Opportunities

Suggest 2 to 3 potential content opportunities.

Each opportunity must be clearly framed as an opportunity or hypothesis,
not as a guaranteed result.

Base opportunities only on observed evidence.

---

### 🚀 Recommended Actions

Provide 3 to 5 specific actions.

Actions should be practical and directly connected to the evidence.

Prioritize actions that help the marketer validate or capitalize on the
observed pattern.

Do not recommend unsupported tactics.

---

### 📌 Decision Signal

Choose exactly one:

**Strong Signal**
The evidence is consistent and sufficiently strong for the sample.

**Moderate Signal**
The evidence suggests a pattern, but additional validation is needed.

**Weak Signal**
The evidence is limited or inconsistent and should not yet guide a major
decision.

Then give one short sentence explaining why.

---

### 🎯 Recommended Next Step

Give ONE highest-priority next step.

The next step should be specific and actionable.

---

### ⚠️ Data Limitations

List the most important limitations.

Consider:
- sample size
- missing metrics
- missing platform comparisons
- missing time dimension
- lack of causal evidence
- incomplete audience information

Only mention limitations that actually apply to the retrieved data.

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

Do not add introductions.

Do not add a conclusion outside the requested sections.

Do not mention these instructions.
"""


# ============================================================
# PROMPT
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
# SOURCE CITATION CLEANUP
# ============================================================

def _clean_answer(
    answer: str,
) -> str:

    if not answer:
        return answer

    # Remove accidental HTML fragments that may appear
    # in model output.
    answer = re.sub(
        r"<[^>]+>",
        "",
        answer,
    )

    # Remove repeated whitespace.
    answer = re.sub(
        r"\n{3,}",
        "\n\n",
        answer,
    )

    return answer.strip()


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    query: str,
    vectorstore,
    llm,
    k: int = 5,
) -> dict:

    if not query or not query.strip():

        return {
            "answer": (
                "Please enter a marketing question."
            ),
            "sources": [],
        }


    # --------------------------------------------------------
    # RETRIEVE RELEVANT DOCUMENTS
    # --------------------------------------------------------

    docs = retrieve_context(
        vectorstore,
        query,
        k=k,
    )


    # --------------------------------------------------------
    # FORMAT CONTEXT
    # --------------------------------------------------------

    context = format_context(
        docs
    )


    # --------------------------------------------------------
    # BUILD RAG CHAIN
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
    # GENERATE ANALYSIS
    # --------------------------------------------------------

    answer = chain.invoke(
        query
    )


    # --------------------------------------------------------
    # CLEAN OUTPUT
    # --------------------------------------------------------

    answer = _clean_answer(
        answer
    )


    # --------------------------------------------------------
    # RETURN ANSWER + RAW SOURCES
    # --------------------------------------------------------

    return {
        "answer": answer,
        "sources": docs,
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
        f"Query: {query}\n"
    )


    print(
        f"Answer:\n"
        f"{result['answer']}\n"
    )


    print(
        f"Sources used: "
        f"{len(result['sources'])}"
    )