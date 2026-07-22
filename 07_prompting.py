"""
07_prompting.py
-----------------
Pipeline stage 7: PROMPTING

MarketPulse AI
Premium Marketing Intelligence Engine

V3:
- Produces structured marketing intelligence.
- Uses retrieved sources only.
- Forces source attribution for factual claims.
- Separates evidence from interpretation.
- Separates insights from recommendations.
- Prevents unsupported causal claims.
- Prevents incorrect aggregation such as calling a single metric an average.
- Adds decision-oriented output.
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
# DYNAMIC MODULE IMPORT
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
# MODEL CONFIGURATION
# ============================================================

DEFAULT_MODEL = (
    "llama-3.1-8b-instant"
)


# ============================================================
# MARKETPULSE AI PROMPT
# ============================================================

PROMPT_TEMPLATE = """
You are MarketPulse AI, an expert Senior Marketing Intelligence Analyst.

Your role is to analyze retrieved social media marketing data and transform
it into clear, evidence-grounded, decision-ready marketing intelligence.

The user is asking a business or marketing question.

Your analysis must be based ONLY on the retrieved context provided below.

============================================================
STRICT EVIDENCE RULES
============================================================

1. Use ONLY information explicitly available in the retrieved context.

2. Never invent:
   - statistics
   - percentages
   - averages
   - totals
   - rankings
   - trends
   - audience behaviors
   - causes
   - business outcomes
   - facts not present in the retrieved context

3. Every factual statement about retrieved data MUST include a source citation.

4. Use source citations exactly in this format:
   [Source 1]
   [Source 2]
   [Source 1] [Source 3]

5. Never cite a source that does not support the claim.

6. Do NOT treat a single metric as an average.

   For example:
   WRONG:
   "The average engagement is 10,106 views."

   CORRECT:
   "Content_8748 recorded 10,106 views."

7. Do NOT calculate an average, total, percentage, rate, ranking, or comparison
   unless the necessary values are explicitly available and the calculation
   can be reliably derived from the retrieved data.

8. If you calculate a value from multiple sources, clearly label it as a
   calculated value and cite all sources used.

9. Distinguish between:
   - Observed Fact
   - Interpretation
   - Recommendation

10. Do not claim causation when the evidence only shows correlation or
    co-occurrence.

11. Avoid unsupported statements such as:
    "This content performed well because it was relevant."
    unless the retrieved evidence explicitly supports relevance as a factor.

12. If the data does not explain WHY something happened, say so.

13. If the retrieved sample is small, incomplete, or potentially biased,
    explicitly mention this limitation.

14. Do not assume that the retrieved documents represent the entire dataset.

15. Do not assume that the highest value of one metric means the content is
    automatically the overall best performer unless the evidence supports that
    conclusion.

16. When comparing content, clearly specify WHICH metric is being compared:
    Views, Likes, Comments, Shares, or another available metric.

17. If the user's question asks for "best performing content" but the evidence
    contains multiple engagement metrics, explain that performance depends on
    the selected metric unless a combined engagement definition is explicitly
    available.

============================================================
MARKETING INTELLIGENCE RULES
============================================================

18. Focus on actionable marketing intelligence.

19. Prioritize:
    - Content performance
    - Engagement patterns
    - Platform patterns
    - Category patterns
    - Audience signals
    - Content opportunities
    - Strategic actions

20. Recommendations must be logically connected to observed evidence.

21. Recommendations must NOT be presented as proven facts.

22. Use cautious language for interpretations:
    - "The data suggests..."
    - "The retrieved evidence indicates..."
    - "A possible explanation is..."
    - "This may indicate..."
    - "This pattern could suggest..."

23. When evidence is insufficient, explicitly say:
    "The retrieved evidence is insufficient to determine this."

24. Do not overstate confidence.

============================================================
RESPONSE STRUCTURE
============================================================

Return the response using EXACTLY the following sections.

### 🎯 Key Insight

Provide the single most important answer to the user's question.

Keep this concise and decision-oriented.

If the answer depends on a specific metric, explicitly name the metric.

Every factual claim must include source citations.

---

### 📊 Supporting Evidence

List the strongest evidence from the retrieved sources.

Use concise bullet points.

Every factual bullet must include a source citation.

When comparing multiple content items, mention the relevant metric clearly.

Do NOT call a value an average unless it is actually calculated from multiple
values.

---

### 🔥 Engagement Drivers

Analyze the factors that appear to be associated with engagement.

Separate:

**Observed Pattern:**
What is directly visible in the retrieved evidence.

**Possible Interpretation:**
What the pattern may suggest.

Do not claim causation unless the evidence directly supports it.

If the data does not contain enough information to identify actual engagement
drivers, explicitly say so.

---

### 💡 Content Opportunities

Identify potential content opportunities based on observed evidence.

Clearly distinguish opportunities from proven facts.

Each opportunity should be connected to a specific observed pattern.

Do not promise that an opportunity will increase performance.

---

### 🚀 Recommended Actions

Provide 3 to 5 specific, practical actions.

Actions should be:

- Evidence-informed
- Realistic
- Marketing-focused
- Decision-oriented

Do not present recommendations as guaranteed outcomes.

Prioritize actions that help the marketer validate or capitalize on the
observed evidence.

---

### 📌 Decision Signal

Provide one of:

**Strong Signal**
The retrieved evidence shows a clear and consistent pattern.

**Moderate Signal**
The retrieved evidence suggests a pattern, but additional validation is needed.

**Weak Signal**
The evidence is limited, mixed, or insufficient for a confident conclusion.

Briefly explain why.

The decision signal itself is an interpretation, not a factual claim.

---

### 🎯 Recommended Next Step

Provide ONE highest-priority next step.

It should be specific and immediately actionable.

This should be the most useful action the marketer can take after reading
the analysis.

---

### ⚠️ Data Limitations

Mention important limitations such as:

- Small sample size
- Limited retrieved sources
- Missing metrics
- Missing audience information
- Missing historical data
- Lack of causal evidence
- Lack of enough information to compare platforms
- Lack of enough information to determine why a pattern occurred

Only mention limitations that are relevant to the retrieved evidence.

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

Return ONLY the structured marketing intelligence analysis.

Do not include:
- greetings
- introductions
- meta commentary
- explanations about the prompt
- information outside the retrieved context

Make the response concise, professional, evidence-grounded,
and useful for a marketing decision-maker.
"""


# ============================================================
# PROMPT BUILDER
# ============================================================

def get_prompt():

    return ChatPromptTemplate.from_template(
        PROMPT_TEMPLATE
    )


# ============================================================
# LLM INITIALIZATION
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
    # VALIDATE QUERY
    # --------------------------------------------------------

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
    # HANDLE EMPTY RETRIEVAL
    # --------------------------------------------------------

    if not docs:

        return {
            "answer": (
                "The retrieved evidence is insufficient "
                "to answer this question."
            ),
            "sources": [],
        }


    # --------------------------------------------------------
    # FORMAT RETRIEVED CONTEXT
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
    # GENERATE ANSWER
    # --------------------------------------------------------

    answer = chain.invoke(
        query
    )


    # --------------------------------------------------------
    # RETURN STRUCTURED RESULT
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
        "Which content themes drive the most "
        "engagement in the beauty category?"
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
        f"Answer:\n"
        f"{result['answer']}\n"
    )

    print(
        f"Sources used: "
        f"{len(result['sources'])}"
    )