"""
07_prompting.py
-----------------
Pipeline stage 7: PROMPTING

V2:
- Produces structured marketing intelligence.
- Uses retrieved sources only.
- Forces source attribution.
- Separates insights from recommendations.
- Returns raw sources for UI verification.
"""

import importlib.util
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


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

    mod = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        mod
    )

    return mod


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


DEFAULT_MODEL = (
    "llama-3.1-8b-instant"
)


PROMPT_TEMPLATE = """
You are MarketPulse AI, an expert Senior Marketing Intelligence Analyst.

Your job is to analyze social media marketing data and turn it into
clear, actionable business insights.

IMPORTANT RULES:

1. Use ONLY the retrieved context provided below.
2. Never invent statistics, percentages, trends, or facts.
3. Every factual claim must include one or more source citations.
4. Use citations exactly like [Source 1] or [Source 2].
5. If the data is insufficient, clearly say so.
6. Separate factual findings from recommendations.
7. Recommendations must be logically derived from the available evidence.
8. Do not claim causation when the data only shows correlation.
9. Keep the answer concise, professional, and decision-oriented.

Structure your response EXACTLY using these sections:

### 🎯 Key Insight

Give the single most important insight answering the user's question.

### 📊 Supporting Evidence

List the strongest evidence from the retrieved data.
Every bullet must include a source citation.

### 🔥 Engagement Drivers

Identify the factors that appear to drive engagement.
Only mention factors supported by the retrieved data.

### 💡 Content Opportunities

Suggest content opportunities based on observed patterns.
Clearly distinguish opportunities from proven facts.

### 🚀 Recommended Actions

Provide 3 to 5 specific actions the marketer could take next.

### ⚠️ Data Limitations

Mention important limitations if the retrieved data is insufficient,
incomplete, or does not directly answer part of the question.

Retrieved Context:

{context}

User Question:

{question}

Return ONLY the structured analysis.
"""


def get_prompt():

    return ChatPromptTemplate.from_template(
        PROMPT_TEMPLATE
    )


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

    docs = retrieve_context(
        vectorstore,
        query,
        k=k,
    )

    context = format_context(
        docs
    )

    chain = (
        {
            "context": lambda _: context,
            "question": RunnablePassthrough(),
        }
        | get_prompt()
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(
        query
    )

    return {
        "answer": answer,
        "sources": docs,
    }


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