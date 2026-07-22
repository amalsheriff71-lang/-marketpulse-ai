"""
streamlit_app.py
----------------
MarketPulse AI
Premium AI-powered Marketing Intelligence Dashboard

Deployment:
- Chroma store is downloaded automatically from Hugging Face
  when it is not available locally.
- GROQ_API_KEY and HF_TOKEN are read from Streamlit Secrets.
"""

import os
import importlib.util

import streamlit as st
from huggingface_hub import snapshot_download


# ============================================================
# HTML RENDER HELPER
# ============================================================
#
# BUG FIX (root cause of raw "<div class=...>" text showing up
# in the deployed app instead of rendered HTML/cards):
#
# Streamlit's st.markdown() runs content through a Markdown
# parser *before* honoring unsafe_allow_html=True. Markdown's
# spec treats any line indented by 4+ spaces (especially one
# following a blank line) as the start of an "indented code
# block", and renders it verbatim as literal code instead of
# parsing it as HTML.
#
# Every HTML block in this file was written as an indented,
# multi-line triple-quoted string (natural for readability), and
# that indentation is exactly what triggered Markdown's
# code-block rule -> the <div> markup was displayed as literal
# text in a code box instead of being rendered as HTML.
#
# unsafe_allow_html=True was already set correctly everywhere;
# it was never the actual problem. The fix is to strip leading
# whitespace from every line of HTML right before handing it to
# st.markdown(), so Markdown never sees a 4-space indent.
#
def render_html(html: str) -> None:
    """
    Render a block of raw HTML safely in Streamlit.

    Strips per-line leading/trailing whitespace so Markdown's
    "indented code block" rule never triggers, then renders the
    HTML via st.markdown(unsafe_allow_html=True).
    """
    lines = [line.strip() for line in html.strip("\n").splitlines()]
    st.markdown("\n".join(lines), unsafe_allow_html=True)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MarketPulse AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# HUGGING FACE CHROMA CONFIG
# ============================================================

HF_REPO_ID = "amal-sherif71/marketpulse-chroma"
HF_CHROMA_SUBFOLDER = "chroma_store"


# ============================================================
# DYNAMIC IMPORT
# ============================================================

def load_module(filename, module_name):

    path = os.path.join(
        BASE_DIR,
        filename,
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found: {filename}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load {filename}"
        )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


# ============================================================
# LOAD PROJECT MODULES
# ============================================================

try:

    prompting_module = load_module(
        "07_prompting.py",
        "prompting_module",
    )

except Exception as e:

    st.error(
        "❌ Failed to load MarketPulse AI modules."
    )

    with st.expander("Show technical error"):
        st.exception(e)

    st.stop()


# ============================================================
# PROJECT FUNCTIONS
# ============================================================

load_vectorstore = prompting_module.load_vectorstore
generate_answer = prompting_module.generate_answer
get_llm = prompting_module.get_llm


# ============================================================
# PREMIUM CSS
# ============================================================

render_html(
    """
    <style>

    .stApp {
    background:
    radial-gradient(
    circle at 8% 0%,
    rgba(99, 102, 241, 0.16),
    transparent 28%
    ),
    radial-gradient(
    circle at 92% 8%,
    rgba(168, 85, 247, 0.12),
    transparent 25%
    ),
    linear-gradient(
    135deg,
    #070b16 0%,
    #0b1020 48%,
    #10162a 100%
    );

    color: #f8fafc;
    }

    header[data-testid="stHeader"] {
    background: transparent;
    }

    .main .block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 4rem;
    }

    section[data-testid="stSidebar"] {
    background:
    linear-gradient(
    180deg,
    #080c18 0%,
    #0d1324 100%
    );

    border-right:
    1px solid
    rgba(255,255,255,0.07);
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
    color: #f8fafc;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li {
    color: #94a3b8;
    }

    h1,
    h2,
    h3,
    h4 {
    letter-spacing: -0.025em;
    }

    h1 {
    font-weight: 800 !important;
    }

    .hero-container {
    padding: 3.2rem 3rem;
    border-radius: 28px;

    background:
    linear-gradient(
    135deg,
    rgba(30,41,59,0.94),
    rgba(15,23,42,0.98)
    );

    border:
    1px solid
    rgba(148,163,184,0.14);

    box-shadow:
    0 30px 80px
    rgba(0,0,0,0.35);

    margin-bottom: 2rem;
    }

    .hero-badge-native {
    display: inline-block;
    padding: 0.45rem 0.9rem;
    border-radius: 999px;

    background:
    rgba(99,102,241,0.14);

    border:
    1px solid
    rgba(129,140,248,0.28);

    color: #c7d2fe;
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    margin-bottom: 1rem;
    }

    .hero-title-native {
    font-size: clamp(2rem, 4vw, 3.4rem);
    line-height: 1.08;
    font-weight: 850;
    color: #ffffff;
    margin-bottom: 1rem;
    }

    .hero-highlight {
    background:
    linear-gradient(
    90deg,
    #818cf8,
    #c4b5fd
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    }

    .hero-subtitle-native {
    font-size: 1.05rem;
    line-height: 1.8;
    color: #94a3b8;
    max-width: 900px;
    }

    .kpi-card-native {
    padding: 1.35rem;
    min-height: 135px;
    border-radius: 20px;

    background:
    linear-gradient(
    145deg,
    rgba(30,41,59,0.82),
    rgba(15,23,42,0.84)
    );

    border:
    1px solid
    rgba(148,163,184,0.13);

    box-shadow:
    0 15px 40px
    rgba(0,0,0,0.16);

    transition:
    transform 0.2s ease,
    box-shadow 0.2s ease,
    border-color 0.2s ease;
    }

    .kpi-card-native:hover {
    transform: translateY(-4px);

    border-color:
    rgba(129,140,248,0.45);

    box-shadow:
    0 20px 50px
    rgba(99,102,241,0.18);
    }

    .kpi-icon-native {
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
    }

    .kpi-title-native {
    color: #94a3b8;
    font-size: 0.75rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    }

    .kpi-text-native {
    color: #f8fafc;
    font-size: 1rem;
    font-weight: 700;
    margin-top: 0.35rem;
    }

    .section-eyebrow-native {
    color: #818cf8;
    font-size: 0.72rem;
    font-weight: 850;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    margin-top: 1.8rem;
    margin-bottom: 0.35rem;
    }

    .explore-card-native {
    min-height: 195px;
    padding: 1.25rem;
    border-radius: 20px;

    background:
    linear-gradient(
    145deg,
    rgba(30,41,59,0.82),
    rgba(15,23,42,0.82)
    );

    border:
    1px solid
    rgba(148,163,184,0.12);

    box-shadow:
    0 15px 40px
    rgba(0,0,0,0.14);

    transition:
    transform 0.2s ease,
    box-shadow 0.2s ease,
    border-color 0.2s ease;
    }

    .explore-card-native:hover {
    transform: translateY(-4px);

    border-color:
    rgba(168,85,247,0.45);

    box-shadow:
    0 20px 50px
    rgba(168,85,247,0.18);
    }

    .explore-icon-native {
    font-size: 2rem;
    margin-bottom: 0.65rem;
    }

    .explore-title-native {
    color: #f8fafc;
    font-size: 0.98rem;
    font-weight: 800;
    margin-bottom: 0.45rem;
    }

    .explore-description-native {
    color: #94a3b8;
    font-size: 0.80rem;
    line-height: 1.6;
    }

    .stButton > button {
    min-height: 44px;
    border-radius: 12px;
    font-weight: 700;

    border:
    1px solid
    rgba(148,163,184,0.18);

    background:
    rgba(30,41,59,0.72);

    color: #f8fafc;

    transition: all 0.2s ease;
    }

    .stButton > button:hover {
    transform: translateY(-2px);

    border-color:
    rgba(129,140,248,0.55);

    background:
    rgba(51,65,85,0.92);
    }

    button[kind="primary"] {
    background:
    linear-gradient(
    135deg,
    #6366f1,
    #8b5cf6
    ) !important;

    border: none !important;
    color: white !important;

    box-shadow:
    0 12px 30px
    rgba(99,102,241,0.25);
    }

    button[kind="primary"]:hover {
    background:
    linear-gradient(
    135deg,
    #818cf8,
    #a78bfa
    ) !important;

    box-shadow:
    0 15px 35px
    rgba(99,102,241,0.35);
    }

    textarea {
    border-radius: 18px !important;

    background:
    rgba(15,23,42,0.88) !important;

    color: #f8fafc !important;

    border:
    1px solid
    rgba(148,163,184,0.16) !important;
    }

    textarea:focus {
    border:
    1px solid
    rgba(129,140,248,0.7) !important;

    box-shadow:
    0 0 0 1px
    rgba(129,140,248,0.25) !important;
    }

    .result-header-native {
    padding: 1.5rem 1.7rem;
    border-radius: 20px;

    background:
    linear-gradient(
    135deg,
    rgba(30,41,59,0.84),
    rgba(15,23,42,0.86)
    );

    border:
    1px solid
    rgba(129,140,248,0.18);

    margin-bottom: 1rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
    padding: 0.4rem;
    border-radius: 22px !important;

    background:
    rgba(15,23,42,0.84);

    border:
    1px solid
    rgba(148,163,184,0.13) !important;

    box-shadow:
    0 20px 55px
    rgba(0,0,0,0.18);
    }

    .insight-label-native {
    color: #a5b4fc;
    font-size: 0.75rem;
    font-weight: 850;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    margin-bottom: 0.8rem;
    }

    .action-card-native {
    padding: 1.25rem;
    border-radius: 16px;

    background:
    rgba(30,41,59,0.68);

    border:
    1px solid
    rgba(148,163,184,0.10);

    margin-top: 1rem;
    }

    .empty-state-native {
    padding: 3rem;
    text-align: center;
    border-radius: 24px;

    background:
    rgba(15,23,42,0.55);

    border:
    1px dashed
    rgba(148,163,184,0.20);

    margin-top: 1.5rem;
    }

    .empty-icon-native {
    font-size: 3rem;
    margin-bottom: 0.8rem;
    }

    .empty-title-native {
    font-size: 1.2rem;
    font-weight: 800;
    color: #f8fafc;
    }

    .empty-description-native {
    color: #94a3b8;
    max-width: 650px;
    margin: 0.5rem auto 0;
    }

    .footer-native {
    text-align: center;
    color: #64748b;
    font-size: 0.78rem;
    padding-top: 3rem;
    margin-top: 4rem;

    border-top:
    1px solid
    rgba(148,163,184,0.08);
    }

    @media (max-width: 768px) {

    .hero-container {
    padding: 2rem 1.4rem;
    }

    .hero-title-native {
    font-size: 2rem;
    }

    .kpi-card-native {
    min-height: 110px;
    }

    }

    </style>
    """
)


# ============================================================
# SESSION STATE
# ============================================================

if "query" not in st.session_state:
    st.session_state.query = ""

if "result" not in st.session_state:
    st.session_state.result = None

if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

if "auto_analyze" not in st.session_state:
    st.session_state.auto_analyze = False


# ============================================================
# LOAD SECRETS
# ============================================================

try:

    groq_api_key = st.secrets["GROQ_API_KEY"]

    os.environ["GROQ_API_KEY"] = groq_api_key

except Exception:

    groq_api_key = None


try:

    hf_token = st.secrets["HF_TOKEN"]

    os.environ["HF_TOKEN"] = hf_token

except Exception:

    hf_token = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🚀 MarketPulse AI")

    st.caption(
        "AI-powered Marketing Intelligence"
    )

    st.divider()

    st.markdown(
        "### 💡 Explore Insights"
    )

    st.markdown(
        """
        📈 **Top Performing Content**

        🔥 **Engagement Drivers**

        👥 **Audience Trends**

        💡 **Content Opportunities**

        🏆 **Platform Performance**
        """
    )

    st.divider()

    st.markdown(
        "### 🧠 How It Works"
    )

    st.markdown(
        """
        **01 — Ask**

        Ask a marketing question.

        **02 — Retrieve**

        Relevant marketing data is retrieved.

        **03 — Analyze**

        AI analyzes the retrieved evidence.

        **04 — Act**

        Turn insights into actionable decisions.
        """
    )

    st.divider()

    st.caption(
        "Powered by RAG + Chroma + Groq"
    )


# ============================================================
# HERO
# ============================================================

render_html(
    """
    <div class="hero-container">

    <div class="hero-badge-native">
    ✨ AI-POWERED MARKETING INTELLIGENCE
    </div>

    <div class="hero-title-native">
    Turn Social Media Data<br>

    <span class="hero-highlight">
    Into Your Next Marketing Move.
    </span>
    </div>

    <div class="hero-subtitle-native">
    MarketPulse AI transforms your marketing knowledge base
    into actionable intelligence. Discover what drives engagement,
    understand audience behavior, identify content opportunities,
    and make smarter decisions — grounded in your own data.
    </div>

    </div>
    """
)


# ============================================================
# KPI CARDS
# ============================================================

kpi_cols = st.columns(4)

kpi_data = [

    (
        "🧠",
        "AI Intelligence",
        "Evidence-grounded analysis",
    ),

    (
        "🔎",
        "RAG-Powered",
        "Answers from your data",
    ),

    (
        "⚡",
        "Actionable",
        "Insights built for decisions",
    ),

    (
        "🚀",
        "Marketing Focused",
        "Built for social intelligence",
    ),

]


for col, item in zip(
    kpi_cols,
    kpi_data,
):

    with col:

        render_html(
            f"""
            <div class="kpi-card-native">

            <div class="kpi-icon-native">
            {item[0]}
            </div>

            <div class="kpi-title-native">
            {item[1]}
            </div>

            <div class="kpi-text-native">
            {item[2]}
            </div>

            </div>
            """
        )


st.markdown("")


# ============================================================
# EXPLORE SECTION
# ============================================================

st.markdown(
    '<div class="section-eyebrow-native">DISCOVER INSIGHTS</div>',
    unsafe_allow_html=True,
)

st.markdown(
    "## 💡 Explore Your Marketing Intelligence"
)

st.caption(
    "Start with a focused insight or ask MarketPulse AI your own question."
)


# ============================================================
# QUICK INSIGHTS
# ============================================================

quick_questions = [

    (
        "📈",
        "Top Performing Content",
        "Which content performs best based on engagement?",
        "Identify the content that generates the strongest engagement signals.",
    ),

    (
        "🔥",
        "Engagement Drivers",
        "What factors drive the most engagement in the social media data?",
        "Discover the patterns and factors behind audience engagement.",
    ),

    (
        "👥",
        "Audience Trends",
        "What audience trends can be identified from the data?",
        "Understand emerging audience behaviors and preferences.",
    ),

    (
        "💡",
        "Content Opportunities",
        "What new content opportunities can we identify from the data?",
        "Find promising content directions based on your evidence.",
    ),

    (
        "🏆",
        "Platform Performance",
        "Which social media platform performs best and why?",
        "Compare platform performance and identify where to focus.",
    ),

]


columns = st.columns(5)


for i, item in enumerate(
    quick_questions
):

    icon = item[0]
    title = item[1]
    question = item[2]
    description = item[3]

    with columns[i]:

        render_html(
            f"""
            <div class="explore-card-native">

            <div class="explore-icon-native">
            {icon}
            </div>

            <div class="explore-title-native">
            {title}
            </div>

            <div class="explore-description-native">
            {description}
            </div>

            </div>
            """
        )

        st.markdown("")

        if st.button(
            "Explore →",
            key=f"quick_{i}",
            use_container_width=True,
        ):

            st.session_state.query = question
            st.session_state.result = None
            st.session_state.analysis_complete = False
            st.session_state.auto_analyze = True

            st.rerun()


# ============================================================
# ASK MARKETPULSE AI
# ============================================================

st.markdown("")

st.markdown(
    '<div class="section-eyebrow-native">ASK YOUR DATA</div>',
    unsafe_allow_html=True,
)

st.markdown(
    "## 🧠 Ask MarketPulse AI"
)

st.caption(
    "Ask a question about your marketing data and get an evidence-grounded AI analysis."
)


query = st.text_area(
    "Marketing Question",
    value=st.session_state.query,
    placeholder=(
        "Example: Which content themes drive the most engagement?"
    ),
    height=120,
    label_visibility="collapsed",
)


st.session_state.query = query


# ============================================================
# SUGGESTED QUESTIONS
# ============================================================

st.markdown(
    "**💬 Try asking:**"
)

suggested_questions = [

    "What content drives the most engagement?",

    "Which platform performs best?",

    "What content opportunities should we explore?",

    "What are the strongest engagement drivers?",

]


suggestion_columns = st.columns(4)


for i, question in enumerate(
    suggested_questions
):

    with suggestion_columns[i]:

        if st.button(
            question,
            key=f"suggestion_{i}",
            use_container_width=True,
        ):

            st.session_state.query = question
            st.session_state.result = None
            st.session_state.analysis_complete = False
            st.session_state.auto_analyze = True

            st.rerun()


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.markdown("")

analyze_clicked = st.button(
    "✨ Analyze & Discover Insights",
    type="primary",
    use_container_width=True,
)


# ============================================================
# DOWNLOAD CHROMA FROM HUGGING FACE
# ============================================================

@st.cache_resource(show_spinner=False)
def get_chroma_directory():

    local_chroma = os.path.join(
        BASE_DIR,
        "chroma_store",
    )

    if os.path.isdir(local_chroma):

        return local_chroma


    try:

        hf_token = st.secrets["HF_TOKEN"]

    except Exception:

        raise RuntimeError(
            "HF_TOKEN is missing from Streamlit Secrets."
        )


    with st.spinner(
        "☁️ Downloading MarketPulse AI knowledge base..."
    ):

        snapshot_path = snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            token=hf_token,
            allow_patterns=[
                "chroma_store/**",
            ],
        )


    chroma_directory = os.path.join(
        snapshot_path,
        HF_CHROMA_SUBFOLDER,
    )


    if not os.path.isdir(chroma_directory):

        raise FileNotFoundError(
            "Downloaded Chroma store was not found."
        )


    return chroma_directory


# ============================================================
# LOAD VECTOR STORE
# ============================================================

@st.cache_resource(
    show_spinner="🚀 Loading MarketPulse AI intelligence engine..."
)
def get_vectorstore():

    chroma_directory = get_chroma_directory()

    return load_vectorstore(
        chroma_directory
    )


# ============================================================
# INITIALIZE VECTOR STORE
# ============================================================

try:

    vectorstore = get_vectorstore()

except Exception as e:

    st.error(
        "❌ Failed to load the Chroma vector store."
    )

    st.markdown(
        """
        ### 🔧 Deployment Diagnostics

        MarketPulse AI could not load the marketing
        knowledge base.

        Please verify:

        1. `HF_TOKEN` exists in Streamlit Secrets.
        2. The Hugging Face repository is accessible.
        3. The repository is:
           `amal-sherif71/marketpulse-chroma`
        4. The Chroma files were uploaded successfully.
        """
    )

    with st.expander(
        "Show technical error"
    ):

        st.exception(e)

    st.stop()


# ============================================================
# RUN ANALYSIS
# ============================================================

should_analyze = (
    analyze_clicked
    or st.session_state.auto_analyze
)


if should_analyze:

    current_query = (
        st.session_state.query
        or ""
    ).strip()


    if not groq_api_key:

        st.error(
            "❌ GROQ_API_KEY is missing."
        )

        st.info(
            "Add GROQ_API_KEY to Streamlit Secrets."
        )

        st.session_state.auto_analyze = False


    elif not current_query:

        st.warning(
            "💡 Enter a marketing question "
            "before running the analysis."
        )

        st.session_state.auto_analyze = False


    else:

        with st.spinner(
            "🧠 MarketPulse AI is analyzing your marketing data..."
        ):

            try:

                llm = get_llm(
                    api_key=groq_api_key
                )


                result = generate_answer(
                    current_query,
                    vectorstore,
                    llm,
                    k=5,
                )


                st.session_state.result = result

                st.session_state.analysis_complete = True

                st.session_state.auto_analyze = False

                st.rerun()


            except Exception as e:

                st.session_state.result = None

                st.session_state.analysis_complete = False

                st.session_state.auto_analyze = False

                st.error(
                    "❌ Something went wrong "
                    "while generating the analysis."
                )

                with st.expander(
                    "Show technical error"
                ):

                    st.exception(e)


# ============================================================
# DISPLAY RESULTS
# ============================================================

if (
    st.session_state.analysis_complete
    and st.session_state.result
):

    result = st.session_state.result


    if not isinstance(
        result,
        dict,
    ):

        st.error(
            "❌ Unexpected response format "
            "from the AI engine."
        )

        st.write(result)


    else:

        answer = result.get(
            "answer",
            "",
        )


        sources = result.get(
            "sources",
            [],
        )


        st.divider()


        render_html(
            """
            <div class="result-header-native">

            <div class="section-eyebrow-native">
            ANALYSIS COMPLETE
            </div>

            <h2>
            🎯 Your Marketing Intelligence
            </h2>

            <p style="color:#94a3b8;">
            AI-generated insights grounded in the
            relevant data retrieved from your knowledge base.
            </p>

            </div>
            """
        )


        render_html(
            """
            iv class="insight-label-native">'
            '🧠 AI Marketing Analysis'
            '</di
            """
        )


        with st.container(
            border=True
        ):

            if answer:

                st.markdown(answer)

            else:

                st.warning(
                    "No AI analysis was returned."
                )


        render_html(
            """
            <div class="action-card-native">

            <strong>
            🚀 Recommended Next Step
            </strong>

            <br><br>

            Use the analysis above as a decision-making
            starting point, then validate the strongest
            patterns against the supporting evidence below.

            </div>
            """
        )


        st.markdown("")


        render_html(
            """
            iv class="section-eyebrow-native">'
            'EVIDENCE LAYER'
            '</di
            """
        )


        st.markdown(
            "## 🔎 Supporting Evidence"
        )


        st.caption(
            "MarketPulse AI grounds its analysis in the "
            "original documents retrieved from your "
            "marketing knowledge base."
        )


        if not sources:

            st.info(
                "No supporting sources were retrieved."
            )


        else:

            for i, doc in enumerate(
                sources,
                start=1,
            ):

                metadata = (
                    getattr(
                        doc,
                        "metadata",
                        {}
                    )
                    or {}
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


                label = (
                    f"🔎 Source {i}  •  "
                    f"{platform}  •  "
                    f"{category}"
                )


                with st.expander(
                    label
                ):

                    col1, col2, col3 = (
                        st.columns(3)
                    )


                    with col1:

                        st.caption(
                            "PLATFORM"
                        )

                        st.write(
                            platform
                        )


                    with col2:

                        st.caption(
                            "CATEGORY"
                        )

                        st.write(
                            category
                        )


                    with col3:

                        st.caption(
                            "CONTENT ID"
                        )

                        st.write(
                            content_id
                        )


                    st.divider()


                    st.markdown(
                        "**Original Retrieved Content**"
                    )


                    st.write(
                        getattr(
                            doc,
                            "page_content",
                            "No content available.",
                        )
                    )


# ============================================================
# EMPTY STATE
# ============================================================

if not st.session_state.analysis_complete:

    render_html(
        """
        <div class="empty-state-native">

        <div class="empty-icon-native">
        🧠
        </div>

        <div class="empty-title-native">
        Your Marketing Intelligence Starts Here
        </div>

        <div class="empty-description-native">
        Choose an insight above or ask your own question.
        MarketPulse AI will retrieve relevant evidence,
        analyze it with AI, and surface actionable insights.
        </div>

        </div>
        """
    )


# ============================================================
# FOOTER
# ============================================================

render_html(
    """
    <div class="footer-native">

    MarketPulse AI · AI-powered Marketing Intelligence

    <br>

    Retrieval-Augmented Generation · Chroma · Groq

    </div>
    """
)