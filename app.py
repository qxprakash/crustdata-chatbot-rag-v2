import streamlit as st
import os
import dotenv
import uuid

# check if it's linux so it works on Streamlit Cloud
if os.name == "posix":
    __import__("pysqlite3")
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, AIMessage
from utils.constants import MODELS, DEFAULT_RAG_URLS
from utils.rag_utils import (
    load_doc_to_db,
    load_url_to_db,
    stream_llm_response,
    stream_llm_rag_response,
)

dotenv.load_dotenv()

st.set_page_config(
    page_title="RAG LLM app?",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)


# --- Header ---
st.html("""<h2 style="text-align: center;">📚🔍 <i> crustdata chatbot </i> 🤖💬</h2>""")


# --- Initial Setup ---
# Initialize all session state variables first
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "default_urls_loaded" not in st.session_state:
    st.session_state.default_urls_loaded = False

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize API keys in session state
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY")

if "anthropic_api_key" not in st.session_state:
    st.session_state.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")


# Now handle the default URL loading
if not st.session_state.default_urls_loaded:
    with st.spinner("Loading default documentation..."):
        for url in DEFAULT_RAG_URLS:
            st.session_state.rag_url = url
            load_url_to_db()
        st.session_state.default_urls_loaded = True
    st.session_state.rag_url = ""


# --- Side Bar LLM API Tokens ---
with st.sidebar:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    # Store API keys in session state
    st.session_state.openai_api_key = openai_api_key
    st.session_state.anthropic_api_key = anthropic_api_key


# --- Main Content ---
# Checking if the user has introduced the OpenAI API Key, if not, a warning is displayed
missing_openai = not openai_api_key or "sk-" not in openai_api_key
missing_anthropic = not anthropic_api_key
if missing_openai and missing_anthropic:
    st.error("No API keys configured. sorry !.")
    st.stop()

else:
    # Sidebar
    with st.sidebar:
        st.divider()
        models = []
        for model in MODELS:
            if "openai" in model and not missing_openai:
                models.append(model)
            elif "anthropic" in model and not missing_anthropic:
                models.append(model)

        if not models:
            st.error("No models available with current configuration")
            st.stop()

        selected_model = st.selectbox(
            "🤖 Current Model",
            options=models,
            key="model",
            disabled=True if len(models) == 1 else False,
        )

        st.info(f"Using: {selected_model}")

        cols0 = st.columns(2)
        with cols0[0]:
            is_vector_db_loaded = (
                "vector_db" in st.session_state
                and st.session_state.vector_db is not None
            )
            st.toggle(
                "Use RAG",
                value=is_vector_db_loaded,
                key="use_rag",
                disabled=not is_vector_db_loaded,
            )

        with cols0[1]:
            st.button(
                "Clear Chat",
                on_click=lambda: st.session_state.messages.clear(),
                type="primary",
            )

        st.header("RAG Sources:")

        # File upload input for RAG with documents
        st.file_uploader(
            "📄 Injest additional Data Sources",
            type=["pdf", "txt", "docx", "md"],
            accept_multiple_files=True,
            on_change=load_doc_to_db,
            key="rag_docs",
        )

        # URL input for RAG with websites
        st.text_input(
            "🌐 Add Data Sources From URL",
            placeholder="https://example.com",
            on_change=load_url_to_db,
            key="rag_url",
        )

        with st.expander(
            f"📚 Documents in DB ({0 if not is_vector_db_loaded else len(st.session_state.rag_sources)})"
        ):
            st.write(
                []
                if not is_vector_db_loaded
                else [source for source in st.session_state.rag_sources]
            )

    # Main chat app
    model_provider = st.session_state.model.split("/")[0]
    if model_provider == "openai":
        llm_stream = ChatOpenAI(
            api_key=openai_api_key,
            model_name=st.session_state.model.split("/")[-1],
            temperature=0.3,
            streaming=True,
        )
    elif model_provider == "anthropic":
        llm_stream = ChatAnthropic(
            api_key=anthropic_api_key,
            model=st.session_state.model.split("/")[-1],
            temperature=0.3,
            streaming=True,
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Your message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            messages = [
                HumanMessage(content=m["content"])
                if m["role"] == "user"
                else AIMessage(content=m["content"])
                for m in st.session_state.messages
            ]

            if not st.session_state.use_rag:
                st.write_stream(stream_llm_response(llm_stream, messages))
            else:
                st.write_stream(stream_llm_rag_response(llm_stream, messages))
