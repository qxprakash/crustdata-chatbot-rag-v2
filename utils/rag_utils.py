import os
import dotenv
from time import time
import streamlit as st

from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    Docx2txtLoader,
)

# pip install docx2txt, pypdf
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from utils.prompts import CRUSTDATA_SYSTEM_PROMPT_WITH_RAG, RAG_PROMPT

dotenv.load_dotenv()

os.environ["USER_AGENT"] = "test_agent"
DOC_LIMIT = 10


def stream_llm_response(llm_stream, messages):
    # Add system message to the beginning of messages
    system_message = CRUSTDATA_SYSTEM_PROMPT_WITH_RAG

    full_messages = [{"role": "system", "content": system_message}] + messages

    response_message = ""
    for chunk in llm_stream.stream(full_messages):
        response_message += chunk.content
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})


def load_doc_to_db():
    # Use loader according to doc type
    if "rag_docs" in st.session_state and st.session_state.rag_docs:
        docs = []
        for doc_file in st.session_state.rag_docs:
            if doc_file.name not in st.session_state.rag_sources:
                if len(st.session_state.rag_sources) < DOC_LIMIT:
                    os.makedirs("source_files", exist_ok=True)
                    file_path = f"./source_files/{doc_file.name}"
                    with open(file_path, "wb") as file:
                        file.write(doc_file.read())

                    try:
                        if doc_file.type == "application/pdf":
                            loader = PyPDFLoader(file_path)
                        elif doc_file.name.endswith(".docx"):
                            loader = Docx2txtLoader(file_path)
                        elif doc_file.type in ["text/plain", "text/markdown"]:
                            loader = TextLoader(file_path)
                        else:
                            st.warning(f"Document type {doc_file.type} not supported.")
                            continue

                        docs.extend(loader.load())
                        st.session_state.rag_sources.append(doc_file.name)

                    except Exception as e:
                        st.toast(
                            f"Error loading document {doc_file.name}: {e}", icon="⚠️"
                        )
                        print(f"Error loading document {doc_file.name}: {e}")

                    finally:
                        os.remove(file_path)

                else:
                    st.error(f"Maximum number of documents reached ({DOC_LIMIT}).")

        if docs:
            _split_and_load_docs(docs)
            st.toast(
                f"Document *{str([doc_file.name for doc_file in st.session_state.rag_docs])[1:-1]}* loaded successfully.",
                icon="✅",
            )


def load_url_to_db():
    if "rag_url" in st.session_state and st.session_state.rag_url:
        url = st.session_state.rag_url
        docs = []
        if url not in st.session_state.rag_sources:
            if len(st.session_state.rag_sources) < 10:
                try:
                    loader = WebBaseLoader(url)
                    docs.extend(loader.load())
                    st.session_state.rag_sources.append(url)

                except Exception as e:
                    st.error(f"Error loading document from {url}: {e}")

                if docs:
                    _split_and_load_docs(docs)
                    st.toast(
                        f"Document from URL *{url}* loaded successfully.", icon="✅"
                    )

            else:
                st.error("Maximum number of documents reached (10).")


def initialize_vector_db(docs):
    api_key = st.session_state.openai_api_key
    if not api_key:
        st.error("OpenAI API key not found in environment variables")
        st.stop()
    embedding = OpenAIEmbeddings(api_key=api_key)

    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        collection_name=f"{str(time()).replace('.', '')[:14]}_"
        + st.session_state["session_id"],
    )

    # We need to manage the number of collections that we have in memory, we will keep the last 20
    chroma_client = vector_db._client
    collection_names = sorted(
        [collection.name for collection in chroma_client.list_collections()]
    )
    print("Number of collections:", len(collection_names))
    while len(collection_names) > 20:
        chroma_client.delete_collection(collection_names[0])
        collection_names.pop(0)

    return vector_db


def _split_and_load_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=1000,
    )

    document_chunks = text_splitter.split_documents(docs)

    if "vector_db" not in st.session_state:
        st.session_state.vector_db = initialize_vector_db(docs)
    else:
        st.session_state.vector_db.add_documents(document_chunks)


def _get_context_retriever_chain(vector_db, llm):
    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},  # Retrieve top 3 most relevant chunks
    )

    print("===================== get context retriever chain  ================")

    # def debug_and_retrieve(query):
    #     docs = retriever.aget_relevant_documents(query)
    #     print("\n=== Retrieved Documents ===")
    #     for i, doc in enumerate(docs, 1):
    #         print(f"\nDocument {i}:")
    #         print(f"Content: {doc.page_content[:200]}...")
    #         print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    #         print("-" * 50)
    #     return docs

    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="messages"),
            ("user", "{input}"),
            ("user", RAG_PROMPT),
        ]
    )

    print(f"prompt for retriever: {prompt}")

    print(f"retriever: {retriever}")
    # Create the retriever chain with the debug wrapper
    retriever_chain = create_history_aware_retriever(
        llm,
        retriever,
        prompt,
    )

    return retriever_chain


def get_conversational_rag_chain(llm):
    retriever_chain = _get_context_retriever_chain(st.session_state.vector_db, llm)
    print("\n=== Retriever Chain Output ===")
    print(f"Type: {type(retriever_chain)}")
    print(f"Content: {retriever_chain}")
    print("================================\n")

    prompt = prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a knowledgeable and helpful customer support agent for Crustdata. Your role is to assist users with technical questions about Crustdata’s APIs, providing accurate answers based on the official documentation and examples.

    If a user asks about API functionality, provide detailed explanations with example requests.
    If a user encounters errors, help troubleshoot and suggest solutions or resources.
    Be conversational and allow follow-up questions.
    Reference and validate any specific requirements or resources, such as standardized region values or API behavior.
    Always provide clear, concise, and actionable responses.

    Focus on delivering accurate information and guiding users effectively to achieve their goals with Crustdata’s APIs.

    {context}""",
            ),
            MessagesPlaceholder(variable_name="messages"),
            ("user", "{input}"),
        ]
    )

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def stream_llm_rag_response(llm_stream, messages):
    print("\n=== RAG Request Started ===")
    print(f"User Query: {messages[-1].content}")

    conversation_rag_chain = get_conversational_rag_chain(llm_stream)
    response_message = "*(RAG Response)*\n"

    # Add timing information
    start_time = time()

    for chunk in conversation_rag_chain.pick("answer").stream(
        {"messages": messages[:-1], "input": messages[-1].content}
    ):
        response_message += chunk
        yield chunk

    print(f"\nTotal RAG processing time: {time() - start_time:.2f} seconds")
    print("=== RAG Request Completed ===\n")
