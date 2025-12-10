import os
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_PATH = os.path.join(BASE_DIR, "speech.txt")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME = "mistral"

#Load text
def load_and_split() -> List:
    """
    Load speech.txt and split into semantic chunks.
    """
    loader = TextLoader(TEXT_PATH, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", " "]
    )
    chunks = splitter.split_documents(documents)
    return chunks

# Load embedding
def build_vectorstore(chunks):
    """
    Create a Chroma vectorstore backed by HuggingFace embeddings.
    """
    os.makedirs(CHROMA_DIR, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    if os.listdir(CHROMA_DIR):
        vectorstore = Chroma(
            collection_name="ambedkar_speech",
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
    else:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="ambedkar_speech",
            persist_directory=CHROMA_DIR,
        )
        vectorstore.persist()

    return vectorstore
# LLM and RAG chain

def build_rag_chain(vectorstore):
    """
    Build a Retrieval-Augmented Generation chain that:
    - Retrieves relevant chunks from Chroma.
    - Asks Ollama Mistral to answer *only* from those chunks.
    """

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    template = """
You are AmbedkarGPT, a Q&A assistant. Use ONLY the information in the provided 
context to answer the user's question. 
If the answer is not in the context, say clearly:
"I cannot answer this from the provided speech."

Context:
{context}

Question:
{question}

Answer in concise English.
"""
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOllama(
        model=OLLAMA_MODEL_NAME,
        temperature=0.1,
    )

    # RAG pipeline
    rag_chain = (
        RunnableParallel(
            context=retriever,
            question=RunnablePassthrough()
        )
        | (lambda x: {"context": format_docs(x["context"]), "question": x["question"]})
        | prompt
        | llm
    )

    return rag_chain

def format_docs(docs) -> str:
    """
    Join retrieved document chunks into a single context string.
    """
    return "\n\n".join(d.page_content for d in docs)



# Simple CLI loop
def interactive_cli(rag_chain):

    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not q:
            continue

        answer = rag_chain.invoke(q)
        print("\nAmbedkarGPT:", answer.content, "\n")

# Self-test with questions 
def run_self_test(rag_chain):
  
    sample_questions = [
        "What is the central theme of this speech?",
        "What does Ambedkar say about caste?",
    ]

    print("\n Self-test: 5 sample questions ---\n")
    for i, q in enumerate(sample_questions, start=1):
        print(f"Q{i}: {q}")
        ans = rag_chain.invoke(q)
        print(f"A{i}: {ans.content}\n")

# main
def main():
    if not os.path.exists(TEXT_PATH):
        raise FileNotFoundError(f"speech.txt not found at {TEXT_PATH}")

    chunks = load_and_split()
    vectorstore = build_vectorstore(chunks)
    rag_chain = build_rag_chain(vectorstore)

    run_self_test(rag_chain)

    interactive_cli(rag_chain)


if __name__ == "__main__":
    main()