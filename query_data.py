import argparse
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq  

from get_embedding_function import get_embedding_function

load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # ✅ Initialize Groq LLM (replace with your Groq model name like "llama3-70b-8192" or "mixtral-8x7b-32768")
    model = ChatGroq(
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        api_key=os.environ["GROQ_API_KEY"]  # Loaded from .env file
    )

    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
