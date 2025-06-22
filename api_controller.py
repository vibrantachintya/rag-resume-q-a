from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY")
)
pinecone = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

def read_document(file_path):
    """Read text from a document (txt or pdf file)."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif ext == '.pdf':
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Split text into chunks of chunk_size with optional overlap.
    Returns a list of text chunks.
    """
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


class ChatRequest(BaseModel):
    query: str

def get_query_embedding(query, model="text-embedding-ada-002"):
    response = client.embeddings.create(
        model=model,
        input=query
    )
    return response.data[0].embedding

def fetch_relevant_chunks(query_embedding, index_name="resume-embeddings", top_k=20):
    index = pinecone.Index(index_name)
    # Pinecone expects a list of vectors for querying
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=False)
    # Each match has an 'id' and 'score'
    chunk_ids = [match['id'] for match in results['matches']]
    return chunk_ids

def get_chunks_by_ids(chunk_ids, all_chunks):
    # chunk_ids are like "chunk-0", "chunk-1", etc.
    indices = [int(cid.split('-')[1]) for cid in chunk_ids if cid.startswith("chunk-")]
    return [all_chunks[i] for i in indices if i < len(all_chunks)]


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_query = request.query

    # Step 1: Get embedding for the user query
    query_embedding = get_query_embedding(user_query)

    document_text = read_document("resume.pdf")
    document_chunks = chunk_text(document_text, chunk_size=100, overlap=25)

    # Step 2: Semantic search in Pinecone for relevant chunks
    chunk_ids = fetch_relevant_chunks(query_embedding)

    chunks = get_chunks_by_ids(chunk_ids, document_chunks)

    context = "\n".join(chunks)

    # Step 5: Make an OpenAI completion call with the context and user query
    prompt = f"resume info in form of embeddings:\n{context}\n\nUser Query: {user_query}\nAnswer:"
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "user", "content": prompt}
        ]
    )
    answer = completion.choices[0].message.content

    return {"response": answer, "prompt": prompt}

