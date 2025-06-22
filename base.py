from openai import OpenAI
import os
from PyPDF2 import PdfReader
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# Required environment variables: OPENAI_API_KEY, PINECONE_API_KEY
client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY")
)
pinecone = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

def store_embeddings(embeddings, index_name="resume-embeddings"):
    """
    Store a list of embeddings in a Pinecone index.
    """
    index = pinecone.Index(index_name)

    # Prepare and upsert embeddings
    vectors = [
        (f"chunk-{i}", embedding)
        for i, embedding in enumerate(embeddings)
    ]
    index.upsert(vectors)

    print(f"Upserted {len(vectors)} embeddings to Pinecone index '{index_name}'")



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

def get_openai_embeddings(chunks, model="text-embedding-ada-002"):
    """
    Given a list of text chunks, return a list of embeddings using OpenAI API.
    """
    embeddings = []
    for chunk in chunks:
        response = client.embeddings.create(
            model=model,
            input=chunk
        )
        embedding = response.data[0].embedding
        embeddings.append(embedding)
    return embeddings

document_text = read_document("genai/resume.pdf")
document_chunks = chunk_text(document_text, chunk_size=100, overlap=25)
document_embeddings = get_openai_embeddings(document_chunks, model="text-embedding-ada-002")    
store_embeddings(document_embeddings, index_name="resume-embeddings")
