# GenAI Resume Chat API

This repository provides a FastAPI-based backend for querying information from a resume (PDF or TXT) using OpenAI and Pinecone for semantic search and chat-based Q&A.

## Features

- Extracts and chunks text from a resume document (PDF or TXT)
- Embeds user queries and performs semantic search using Pinecone
- Uses OpenAI's GPT model to answer questions based on the most relevant resume chunks
- REST API endpoint for chat-based interaction

## Requirements

- Python 3.8+
- OpenAI API key
- Pinecone API key
- FastAPI
- PyPDF2
- python-dotenv

## Setup

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd <repo-directory>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory and add your API keys:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   ```
4. Place your resume file (PDF or TXT) in the `genai/` directory and name it `resume.pdf` or `resume.txt`.

## Running the API

```bash
uvicorn genai.api_controller:app --reload
```

## API Usage

- **POST** `/chat`
  - Request body: `{ "query": "Your question about the resume" }`
  - Response: `{ "response": "Answer", "prompt": "..." }`

## Example

```bash
curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"query": "What is the candidate's work experience?"}'
```

## Notes

- Make sure your Pinecone index is set up and named `resume-embeddings` (or update the code accordingly).
- The API currently reads from `resume.pdf` in the `genai/` directory. Adjust the path in `api_controller.py` if needed.

## License

MIT
