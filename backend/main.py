import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
from dotenv import load_dotenv
import asyncio

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for simplicity
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- API Key and Client Configuration ---
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY")

# DeepSeek Client
deepseek_client = openai.AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1"
) if DEEPSEEK_API_KEY and "YOUR_" not in DEEPSEEK_API_KEY else None

# Qwen Client (placeholder, assuming OpenAI-compatible endpoint)
qwen_client = openai.AsyncOpenAI(
    api_key=QWEN_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
) if QWEN_API_KEY and "YOUR_" not in QWEN_API_KEY else None

# OpenAI Client
openai_client = openai.AsyncOpenAI(
    api_key=OPENAI_API_KEY
) if OPENAI_API_KEY and "YOUR_" not in OPENAI_API_KEY else None

# Doubao Client (Guessed endpoint and model, please verify)
doubao_client = openai.AsyncOpenAI(
    api_key=DOUBAO_API_KEY,
    base_url="https://api.doubao.com/v1" # ASSUMPTION: Please verify this URL
) if DOUBAO_API_KEY and "YOUR_" not in DOUBAO_API_KEY else None


# --- Pydantic Models for data validation ---
class QueryRequest(BaseModel):
    prompt: str

class ModelResponse(BaseModel):
    model: str
    response: str
    error: str = None

# --- Asynchronous API Call Functions ---
async def query_deepseek(prompt: str):
    if not deepseek_client:
        return ModelResponse(model="DeepSeek", response="", error="API key not configured.")
    try:
        chat_completion = await deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
        )
        return ModelResponse(model="DeepSeek", response=chat_completion.choices[0].message.content)
    except Exception as e:
        return ModelResponse(model="DeepSeek", response="", error=str(e))

async def query_qwen(prompt: str):
    if not qwen_client:
        return ModelResponse(model="Qwen", response="", error="API key not configured.")
    try:
        chat_completion = await qwen_client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
        )
        return ModelResponse(model="Qwen", response=chat_completion.choices[0].message.content)
    except Exception as e:
        return ModelResponse(model="Qwen", response="", error=str(e))

async def query_openai(prompt: str):
    if not openai_client:
        return ModelResponse(model="OpenAI", response="", error="API key not configured.")
    try:
        chat_completion = await openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        return ModelResponse(model="OpenAI", response=chat_completion.choices[0].message.content)
    except Exception as e:
        return ModelResponse(model="OpenAI", response="", error=str(e))

async def query_doubao(prompt: str):
    if not doubao_client:
        return ModelResponse(model="Doubao", response="", error="API key not configured.")
    try:
        chat_completion = await doubao_client.chat.completions.create(
            model="doubao-pro", # ASSUMPTION: Please verify this model name
            messages=[{"role": "user", "content": prompt}],
        )
        return ModelResponse(model="Doubao", response=chat_completion.choices[0].message.content)
    except Exception as e:
        return ModelResponse(model="Doubao", response="", error=str(e))

# --- API Endpoint ---
@app.post("/api/query", response_model=list[ModelResponse])
async def run_queries(query: QueryRequest):
    """
    Receives a prompt and sends it to all configured models concurrently.
    """
    tasks = [
        query_deepseek(query.prompt),
        query_qwen(query.prompt),
        query_openai(query.prompt),
        query_doubao(query.prompt),
    ]
    
    results = await asyncio.gather(*tasks)
    return results

# --- Root endpoint for basic check ---
@app.get("/")
def read_root():
    return {"status": "Backend is running"}
