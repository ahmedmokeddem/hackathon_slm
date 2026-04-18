
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI

import inference
from prompts import build_messages
from schemas import ChatRequest, ChatResponse

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    inference.load_model()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    messages = build_messages(payload.message, payload.tables)
    return ChatResponse(response=inference.generate(messages))
