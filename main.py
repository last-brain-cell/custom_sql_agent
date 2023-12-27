import uvicorn
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from custom_agent import CustomAgent
from langchain.llms.vertexai import VertexAI
import time
import os
import dotenv

dotenv.load_dotenv(".env")


def agent_config():
    return CustomAgent(
            database_url=os.getenv("DATABASE_URL"),
            llm=VertexAI(
                model='gemini-pro',
                verbose=True
            ),
            verbose=False
        )


agents = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    start = time.perf_counter()
    agents["gemini"] = agent_config()
    end = time.perf_counter()
    print(f"Application started in {round(end - start):.2f}s")
    yield
    agents.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/predict")
async def predict(prompt: str):
    try:
        result = agents["gemini"] = agent_config()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def home():
    return {"message": "Custom SQL Agent Web Application"}


if __name__ == "__main__":
    uvicorn.run(app="main:app", port=8000, reload=True)
