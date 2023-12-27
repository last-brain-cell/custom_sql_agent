import uvicorn
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from custom_agent import CustomAgent
from langchain.llms.vertexai import VertexAI
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
        verbose=False)


ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["answer_to_everything"] = agent_config()
    yield
    ml_models.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/predict")
async def predict(prompt: str):
    try:
        agent = ml_models["answer_to_everything"]
        result = agent.run(prompt)
        print(result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app="main:app", port=8000, reload=True)
