from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from custom_agent import CustomAgent
from langchain.llms.vertexai import VertexAI
import os


def agent_config():
    return CustomAgent(database_url=os.getenv("DATABASE_URL"), llm=VertexAI(model='code-bison'), verbose=False)


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
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
