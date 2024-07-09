from typing import Union
from fastapi import FastAPI
from model import main

app = FastAPI()

@app.get("/")
def read_root():
    return "Server is Up and Running..."

@app.get("/query/")
def read_item(q: str):
    return main(q)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)