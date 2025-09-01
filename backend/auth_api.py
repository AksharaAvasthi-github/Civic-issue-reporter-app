from fastapi import FastAPI

app = FastAPI()

@app.post("/signup")
def signup():
    return {"message": "signup works"}

@app.post("/login")
def login():
    return {"message": "login works"}
