from fastapi import FastAPI

app = FastAPI()
@app.get("/home")
def home():
    return {"message": "Welcome To my medical cost prediction app "}