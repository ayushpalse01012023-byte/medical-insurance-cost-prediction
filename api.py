from fastapi import FastAPI

app = FastAPI()
def home():
    return {"message": "Welcome To my medical cost prediction app "}