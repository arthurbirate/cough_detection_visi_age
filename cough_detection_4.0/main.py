from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

app = FastAPI()

class CoughData(BaseModel):
    date: str
    time: str
    probability: float

@app.post("/cough_data/")
async def receive_cough_data(data: CoughData):
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Received cough data at {current_datetime}")
    print(f"Date: {data.date}, Time: {data.time}, Probability: {data.probability}%")

    # Here you can add your own logic to process or store the received data

    return {"message": "Cough data received successfully"}
