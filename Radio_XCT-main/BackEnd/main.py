from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from utils import handle_prediction
import shutil
import os
import uuid

app = FastAPI()

# CORS for React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or your React URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(
    name: str = Form(...),
    dob: str = Form(...),
    gender: str = Form(...),
    age: int = Form(...),
    organ: str = Form(...),
    file: UploadFile = File(...)
):
    # Save uploaded file
    unique_name = str(uuid.uuid4()) + "_" + file.filename
    upload_path = os.path.join("uploads", unique_name)
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Call your handler
    pdf_path = handle_prediction(organ, upload_path, name, dob, gender, age)

    return FileResponse(pdf_path, filename=os.path.basename(pdf_path), media_type='application/pdf')