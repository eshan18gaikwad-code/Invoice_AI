from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

from ocr import extract_words_and_boxes
from inference import predict_labels
from postprocess import postprocess_invoice


app = FastAPI(
    title="Invoice AI API",
    description="Invoice information extraction using PaddleOCR and LayoutLMv3",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "Invoice AI API is running"
    }


@app.post("/predict")
async def predict_invoice(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    ocr_words = extract_words_and_boxes(image)

    predicted_labels = predict_labels(
        image,
        ocr_words
    )

    result = postprocess_invoice(
        predicted_labels,
        ocr_words
    )

    return result