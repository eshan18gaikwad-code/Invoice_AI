# Invoice AI

Invoice information extraction system using PaddleOCR and LayoutLMv3.

## Project Overview

This project takes an invoice image and extracts structured invoice information.

Pipeline:

Invoice Image
    ↓
PaddleOCR
    ↓
Words + Bounding Boxes
    ↓
LayoutLMv3
    ↓
Predicted Labels
    ↓
Post-processing
    ↓
JSON

## Project Structure

invoice-ai/
│
├── model/
│   └── fatura2_layoutlmv3/
│
├── ocr.py
├── inference.py
├── postprocess.py
├── api.py
├── requirements.txt
└── README.md

## Model

The model is a fine-tuned LayoutLMv3 token-classification model trained on the FATURA2 invoice dataset.

The model directory must be:

model/fatura2_layoutlmv3/

Do not rename this directory unless MODEL_DIR in inference.py is also changed.

## OCR

PaddleOCR is used to extract:

- Text
- Word-level bounding boxes

OCR is initialized once when the application starts.

MKL-DNN is disabled because it caused an inference compatibility issue in the development environment.

## Inference

LayoutLMv3 receives:

- Invoice image
- OCR words
- Normalized bounding boxes

Bounding boxes are normalized to the LayoutLM coordinate range of 0–1000.

The model predicts an entity label for each OCR word.

## Post-processing

The predicted labels are converted into structured invoice information.

Current output format:

{
    "seller": "...",
    "buyer": {
        "name": "...",
        "email": "...",
        "phone": "...",
        "website": "..."
    },
    "invoice_number": "...",
    "date": "...",
    "due_date": "...",
    "total": "...",
    "total_words": "..."
}

## Installation

Create/activate a Python environment and install dependencies:

pip install -r requirements.txt

## Run the API

From the project root:

uvicorn api:app --reload

The API will run at:

http://127.0.0.1:8000

## API Documentation

After starting the server, open:

http://127.0.0.1:8000/docs

FastAPI provides an interactive Swagger interface.

## Prediction Endpoint

Endpoint:

POST /predict

Upload an invoice image using the `file` field.

The API runs the complete OCR → LayoutLMv3 → post-processing pipeline and returns JSON.

## Important Notes

The model was trained on the FATURA2 invoice dataset.

Performance can vary depending on invoice layout and design.

Invoices similar to the training data are expected to produce better results than completely different invoice formats.

The current system is a prototype for invoice information extraction and can be improved with additional training data, stronger post-processing, and validation rules.

## Development Environment

The project was tested on Windows.

The model can run on CPU, although GPU inference is recommended for better performance when available.

## Files

### ocr.py

Runs PaddleOCR and returns OCR words and bounding boxes.

### inference.py

Loads the trained LayoutLMv3 model and performs token classification.

### postprocess.py

Converts model predictions into structured invoice JSON.

### api.py

Exposes the invoice extraction pipeline through a FastAPI REST API.

### requirements.txt

Contains the Python dependencies required to run the project.