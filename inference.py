import torch
from collections import Counter
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification
)


# Path to the trained model
MODEL_DIR = "model/fatura2_layoutlmv3"


# Select device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Load processor
processor = LayoutLMv3Processor.from_pretrained(
    MODEL_DIR,
    apply_ocr=False
)


# Load trained LayoutLMv3 model
model = LayoutLMv3ForTokenClassification.from_pretrained(
    MODEL_DIR
)

model.to(DEVICE)
model.eval()


def scale_bbox(bbox, width, height):
    """
    Convert image coordinates to LayoutLMv3's 0-1000 coordinate system.
    """

    x0, y0, x1, y1 = bbox

    return [
        max(0, min(1000, int(1000 * x0 / width))),
        max(0, min(1000, int(1000 * y0 / height))),
        max(0, min(1000, int(1000 * x1 / width))),
        max(0, min(1000, int(1000 * y1 / height)))
    ]


def predict_labels(image, ocr_words):
    """
    Run LayoutLMv3 on OCR words and bounding boxes.

    Args:
        image: PIL Image
        ocr_words: list of dictionaries containing
                    'word' and 'bbox'

    Returns:
        list of dictionaries containing word, bbox and predicted label.
    """

    image = image.convert("RGB")

    width, height = image.size

    words = [
        item["word"]
        for item in ocr_words
    ]

    boxes = [
        scale_bbox(
            item["bbox"],
            width,
            height
        )
        for item in ocr_words
    ]

    # Process image + OCR information
    encoding = processor(
        images=image,
        text=words,
        boxes=boxes,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )

    # Move tensors to the correct device
    inputs = {
        "input_ids": encoding["input_ids"].to(DEVICE),
        "attention_mask": encoding["attention_mask"].to(DEVICE),
        "bbox": encoding["bbox"].to(DEVICE),
        "pixel_values": encoding["pixel_values"].to(DEVICE)
    }

    # Run model
    with torch.no_grad():
        outputs = model(**inputs)

    predictions = outputs.logits.argmax(dim=-1)

    # Map LayoutLMv3 tokens back to OCR words
    word_ids = encoding.word_ids(batch_index=0)

    predicted_labels = []

    for word_idx, word in enumerate(ocr_words):

        token_positions = [
            i
            for i, word_id in enumerate(word_ids)
            if word_id == word_idx
        ]

        if not token_positions:
            continue

        token_predictions = [
            predictions[0, i].item()
            for i in token_positions
        ]

        # Majority vote when one OCR word becomes
        # multiple LayoutLMv3 subword tokens
        label_id = Counter(
            token_predictions
        ).most_common(1)[0][0]

        label = model.config.id2label[label_id]

        predicted_labels.append({
            "word": word["word"],
            "bbox": word["bbox"],
            "label": label
        })

    return predicted_labels