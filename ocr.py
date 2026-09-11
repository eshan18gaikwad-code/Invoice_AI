import numpy as np
from PIL import Image
from paddleocr import PaddleOCR


# Initialize OCR once when the application starts
ocr = PaddleOCR(
    lang="en",
    enable_mkldnn=False,
    return_word_box=True
)


def extract_words_and_boxes(image: Image.Image):
    """
    Run PaddleOCR on an invoice image.

    Returns:
        list[dict]: OCR words with bounding boxes.
    """

    image = image.convert("RGB")
    image_np = np.array(image)

    ocr_result = ocr.predict(image_np)
    result = ocr_result[0]

    ocr_words = []

    for word_parts, word_boxes in zip(
        result["text_word"],
        result["text_word_boxes"]
    ):
        for word, box in zip(word_parts, word_boxes):

            word = word.strip()

            # Ignore whitespace-only OCR pieces
            if not word:
                continue

            ocr_words.append({
                "word": word,
                "bbox": box.tolist()
            })

    return ocr_words