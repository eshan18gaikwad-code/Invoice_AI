import re


def postprocess_invoice(predicted_labels, ocr_words):

    # ==========================================
    # Helper
    # ==========================================
    def words_for_label(label):
        return [
            item["word"]
            for item in predicted_labels
            if item["label"] == label
        ]

    # ==========================================
    # Find BILL_TO boundary
    # ==========================================
    bill_start = None

    for i, item in enumerate(ocr_words):
        if item["word"].upper() == "BILL":
            bill_start = i
            break

    # ==========================================
    # SELLER
    # ==========================================
    seller_words = []

    for item in predicted_labels:

        if item["label"] != "SELLER":
            continue

        for i, ocr_item in enumerate(ocr_words):

            if (
                ocr_item["word"] == item["word"]
                and (
                    bill_start is None
                    or i < bill_start
                )
            ):
                seller_words.append(item["word"])
                break

    seller = " ".join(seller_words)

    seller = re.sub(
        r"\s+([,.])",
        r"\1",
        seller
    )

    # ==========================================
    # DATE
    # ==========================================
    date_match = re.search(
        r"\d{1,2}-[A-Za-z]{3}-\d{4}",
        " ".join(words_for_label("DATE"))
    )

    date = (
        date_match.group(0)
        if date_match
        else None
    )

    # ==========================================
    # DUE DATE
    # ==========================================
    due_match = re.search(
        r"\d{1,2}-[A-Za-z]{3}-\d{4}",
        " ".join(words_for_label("DUE_DATE"))
    )

    due_date = (
        due_match.group(0)
        if due_match
        else None
    )

    # ==========================================
    # INVOICE NUMBER
    # ==========================================
    invoice_match = re.search(
        r"\b[A-Z]{2,}\d+\b",
        " ".join(words_for_label("INVOICE_NUMBER"))
    )

    invoice_number = (
        invoice_match.group(0)
        if invoice_match
        else None
    )

    # ==========================================
    # TOTAL
    # ==========================================
    total_match = re.search(
        r"TOTAL\s*:\s*([\d,]+(?:\.\d+)?)",
        " ".join(words_for_label("TOTAL")),
        re.IGNORECASE
    )

    total = (
        total_match.group(1)
        if total_match
        else None
    )

    # ==========================================
    # TOTAL IN WORDS
    # ==========================================
    total_words = " ".join(
        words_for_label("TOTAL_WORDS")
    )

    total_words = re.sub(
        r"^Total\s+in\s+words\s*:\s*",
        "",
        total_words,
        flags=re.IGNORECASE
    )

    # ==========================================
    # BILL_TO / BUYER
    # ==========================================
    bill_to_words = []

    inside_bill_to = False

    for item in ocr_words:

        word = item["word"].strip()

        if word.upper() == "BILL":
            inside_bill_to = True

        if inside_bill_to:
            bill_to_words.append(word)

        if (
            inside_bill_to
            and word.lower() == "due"
        ):
            break

    bill_to_text = " ".join(bill_to_words)

    # Remove BILL_TO heading
    bill_to_text = re.sub(
        r"^BILL\s*_\s*TO\s*:\s*",
        "",
        bill_to_text,
        flags=re.IGNORECASE
    )

    # Remove trailing Due
    bill_to_text = re.sub(
        r"\s+Due\s*$",
        "",
        bill_to_text,
        flags=re.IGNORECASE
    )

    # ==========================================
    # Normalize OCR spacing
    # ==========================================
    normalized = bill_to_text

    normalized = re.sub(
        r"\s*@\s*",
        "@",
        normalized
    )

    normalized = re.sub(
        r"\s*\.\s*",
        ".",
        normalized
    )

    normalized = re.sub(
        r"\s*:\s*//\s*",
        "://",
        normalized
    )

    # ==========================================
    # BUYER NAME
    # ==========================================
    name_words = bill_to_text.split()

    buyer_name = (
        " ".join(name_words[:2])
        if len(name_words) >= 2
        else None
    )

    # ==========================================
    # EMAIL
    # ==========================================
    email_match = re.search(
        r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
        normalized
    )

    email = (
        email_match.group(0)
        if email_match
        else None
    )

    # ==========================================
    # PHONE
    # ==========================================
    phone_match = re.search(
        r"\+\s*\(\s*\d+\s*\)\s*[\d-]+",
        bill_to_text
    )

    phone = (
        phone_match.group(0)
        if phone_match
        else None
    )

    if phone:
        phone = re.sub(
            r"\s+",
            "",
            phone
        )

    # ==========================================
    # WEBSITE
    # ==========================================
    website_match = re.search(
        r"https?://[\w.-]+",
        normalized,
        re.IGNORECASE
    )

    website = (
        website_match.group(0)
        if website_match
        else None
    )

    # ==========================================
    # FINAL RESULT
    # ==========================================
    return {
        "seller": seller,
        "buyer": {
            "name": buyer_name,
            "email": email,
            "phone": phone,
            "website": website
        },
        "invoice_number": invoice_number,
        "date": date,
        "due_date": due_date,
        "total": total,
        "total_words": total_words
    }