import easyocr

reader = easyocr.Reader(['en'])

def identify_player(frame, bbox, target_number=None, target_color=None):
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]

    # OCR numéro
    result = reader.readtext(crop)

    number_detected = None
    if result:
        number_detected = result[0][1]

    # couleur moyenne
    avg_color = crop.mean(axis=(0,1))

    return {
        "number": number_detected,
        "color": avg_color
    }