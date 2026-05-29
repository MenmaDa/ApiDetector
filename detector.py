from flask import Flask, request, jsonify
from flask_cors import CORS
from paddleocr import PaddleOCR
import cv2
import uuid
import os

app = Flask(__name__)

# 🔥 ESTO ES LO QUE TE FALTA
CORS(app)

UPLOAD_FOLDER = "temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# OCR estable
ocr = None

def get_ocr():
    global ocr
    if ocr is None:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang="en")
    return ocr

def preprocess_image(path):
    img = cv2.imread(path)

    if img is None:
        raise ValueError("No se pudo leer la imagen")

    # Mejora básica para OCR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.equalizeHist(gray)

    return gray


@app.route("/detect", methods=["POST"])
def detect():
    try:
        if "image" not in request.files:
            return jsonify({"ok": False, "error": "No image provided"}), 400

        file = request.files["image"]

        temp_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.png")
        file.save(temp_path)

        processed = preprocess_image(temp_path)

        processed_path = os.path.join(
            UPLOAD_FOLDER,
            f"proc_{uuid.uuid4().hex}.png"
        )

        cv2.imwrite(processed_path, processed)

        # OCR
        engine = get_ocr()
        result = engine.ocr(processed_path, cls=False)

        # 🔥 DEBUG IMPORTANTE
        print("========== OCR RAW RESULT ==========")
        print(result)
        print("====================================")

        text_parts = []

        if result and result[0]:
            for line in result[0]:
                print("LINE:", line)  # 🔥 ver cada detección

                try:
                    text_parts.append(str(line[1][0]))
                except Exception as e:
                    print("ERROR PARSING LINE:", e)

        text = " ".join(text_parts).strip()

        print("FINAL TEXT:", text)

        # 🔥 BORRAR IMÁGENES TEMPORALES
        if os.path.exists(temp_path):
            os.remove(temp_path)

        if os.path.exists(processed_path):
            os.remove(processed_path)

        return jsonify({
            "ok": True,
            "text": text,
            "debug_raw": str(result)
        })

    except Exception as e:
        print("🔥 FLASK ERROR:", str(e))
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)