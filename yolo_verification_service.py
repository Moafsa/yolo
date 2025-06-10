from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import cv2
from ultralytics import YOLO
from functools import wraps
from flask_cors import CORS
import requests  # NOVO

app = Flask(__name__)
CORS(app, origins=["https://vacancy.services"])
UPLOAD_FOLDER = './uploads'
LOG_FILE = './verification_audit.log'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Baixe o modelo YOLOv8n.pt com: wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
YOLO_MODEL_PATH = './yolov8n.pt'
yolo = YOLO(YOLO_MODEL_PATH)

AUTH_TOKEN = os.environ.get('AUTH_TOKEN', 'default_token')

# Função para logar auditoria
def log_audit(user_file, file_type, result, reason):
    with open(LOG_FILE, 'a') as f:
        f.write(f"{datetime.now().isoformat()} | {user_file} | {file_type} | {result} | {reason}\n")

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token or token != f"Bearer {AUTH_TOKEN}":
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/verify', methods=['POST'])
@require_auth
def verify():
    file = request.files.get('file')
    file_url = request.form.get('fileUrl')  # NOVO: aceita URL
    file_type = request.form.get('fileType')
    filename = None
    file_path = None

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
    elif file_url:
        filename = f"{datetime.now().timestamp()}-downloaded.jpg"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            r = requests.get(file_url, timeout=10)
            r.raise_for_status()
            with open(file_path, 'wb') as f:
                f.write(r.content)
        except Exception as e:
            return jsonify({'result': 'REJECTED', 'reason': f'Failed to download image: {str(e)}'}), 400
    else:
        return jsonify({'result': 'REJECTED', 'reason': 'Missing file or fileUrl'}), 400

    result = {'result': 'APPROVED'}
    reason = ''
    try:
        if file_type == 'selfie':
            # YOLO para detecção de rosto
            img = cv2.imread(file_path)
            yolo_results = yolo(img)
            faces = [d for d in yolo_results[0].boxes.cls.tolist() if int(d) == 0]  # classe 0 geralmente é 'person/face'
            if len(faces) == 0:
                result = {'result': 'REJECTED', 'reason': 'Face not detected'}
                reason = 'Face not detected'
            else:
                result = {'result': 'APPROVED'}
                reason = 'Face detected'
        elif file_type == 'document':
            # YOLO para detecção de documento (ou fallback para retângulo)
            img = cv2.imread(file_path)
            yolo_results = yolo(img)
            # Se modelo não for treinado para documento, usar contorno retangular
            found_rect = False
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 75, 200)
            contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4 and cv2.contourArea(approx) > 10000:
                    found_rect = True
                    break
            if found_rect:
                result = {'result': 'APPROVED'}
                reason = 'Document detected (rectangle)'
            else:
                result = {'result': 'REJECTED', 'reason': 'Document not detected'}
                reason = 'Document not detected'
        else:
            result = {'result': 'APPROVED'}
            reason = 'Default approval'
    except Exception as e:
        result = {'result': 'REJECTED', 'reason': f'YOLO error: {str(e)}'}
        reason = f'YOLO error: {str(e)}'
    finally:
        log_audit(filename, file_type, result['result'], reason)
        try:
            os.remove(file_path)
        except Exception:
            pass
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False) 