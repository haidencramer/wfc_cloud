import os
import uuid
import json
from flask import Flask, render_template, request, jsonify
from google.cloud import storage, pubsub_v1, firestore
from PIL import Image
import io

app = Flask(__name__)

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id")
INPUT_BUCKET = f"wfc-inputs-{PROJECT_ID}"
OUTPUT_BUCKET = f"wfc-outputs-{PROJECT_ID}"
DB_NAME = os.environ.get("FIRESTORE_DB_NAME", "wfc-db")

storage_client = storage.Client()
publisher = pubsub_v1.PublisherClient()
TOPIC_PATH = publisher.topic_path(PROJECT_ID, "wfc-work-queue")
firestore_client = firestore.Client(database=DB_NAME)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'seed_image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['seed_image']
    patch_size = int(request.form.get('patch_size', 3))
    
    if patch_size < 2 or patch_size > 5:
        return jsonify({"error": "Patch size must be between 2 and 5"}), 400

    image_bytes = file.read()
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.width > 64 or img.height > 64:
            return jsonify({"error": f"Image is {img.width}x{img.height}. Max allowed is 64x64."}), 400
    except Exception as e:
        return jsonify({"error": "Invalid image file"}), 400

    job_id = f"job-{uuid.uuid4().hex[:8]}"
    filename = f"{job_id}.png"

    bucket = storage_client.bucket(INPUT_BUCKET)
    blob = bucket.blob(filename)
    blob.upload_from_string(image_bytes, content_type=file.content_type)

    doc_ref = firestore_client.collection("wfc_jobs").document(job_id)
    doc_ref.set({
        "status": "PENDING",
        "patch_size": patch_size,
        "output_url": None
    })

    work_order = {
        "job_id": job_id,
        "input_bucket": INPUT_BUCKET,
        "input_filename": filename,
        "output_bucket": OUTPUT_BUCKET,
        "patch_size": patch_size
    }
    
    publisher.publish(TOPIC_PATH, json.dumps(work_order).encode("utf-8"))
    return jsonify({"job_id": job_id})

@app.route('/status/<job_id>', methods=['GET'])
def check_status(job_id):
    doc = firestore_client.collection("wfc_jobs").document(job_id).get()
    if not doc.exists:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(doc.to_dict())

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)