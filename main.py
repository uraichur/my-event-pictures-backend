# Force fresh build
import os
import cv2
import json
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from insightface.app import FaceAnalysis
from supabase import create_client, Client
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- FastAPI Initialization ---
app = FastAPI()
app.mount("/event_photos", StaticFiles(directory="event_photos"), name="event_photos")

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- Face Analysis Model ---
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

# --- Directories ---
EVENT_PHOTO_DIR = "event_photos"
FACE_CROP_DIR = "face_crops"
SELFIE_DIR = "guest_selfies"
os.makedirs(EVENT_PHOTO_DIR, exist_ok=True)
os.makedirs(FACE_CROP_DIR, exist_ok=True)
os.makedirs(SELFIE_DIR, exist_ok=True)

# --- Upload Group Photo and Store Faces ---
@app.post("/upload-group-photo")
async def upload_group_photo(file: UploadFile = File(...)):
    # Save image
    file_path = os.path.join(EVENT_PHOTO_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    print(f"📸 Saved group image: {file.filename}")

    # Read and detect faces
    img = cv2.imread(file_path)
    if img is None:
        return {"message": "Failed to read image."}

    faces = face_app.get(img)
    print(f"🔍 Detected {len(faces)} faces in {file.filename}")

    face_id = len(os.listdir(FACE_CROP_DIR)) + 1
    for face in faces:
        x, y, w, h = face.bbox.astype(int)
        crop = img[y:h, x:w]
        face_filename = f"face_{face_id}.jpg"
        crop_path = os.path.join(FACE_CROP_DIR, face_filename)
        cv2.imwrite(crop_path, crop)

        vector = face.embedding.tolist()
        payload = {
            "filename": face_filename,
            "vector": json.dumps(vector),
            "source_image": file.filename
        }

        try:
            supabase.table("event_faces").insert(payload).execute()
            print(f"✅ Inserted {face_filename} from {file.filename}")
        except Exception as e:
            print(f"❌ Failed to insert {face_filename}: {e}")

        face_id += 1

    return {"message": f"✅ Uploaded and processed {file.filename}"}

# --- Upload Selfie and Match ---
@app.post("/upload-selfie")
async def upload_selfie(file: UploadFile = File(...)):
    selfie_path = os.path.join(SELFIE_DIR, "selfie.jpg")
    with open(selfie_path, "wb") as f:
        f.write(await file.read())

    print("📸 Saved selfie as selfie.jpg")

    img = cv2.imread(selfie_path)
    faces = face_app.get(img)
    if len(faces) == 0:
        return {"message": "❌ No face detected in selfie."}

    embedding = faces[0].embedding

    # Fetch faces from Supabase
    response = supabase.table("event_faces").select("*").execute()
    rows = response.data

    matched_images = set()
    for row in rows:
        try:
            db_vector = np.array(json.loads(row["vector"]), dtype=np.float32)
            cosine_sim = np.dot(embedding, db_vector) / (np.linalg.norm(embedding) * np.linalg.norm(db_vector))
            if cosine_sim >= 0.25:
                matched_images.add(row["source_image"])
        except Exception as e:
            print(f"⚠️ Error parsing vector: {e}")

    if not matched_images:
        return {"message": "❌ No matching faces found with score ≥ 0.25"}
    else:
        return {"matched_images": list(matched_images)}

@app.get("/")
def home():
    return {"status": "API Running"}

