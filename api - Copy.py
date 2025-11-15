# # from fastapi import FastAPI
# # from fastapi.responses import FileResponse, HTMLResponse
# # import os

# # app = FastAPI(title="Clinical Note Generation API")

# # @app.get("/", response_class=HTMLResponse)
# # def home():
# #     return "<html><body><h2>✅ Results ready — check final_clinical_output.csv</h2></body></html>"

# # @app.get("/results")
# # def download_results():
# #     file_path = "final_clinical_output.csv"
# #     if os.path.exists(file_path):
# #         return FileResponse(file_path, media_type="text/csv", filename="final_clinical_output.csv")
# #     return {"detail": "results not found"}



# # api.py
# from fastapi import FastAPI, File, UploadFile, Form
# from fastapi.responses import JSONResponse, FileResponse
# from pydantic import BaseModel
# from typing import Optional
# import uvicorn
# import os
# import json
# from PIL import Image
# import shutil
# import io

# # Helper imports from your main pipeline (or reimplement minimal logic here)
# import pandas as pd

# app = FastAPI(title="Breast Ultrasound EHR AI Integration")

# # Ensure folders exist
# os.makedirs("ehr_integration", exist_ok=True)
# os.makedirs("enhanced_results", exist_ok=True)
# os.makedirs("enhanced_outputs", exist_ok=True)

# # Minimal ICD map (same as in main.py)
# icd10_map = {
#     'C50.9': 'Malignant neoplasm of breast, unspecified',
#     'D05.1': 'Lobular carcinoma in situ (benign)',
#     'C50.3': 'Malignant neoplasm of upper-inner quadrant of breast',
#     'N63': 'Unspecified lump in breast (normal)'
# }

# def predict_icd10_from_text(text: str) -> str:
#     t = text.lower()
#     if "lump" in t or "mass" in t: return "N63"
#     if "carcinoma" in t or "malignant" in t: return "C50.9"
#     if "benign" in t or "in situ" in t: return "D05.1"
#     return "C50.3"

# def generate_clinical_note_basic(age:int, gender:str, doctor_notes:str, icd_code:str):
#     return (f"{age}-year-old {gender} patient shows clinical signs of "
#             f"{icd10_map.get(icd_code, 'breast abnormality')}. Observation summary: {doctor_notes}")

# class PredictResponse(BaseModel):
#     patient_id: str
#     generated_note: str
#     predicted_icd_code: str
#     predicted_icd_description: Optional[str]
#     saved_to_ehr: bool

# @app.post("/predict", response_model=PredictResponse)
# async def predict(
#     patient_id: str = Form(...),
#     age: int = Form(...),
#     gender: str = Form(...),
#     doctor_notes: str = Form(...),
#     image: UploadFile = File(None)
# ):
#     # Save uploaded image if present
#     saved_image_path = None
#     if image is not None:
#         filename = f"{patient_id}_{image.filename}"
#         saved_image_path = os.path.join("enhanced_outputs", filename)
#         with open(saved_image_path, "wb") as f:
#             shutil.copyfileobj(image.file, f)

#         # Also create a grayscale resized preview
#         try:
#             im = Image.open(saved_image_path).convert("L").resize((224,224))
#             preview_path = os.path.join("enhanced_outputs", f"preview_{filename}")
#             im.save(preview_path)
#         except Exception:
#             pass

#     # Predict ICD using simple heuristic (same as main.py)
#     # If doctor_notes mentions ICD codes directly, prefer that
#     predicted_code = predict_icd10_from_text(doctor_notes)
#     predicted_description = icd10_map.get(predicted_code, "Unknown")

#     generated_note = generate_clinical_note_basic(age, gender, doctor_notes, predicted_code)

#     # Simulate EHR ingest: create JSON record
#     ehr_record = {
#         "patient_id": patient_id,
#         "age": age,
#         "gender": gender,
#         "doctor_notes": doctor_notes,
#         "image_path": saved_image_path,
#         "generated_note": generated_note,
#         "predicted_icd_code": predicted_code,
#         "predicted_icd_description": predicted_description
#     }

#     ehr_file = os.path.join("ehr_integration", f"{patient_id}.json")
#     with open(ehr_file, "w", encoding="utf-8") as ef:
#         json.dump(ehr_record, ef, indent=2)

#     # Also save to enhanced_results for audit
#     result_file = os.path.join("enhanced_results", f"{patient_id}_result.json")
#     with open(result_file, "w", encoding="utf-8") as rf:
#         json.dump(ehr_record, rf, indent=2)

#     return PredictResponse(
#         patient_id=patient_id,
#         generated_note=generated_note,
#         predicted_icd_code=predicted_code,
#         predicted_icd_description=predicted_description,
#         saved_to_ehr=True
#     )

# @app.post("/enhance-image")
# async def enhance_image(image: UploadFile = File(...)):
#     """Return a processed preview image (grayscale 224x224)"""
#     contents = await image.read()
#     try:
#         im = Image.open(io.BytesIO(contents)).convert("L").resize((224,224))
#         preview_path = os.path.join("enhanced_outputs", f"preview_{image.filename}")
#         im.save(preview_path)
#         return FileResponse(preview_path, media_type="image/png")
#     except Exception as e:
#         return JSONResponse({"error": str(e)}, status_code=400)

# @app.get("/ehr/{patient_id}")
# def get_ehr(patient_id: str):
#     fname = os.path.join("ehr_integration", f"{patient_id}.json")
#     if not os.path.exists(fname):
#         return JSONResponse({"error": "patient not found"}, status_code=404)
#     with open(fname, "r", encoding="utf-8") as f:
#         return JSONResponse(json.load(f))



# # api.py
# from fastapi import FastAPI, File, UploadFile, Form
# from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from pydantic import BaseModel
# from typing import Optional
# import uvicorn
# import os
# import json
# from PIL import Image
# import shutil
# import io
# import pandas as pd

# app = FastAPI(title="Breast Ultrasound EHR AI Integration")

# # Serve static files (HTML)
# app.mount("/static", StaticFiles(directory="static"), name="static")


# # ---------------- HOME ROUTE (Humanised) ----------------
# @app.get("/", response_class=HTMLResponse)
# def home():
#     return """
#     <html>
#         <body style='font-family: Arial; padding: 40px;'>
#             <h2>💗 Breast Ultrasound EHR AI — API is Running</h2>
#             <p>Your backend is live and ready. Use the <b>/predict</b> endpoint or open:</p>
#             <a href='/static/index.html' style='font-size:18px;'>👉 Open Frontend Form</a>
#         </body>
#     </html>
#     """


# # ------------------- FOLDER CREATION -------------------
# os.makedirs("ehr_integration", exist_ok=True)
# os.makedirs("enhanced_results", exist_ok=True)
# os.makedirs("enhanced_outputs", exist_ok=True)


# # ------------------- ICD MAP -------------------
# icd10_map = {
#     'C50.9': 'Malignant neoplasm of breast, unspecified',
#     'D05.1': 'Lobular carcinoma in situ (benign)',
#     'C50.3': 'Malignant neoplasm of upper-inner quadrant of breast',
#     'N63': 'Unspecified lump in breast (normal)'
# }


# # ------------------- SIMPLE ICD PREDICTOR -------------------
# def predict_icd10_from_text(text: str) -> str:
#     t = text.lower()
#     if "lump" in t or "mass" in t:
#         return "N63"
#     if "carcinoma" in t or "malignant" in t:
#         return "C50.9"
#     if "benign" in t or "in situ" in t:
#         return "D05.1"
#     return "C50.3"


# # ------------------- CLINICAL NOTE GENERATOR -------------------
# def generate_clinical_note_basic(age: int, gender: str, doctor_notes: str, icd_code: str):
#     return (
#         f"{age}-year-old {gender} patient shows clinical signs of "
#         f"{icd10_map.get(icd_code, 'breast abnormality')}. "
#         f"Observation summary: {doctor_notes}"
#     )


# # ------------------- RESPONSE MODEL -------------------
# class PredictResponse(BaseModel):
#     patient_id: str
#     generated_note: str
#     predicted_icd_code: str
#     predicted_icd_description: Optional[str]
#     saved_to_ehr: bool


# # ------------------- /PREDICT ROUTE -------------------
# @app.post("/predict", response_model=PredictResponse)
# async def predict(
#     patient_id: str = Form(...),
#     age: int = Form(...),
#     gender: str = Form(...),
#     doctor_notes: str = Form(...),
#     image: UploadFile = File(None)
# ):
#     # Save uploaded image if present
#     saved_image_path = None
#     if image is not None:
#         filename = f"{patient_id}_{image.filename}"
#         saved_image_path = os.path.join("enhanced_outputs", filename)

#         with open(saved_image_path, "wb") as f:
#             shutil.copyfileobj(image.file, f)

#         # Create grayscale preview
#         try:
#             im = Image.open(saved_image_path).convert("L").resize((224, 224))
#             preview_path = os.path.join("enhanced_outputs", f"preview_{filename}")
#             im.save(preview_path)
#         except Exception:
#             pass

#     # ICD prediction
#     predicted_code = predict_icd10_from_text(doctor_notes)
#     predicted_description = icd10_map.get(predicted_code, "Unknown")

#     # Generate note
#     generated_note = generate_clinical_note_basic(age, gender, doctor_notes, predicted_code)

#     # Prepare EHR record
#     ehr_record = {
#         "patient_id": patient_id,
#         "age": age,
#         "gender": gender,
#         "doctor_notes": doctor_notes,
#         "image_path": saved_image_path,
#         "generated_note": generated_note,
#         "predicted_icd_code": predicted_code,
#         "predicted_icd_description": predicted_description
#     }

#     # Save to ehr_integration
#     ehr_file = os.path.join("ehr_integration", f"{patient_id}.json")
#     with open(ehr_file, "w", encoding="utf-8") as ef:
#         json.dump(ehr_record, ef, indent=2)

#     # Save result copy
#     result_file = os.path.join("enhanced_results", f"{patient_id}_result.json")
#     with open(result_file, "w", encoding="utf-8") as rf:
#         json.dump(ehr_record, rf, indent=2)

#     return PredictResponse(
#         patient_id=patient_id,
#         generated_note=generated_note,
#         predicted_icd_code=predicted_code,
#         predicted_icd_description=predicted_description,
#         saved_to_ehr=True
#     )


# # ---------------- IMAGE ENHANCEMENT -------------------
# @app.post("/enhance-image")
# async def enhance_image(image: UploadFile = File(...)):
#     contents = await image.read()
#     try:
#         im = Image.open(io.BytesIO(contents)).convert("L").resize((224, 224))
#         preview_path = os.path.join("enhanced_outputs", f"preview_{image.filename}")
#         im.save(preview_path)
#         return FileResponse(preview_path, media_type="image/png")
#     except Exception as e:
#         return JSONResponse({"error": str(e)}, status_code=400)


# # ---------------- FETCH EHR -------------------
# @app.get("/ehr/{patient_id}")
# def get_ehr(patient_id: str):
#     fname = os.path.join("ehr_integration", f"{patient_id}.json")
#     if not os.path.exists(fname):
#         return JSONResponse({"error": "patient not found"}, status_code=404)

#     with open(fname, "r", encoding="utf-8") as f:
#         return JSONResponse(json.load(f))


# # ---------------- RUN SERVER -------------------
# if __name__ == "__main__":
#     uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)









from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import json
from PIL import Image
import shutil

app = FastAPI(title="Breast Ultrasound EHR AI Integration")

# Serve static HTML files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Ensure required folders exist
os.makedirs("ehr_integration", exist_ok=True)
os.makedirs("enhanced_results", exist_ok=True)
os.makedirs("enhanced_outputs", exist_ok=True)

# -----------------------------
# ICD MAPPING RULES
# -----------------------------
ICD_RULES = {
    "breast cancer": "C50.3",
    "tumor": "C50.3",
    "lump": "C50.3",

    # Endocrine
    "diabetes": "E11",
    "thyroid": "E03.9",
    "hypothyroidism": "E03.9",
    "hyperthyroidism": "E05.9",

    # Respiratory
    "asthma": "J45",
    "pneumonia": "J18.9",
    "bronchitis": "J20.9",
    "copd": "J44.9",
    "tuberculosis": "A15.0",

    # Cardiovascular
    "hypertension": "I10",
    "high blood pressure": "I10",
    "heart attack": "I21",
    "myocardial infarction": "I21",
    "heart failure": "I50.9",
    "arrhythmia": "I49.9",

    # Neurology
    "stroke": "I63.9",
    "epilepsy": "G40.9",
    "headache": "R51",

    # Gastrointestinal
    "gastritis": "K29.7",
    "ulcer": "K25.9",
    "liver disease": "K76.9",
    "hepatitis": "B19.9",

    # Renal
    "kidney failure": "N17.9",
    "uti": "N39.0",
    "urinary infection": "N39.0",

    # Musculoskeletal
    "arthritis": "M19.90",
    "back pain": "M54.5",
    "fracture": "S52.90",

    # Infection / Fever
    "fever": "R50.9",
    "viral infection": "B34.9",
    "bacterial infection": "A49.9",

    # OBGYN
    "pregnancy": "Z34.90",
    "pcos": "E28.2",
    "fibroid": "D25.9",

    # Dermatology
    "skin infection": "L08.9",
    "dermatitis": "L30.9",

    # Mental health
    "depression": "F32.9",
    "anxiety": "F41.9",
}

DEFAULT_ICD = "C50.3"


# -----------------------------------------
# ICD Prediction
# -----------------------------------------
def predict_icd10_from_notes(text: str):
    text = text.lower()
    for keyword, code in ICD_RULES.items():
        if keyword in text:
            return code
    return DEFAULT_ICD


# -----------------------------------------
# Note Generator
# -----------------------------------------
def make_note(age, gender, doctor_notes, icd_code):
    return (
        f"{age}-year-old {gender} patient shows clinical signs related to ICD-10 "
        f"code {icd_code}. Summary of symptoms: {doctor_notes}"
    )


# -----------------------------------------
# Response Model
# -----------------------------------------
class PredictResponse(BaseModel):
    patient_id: str
    generated_note: str
    predicted_icd_code: str
    saved_to_ehr: bool


# -----------------------------------------
# MAIN PREDICTION API
# -----------------------------------------
@app.post("/predict", response_model=PredictResponse)
async def predict(
    patient_id: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    doctor_notes: str = Form(...),
    image: UploadFile = File(None)
):

    # -----------------------------
    # IMAGE VALIDATION (OPTIONAL)
    # -----------------------------
    img_path = None

    if image and image.filename != "":
        allowed = (".png", ".jpg", ".jpeg", ".dcm")

        if not image.filename.lower().endswith(allowed):
            return {"error": "Only medical image formats allowed: PNG, JPG, JPEG, DCM"}

        # Save original image
        img_path = os.path.join("enhanced_outputs", f"{patient_id}_{image.filename}")
        with open(img_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        # Try to make a preview image
        try:
            img = Image.open(img_path).convert("L").resize((224, 224))
            preview_path = os.path.join("enhanced_outputs", f"preview_{patient_id}.png")
            img.save(preview_path)
        except Exception:
            pass

    # -----------------------------
    # ICD PREDICTION
    # -----------------------------
    predicted_code = predict_icd10_from_notes(doctor_notes)

    # Generate clinical note
    generated_note = make_note(age, gender, doctor_notes, predicted_code)

    # -----------------------------
    # Save EHR Record
    # -----------------------------
    ehr_record = {
        "patient_id": patient_id,
        "age": age,
        "gender": gender,
        "doctor_notes": doctor_notes,
        "image_path": img_path,
        "generated_note": generated_note,
        "predicted_icd_code": predicted_code
    }

    with open(f"ehr_integration/{patient_id}.json", "w") as f:
        json.dump(ehr_record, f, indent=2)

    return PredictResponse(
        patient_id=patient_id,
        generated_note=generated_note,
        predicted_icd_code=predicted_code,
        saved_to_ehr=True
    )


# -----------------------------------------
# EHR FETCH
# -----------------------------------------
@app.get("/ehr/{patient_id}")
def get_ehr(patient_id: str):
    file_path = f"ehr_integration/{patient_id}.json"
    if not os.path.exists(file_path):
        return JSONResponse({"error": "Patient not found"}, status_code=404)
    return json.load(open(file_path))
