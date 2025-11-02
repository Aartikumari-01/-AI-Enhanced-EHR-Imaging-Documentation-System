# import pandas as pd
# import matplotlib.pyplot as plt
# from PIL import Image
# import zipfile
# import io
# import os
# import json


# structured_df = pd.read_csv('structured_data.csv')
# unstructured_df = pd.read_csv('unstructured_notes.csv')


# clinical_df = pd.merge(structured_df, unstructured_df, on='Patient_ID')
# print("Merged Clinical Data:")
# print(clinical_df.head())

# clinical_df.dropna(subset=['ICD10_Code', 'Doctor_Notes', 'Age'], inplace=True)

# clinical_df['Age'] = pd.to_numeric(clinical_df['Age'], errors='coerce')
# clinical_df.dropna(subset=['Age'], inplace=True)


# clinical_df['Gender'] = clinical_df['Gender'].str.upper().map({'M': 'Male', 'F': 'Female'})

# print("Cleaned Clinical Data:")
# print(clinical_df.head())

# zip_file_path = 'dataset.zip'


# image_entries = []

# with zipfile.ZipFile(zip_file_path, 'r') as archive:
#     all_files = archive.namelist()
#     print("Found files in ZIP:", all_files[:5])

    
#     image_entries = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    
#     for img_file in image_entries[:3]:
#         with archive.open(img_file) as file:
#             img = Image.open(io.BytesIO(file.read()))
#             img = img.convert('L').resize((224, 224))  
#             plt.imshow(img, cmap='gray')
#             plt.title(f"Preview: {img_file}")
#             plt.axis('off')
#             plt.show()



# icd10_map = {
#     'C50.9': 'Malignant neoplasm of breast, unspecified',
#     'D05.1': 'Lobular carcinoma in situ',
#     'C50.3': 'Malignant neoplasm of upper-inner quadrant of breast',
#     'N63': 'Unspecified lump in breast',
    
# }


# genai_jsonl_data = []
# vlm_csv_data = []

# with zipfile.ZipFile(zip_file_path, 'r') as archive:
#     for _, row in clinical_df.iterrows():
#         patient_id = row['Patient_ID']
#         age = int(row['Age'])
#         gender = row['Gender']
#         icd_code = row['ICD10_Code']
#         diagnosis = icd10_map.get(icd_code, icd_code)
#         notes = row['Doctor_Notes']

        
#         matched_img = next((img for img in image_entries if img.startswith(patient_id)), None)

#         if matched_img:
            
#             prompt = f"Patient ({gender}, {age} years old):\nClinical Notes: {notes}\nWhat is the likely diagnosis?"
#             response = diagnosis
#             genai_jsonl_data.append({
#                 "patient_id": patient_id,
#                 "image_file": matched_img,
#                 "prompt": prompt,
#                 "response": response
#             })

            
#             vlm_csv_data.append({
#                 "patient_id": patient_id,
#                 "image_file": matched_img,
#                 "text_prompt": notes,
#                 "label": diagnosis
#             })
#         else:
#             print(f"No image found for {patient_id}")


# with open("genai_dataset.jsonl", "w", encoding='utf-8') as f:
#     for entry in genai_jsonl_data:
#         f.write(json.dumps(entry) + '\n')
# print("Saved GenAI prompt-response dataset to genai_dataset.jsonl")


# pd.DataFrame(vlm_csv_data).to_csv("vlm_dataset.csv", index=False)
# print("Saved Vision-Language dataset to vlm_dataset.csv")

# plt.figure(figsize=(8, 5))
# clinical_df['ICD10_Code'].map(icd10_map).value_counts().plot(kind='bar', color='skyblue')
# plt.title("Diagnosis Distribution")
# plt.xlabel("Diagnosis")
# plt.ylabel("Number of Patients")
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(6, 4))
# plt.hist(clinical_df['Age'], bins=10, color='lightgreen', edgecolor='black')
# plt.title("Age Distribution of Patients")
# plt.xlabel("Age")
# plt.ylabel("Frequency")
# plt.grid(True)
# plt.show()

# print("Data preparation complete.")


# import pandas as pd
# import matplotlib.pyplot as plt
# from PIL import Image
# import glob
# import json
# import os

# # ====================================================
# # 1) LOAD STRUCTURED & TEXT DATA
# # ====================================================

# print("\nLoading data...")

# structured_df = pd.read_csv('structured_data.csv')
# unstructured_df = pd.read_csv('unstructured_notes.csv')

# # Merge EHR + Doctor notes
# clinical_df = pd.merge(structured_df, unstructured_df, on='Patient_ID')
# print("\nMerged sample data:")
# print(clinical_df.head())

# # ====================================================
# # 2) DATA CLEANING
# # ====================================================

# print("\n🧹 Cleaning data...")

# clinical_df.dropna(subset=['ICD10_Code', 'Doctor_Notes', 'Age'], inplace=True)
# clinical_df['Age'] = pd.to_numeric(clinical_df['Age'], errors='coerce')
# clinical_df.dropna(subset=['Age'], inplace=True)
# clinical_df['Gender'] = clinical_df['Gender'].str.upper().map({'M': 'Male', 'F': 'Female'})

# print("\nAfter cleaning:")
# print(clinical_df.head())

# # ====================================================
# # 3) LOAD MEDICAL IMAGES FROM FOLDER
# # ====================================================

# print("\nLoading medical images...")

# image_folder = "Dataset_BUSI_with_GT"

# # Recursively search for all image files
# image_entries = glob.glob(f"{image_folder}/**/*.png", recursive=True)
# image_entries += glob.glob(f"{image_folder}/**/*.jpg", recursive=True)
# image_entries += glob.glob(f"{image_folder}/**/*.jpeg", recursive=True)

# print(f"Total image files found (including subfolders): {len(image_entries)}")

# if len(image_entries) == 0:
#     print("No image files found. Please check folder path or file extensions.")
# else:
#     print(f"Found {len(image_entries)} image files.")
#     # Preview first 3 images
#     for img_path in image_entries[:3]:
#         img = Image.open(img_path).convert("L").resize((224, 224))
#         plt.imshow(img, cmap='gray')
#         plt.title(f"Preview: {os.path.basename(img_path)}")
#         plt.axis('off')
#         plt.show()

# # ====================================================
# # 4) ICD-10 CODE MAPPING
# # ====================================================

# icd10_map = {
#     'C50.9': 'Malignant neoplasm of breast, unspecified',
#     'D05.1': 'Lobular carcinoma in situ (benign)',
#     'C50.3': 'Malignant neoplasm of upper-inner quadrant of breast',
#     'N63': 'Unspecified lump in breast (normal)',
# }

# # Helper: match image based on ICD type
# def match_image_by_diagnosis(icd_code, image_list):
#     if "C50" in icd_code:  # malignant
#         for path in image_list:
#             if "malignant" in path.lower():
#                 return path
#     elif "D05" in icd_code:  # benign
#         for path in image_list:
#             if "benign" in path.lower():
#                 return path
#     elif "N63" in icd_code:  # normal
#         for path in image_list:
#             if "normal" in path.lower():
#                 return path
#     return None

# genai_jsonl_data = []
# vlm_csv_data = []

# for _, row in clinical_df.iterrows():
#     pid = row['Patient_ID']
#     age, gender = int(row['Age']), row['Gender']
#     code = row['ICD10_Code']
#     diagnosis = icd10_map.get(code, code)
#     notes = row['Doctor_Notes']

#     matched_img = match_image_by_diagnosis(code, image_entries)

#     if matched_img:
#         prompt = (
#             f"Patient ({gender}, {age} years) presents with breast symptoms.\n"
#             f"Notes: {notes}\n"
#             "Give likely diagnosis:"
#         )

#         genai_jsonl_data.append({
#             "patient_id": pid,
#             "image_file": matched_img,
#             "prompt": prompt,
#             "response": diagnosis
#         })

#         vlm_csv_data.append({
#             "patient_id": pid,
#             "image_file": matched_img,
#             "text_prompt": notes,
#             "label": diagnosis
#         })
#     else:
#         print(f"No image found for {pid} ({code})")

# # Save datasets
# with open("genai_dataset.jsonl", "w", encoding='utf-8') as f:
#     for entry in genai_jsonl_data:
#         f.write(json.dumps(entry) + '\n')

# pd.DataFrame(vlm_csv_data).to_csv("vlm_dataset.csv", index=False)

# print("\nGenerated genAI + VLM training datasets")

# # ====================================================
# # 5) VISUAL ANALYSIS PLOTS
# # ====================================================

# plt.figure(figsize=(8, 5))
# clinical_df['ICD10_Code'].map(icd10_map).value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
# plt.title("Diagnosis Frequency")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(6, 4))
# plt.hist(clinical_df['Age'], bins=10, edgecolor='black', color='lightcoral')
# plt.title("Age Distribution of Patients")
# plt.xlabel("Age")
# plt.ylabel("Count")
# plt.grid(True)
# plt.show()

# print("\nData preparation complete.")

# # ====================================================
# # 6) MODULE-3: CLINICAL NOTE GENERATION + ICD PREDICT
# # ====================================================

# print("\n===Module-3: Clinical Note Generation & ICD-10 Automation ===")

# # Rule-based ICD prediction (simple NLP demo)
# def predict_icd10(text):
#     t = text.lower()
#     if "lump" in t or "mass" in t: return "N63"
#     if "carcinoma" in t or "malignant" in t: return "C50.9"
#     if "in situ" in t or "benign" in t: return "D05.1"
#     return "C50.3"  # fallback

# # Clinical note generator
# def generate_note(row):
#     return (f"{row['Age']}-year-old {row['Gender']} with clinical signs of "
#             f"{icd10_map.get(row['ICD10_Code'], 'breast abnormality')}. "
#             f"Clinical evaluation supports presence of breast pathology.")

# clinical_df["Generated_Notes"] = clinical_df.apply(generate_note, axis=1)
# clinical_df["Predicted_ICD_Code"] = clinical_df["Generated_Notes"].apply(predict_icd10)
# clinical_df["Predicted_ICD_Description"] = clinical_df["Predicted_ICD_Code"].map(icd10_map)

# print("\nOutput:\n")
# print(clinical_df[[
#     "Patient_ID", "Generated_Notes",
#     "Predicted_ICD_Code", "Predicted_ICD_Description"
# ]].head())

# clinical_df.to_csv("final_clinical_output.csv", index=False)

# print("\nFinal AI-Generated output saved as: final_clinical_output.csv")
# print("Module-3 completed successfully!")




import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import glob
import json
import os

# ====================================================
# 1) LOAD STRUCTURED & TEXT DATA
# ====================================================

print("\n=== Clinical Note Generation & ICD-10 Automation ===\n")

structured_df = pd.read_csv('structured_data.csv')
unstructured_df = pd.read_csv('unstructured_notes.csv')

# Merge EHR + Doctor notes
clinical_df = pd.merge(structured_df, unstructured_df, on='Patient_ID')
print("\n📘 Merged Clinical Data (Sample):")
print(clinical_df.head(), "\n")

# ====================================================
# 2) DATA CLEANING
# ====================================================

print("Cleaning data...")

clinical_df.dropna(subset=['ICD10_Code', 'Doctor_Notes', 'Age'], inplace=True)
clinical_df['Age'] = pd.to_numeric(clinical_df['Age'], errors='coerce')
clinical_df.dropna(subset=['Age'], inplace=True)
clinical_df['Gender'] = clinical_df['Gender'].str.upper().map({'M': 'Male', 'F': 'Female'})

print("\nCleaned Data (Sample):")
print(clinical_df.head(), "\n")

# ====================================================
# 3) LOAD MEDICAL IMAGES FROM FOLDER
# ====================================================

print("Loading medical images from Dataset_BUSI_with_GT...")

image_folder = "Dataset_BUSI_with_GT"
image_entries = glob.glob(f"{image_folder}/**/*.png", recursive=True)
image_entries += glob.glob(f"{image_folder}/**/*.jpg", recursive=True)
image_entries += glob.glob(f"{image_folder}/**/*.jpeg", recursive=True)

print(f"Found total {len(image_entries)} image files.\n")

if len(image_entries) > 0:
    for img_path in image_entries[:3]:
        img = Image.open(img_path).convert("L").resize((224, 224))
        plt.imshow(img, cmap='gray')
        plt.title(f"Preview: {os.path.basename(img_path)}")
        plt.axis('off')
        plt.show()
else:
    print("No image files found! Please check dataset path or file format.\n")

# ====================================================
# 4) ICD-10 CODE MAPPING
# ====================================================

icd10_map = {
    'C50.9': 'Malignant neoplasm of breast, unspecified',
    'D05.1': 'Lobular carcinoma in situ (benign)',
    'C50.3': 'Malignant neoplasm of upper-inner quadrant of breast',
    'N63': 'Unspecified lump in breast (normal)'
}

def match_image_by_diagnosis(icd_code, image_list):
    """Match image based on diagnosis keyword."""
    icd_code = icd_code.upper()
    if "C50" in icd_code:  # malignant
        for path in image_list:
            if "malignant" in path.lower():
                return path
    elif "D05" in icd_code:  # benign
        for path in image_list:
            if "benign" in path.lower():
                return path
    elif "N63" in icd_code:  # normal
        for path in image_list:
            if "normal" in path.lower():
                return path
    return None

# ====================================================
# 5) CREATE GENAI + VLM TRAINING DATASETS
# ====================================================

genai_jsonl_data = []
vlm_csv_data = []

for _, row in clinical_df.iterrows():
    pid = row['Patient_ID']
    age, gender = int(row['Age']), row['Gender']
    code = row['ICD10_Code']
    diagnosis = icd10_map.get(code, code)
    notes = row['Doctor_Notes']

    matched_img = match_image_by_diagnosis(code, image_entries)

    if matched_img:
        prompt = (
            f"Patient ({gender}, {age} years old):\n"
            f"Doctor Observation: {notes}\n"
            f"What is the likely diagnosis?"
        )

        genai_jsonl_data.append({
            "patient_id": pid,
            "image_file": matched_img,
            "prompt": prompt,
            "response": diagnosis
        })

        vlm_csv_data.append({
            "patient_id": pid,
            "image_file": matched_img,
            "text_prompt": notes,
            "label": diagnosis
        })
    else:
        print(f"No image found for {pid} ({code})")

with open("genai_dataset.jsonl", "w", encoding='utf-8') as f:
    for entry in genai_jsonl_data:
        f.write(json.dumps(entry) + '\n')

pd.DataFrame(vlm_csv_data).to_csv("vlm_dataset.csv", index=False)

print("\nGenerated GenAI + VLM training datasets successfully.\n")

# ====================================================
# 6) VISUAL ANALYSIS
# ====================================================

plt.figure(figsize=(8, 5))
clinical_df['ICD10_Code'].map(icd10_map).value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Diagnosis Frequency")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.hist(clinical_df['Age'], bins=10, color='lightgreen', edgecolor='black')
plt.title("Age Distribution of Patients")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# ====================================================
# 7) CLINICAL NOTE GENERATION + ICD PREDICTION
# ====================================================

print("\nGenerating clinical notes & predicting ICD-10 codes...\n")

def predict_icd10(text):
    t = text.lower()
    if "lump" in t or "mass" in t: return "N63"
    if "carcinoma" in t or "malignant" in t: return "C50.9"
    if "benign" in t or "in situ" in t: return "D05.1"
    return "C50.3"

def generate_clinical_note(row):
    return (f"{row['Age']}-year-old {row['Gender']} patient shows clinical signs of "
            f"{icd10_map.get(row['ICD10_Code'], 'breast abnormality')}. "
            f"Observation summary: {row['Doctor_Notes']}")

clinical_df["Generated_Notes"] = clinical_df.apply(generate_clinical_note, axis=1)
clinical_df["Predicted_ICD_Code"] = clinical_df["Generated_Notes"].apply(predict_icd10)
clinical_df["Predicted_ICD_Description"] = clinical_df["Predicted_ICD_Code"].map(icd10_map)

# Perfect alignment for output display
print("🩺 Sample Output:\n")
print("{:<10} {:<80} {:<20} {:<50}".format(
    "Patient_ID", "Generated_Notes", "Predicted_ICD_Code", "Predicted_ICD_Description"
))
print("-" * 170)
for _, row in clinical_df.iterrows():
    print("{:<10} {:<80} {:<20} {:<50}".format(
        row["Patient_ID"],
        row["Generated_Notes"][:75] + "...",
        row["Predicted_ICD_Code"],
        row["Predicted_ICD_Description"]
    ))

clinical_df.to_csv("final_clinical_output.csv", index=False)

print("\nFinal AI-Generated output saved as: final_clinical_output.csv")
print("Module completed successfully!")

# ====================================================
# 8) OPTIONAL: UNSTRUCTURED DATA ONLY
# ====================================================

ask_extra = input("\nDo you also want to generate from *unstructured text only*? (yes/no): ").strip().lower()

if ask_extra == "yes":
    print("\nGenerating unstructured clinical notes...\n")
    unstructured_df["Generated_Note"] = unstructured_df["Doctor_Notes"].apply(
        lambda x: f"Unstructured clinical summary: {x}"
    )
    unstructured_df.to_csv("unstructured_generated_notes.csv", index=False)
    print("Saved unstructured-only notes as unstructured_generated_notes.csv\n")
else:
    print("\nSkipped unstructured-only generation.\n")
