import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import zipfile
import io
import os
import json


structured_df = pd.read_csv('structured_data.csv')
unstructured_df = pd.read_csv('unstructured_notes.csv')


clinical_df = pd.merge(structured_df, unstructured_df, on='Patient_ID')
print("Merged Clinical Data:")
print(clinical_df.head())

clinical_df.dropna(subset=['ICD10_Code', 'Doctor_Notes', 'Age'], inplace=True)

clinical_df['Age'] = pd.to_numeric(clinical_df['Age'], errors='coerce')
clinical_df.dropna(subset=['Age'], inplace=True)


clinical_df['Gender'] = clinical_df['Gender'].str.upper().map({'M': 'Male', 'F': 'Female'})

print("Cleaned Clinical Data:")
print(clinical_df.head())

zip_file_path = 'dataset.zip'


image_entries = []

with zipfile.ZipFile(zip_file_path, 'r') as archive:
    all_files = archive.namelist()
    print("Found files in ZIP:", all_files[:5])

    
    image_entries = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    
    for img_file in image_entries[:3]:
        with archive.open(img_file) as file:
            img = Image.open(io.BytesIO(file.read()))
            img = img.convert('L').resize((224, 224))  
            plt.imshow(img, cmap='gray')
            plt.title(f"Preview: {img_file}")
            plt.axis('off')
            plt.show()



icd10_map = {
    'C50.9': 'Malignant neoplasm of breast, unspecified',
    'D05.1': 'Lobular carcinoma in situ',
    'C50.3': 'Malignant neoplasm of upper-inner quadrant of breast',
    'N63': 'Unspecified lump in breast',
    
}


genai_jsonl_data = []
vlm_csv_data = []

with zipfile.ZipFile(zip_file_path, 'r') as archive:
    for _, row in clinical_df.iterrows():
        patient_id = row['Patient_ID']
        age = int(row['Age'])
        gender = row['Gender']
        icd_code = row['ICD10_Code']
        diagnosis = icd10_map.get(icd_code, icd_code)
        notes = row['Doctor_Notes']

        
        matched_img = next((img for img in image_entries if img.startswith(patient_id)), None)

        if matched_img:
            
            prompt = f"Patient ({gender}, {age} years old):\nClinical Notes: {notes}\nWhat is the likely diagnosis?"
            response = diagnosis
            genai_jsonl_data.append({
                "patient_id": patient_id,
                "image_file": matched_img,
                "prompt": prompt,
                "response": response
            })

            
            vlm_csv_data.append({
                "patient_id": patient_id,
                "image_file": matched_img,
                "text_prompt": notes,
                "label": diagnosis
            })
        else:
            print(f"No image found for {patient_id}")


with open("genai_dataset.jsonl", "w", encoding='utf-8') as f:
    for entry in genai_jsonl_data:
        f.write(json.dumps(entry) + '\n')
print("Saved GenAI prompt-response dataset to genai_dataset.jsonl")


pd.DataFrame(vlm_csv_data).to_csv("vlm_dataset.csv", index=False)
print("Saved Vision-Language dataset to vlm_dataset.csv")

plt.figure(figsize=(8, 5))
clinical_df['ICD10_Code'].map(icd10_map).value_counts().plot(kind='bar', color='skyblue')
plt.title("Diagnosis Distribution")
plt.xlabel("Diagnosis")
plt.ylabel("Number of Patients")
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

print("Data preparation complete.")

