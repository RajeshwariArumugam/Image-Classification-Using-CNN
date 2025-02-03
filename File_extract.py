import os

zip_path = "ZIP FILE PATH"  # Update path

if os.path.exists(zip_path):
    print("ZIP file found.")
else:
    print("ZIP file NOT found. Check the path!")

import zipfile
import os

zip_path = r"ZIP FILE PATH"  # Use raw string (r"")
extract_path = r"PATH WHERE THE EXTRACTED TO BE SAVED"  # Change this to a valid folder

# Ensure the extraction path exists
os.makedirs(extract_path, exist_ok=True)

# Extract ZIP
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Dataset extracted successfully!")

print("Extracted folders:", os.listdir(extract_path))
