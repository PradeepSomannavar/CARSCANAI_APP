import os
import cloudinary
import cloudinary.uploader
from supabase import create_client, Client
from dotenv import load_dotenv
from pathlib import Path

# Load env from current directory
load_dotenv(Path(__file__).parent / ".env")

# Cloudinary Config
# These should be set in your .env file
cloudinary.config(
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key = os.getenv("CLOUDINARY_API_KEY"),
    api_secret = os.getenv("CLOUDINARY_API_SECRET"),
    secure = True
)

# Supabase Config
# These should be set in your .env file
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def get_supabase_client():
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Error initializing Supabase client: {e}")
        return None

def upload_to_cloudinary(file_content: bytes, folder: str, filename: str) -> str:
    """
    Uploads file bytes to Cloudinary and returns the secure URL.
    :param file_content: The bytes of the file to upload.
    :param folder: The folder name in Cloudinary.
    :param filename: The desired filename (public_id).
    """
    try:
        # Use resource_type="auto" to handle both images and raw files (PDFs)
        response = cloudinary.uploader.upload(
            file_content,
            folder=folder,
            public_id=filename.split('.')[0],
            resource_type="auto"
        )
        return response.get("secure_url")
    except Exception as e:
        print(f"Cloudinary upload error: {e}")
        return None

def insert_car_scan(severity: float, image_url: str, pdf_url: str):
    """
    Inserts a scan record into Supabase car_scans table.
    :param severity: The calculated severity score (0-100).
    :param image_url: The Cloudinary URL for the annotated image.
    :param pdf_url: The Cloudinary URL for the generated PDF report.
    """
    supabase = get_supabase_client()
    if not supabase:
        print("Supabase client not initialized. Check your credentials in .env")
        return None
    
    try:
        data = {
            "severity": severity,
            "image_url": image_url,
            "pdf_url": pdf_url
        }
        # Note: Ensure the table 'car_scans' exists in your Supabase project.
        res = supabase.table("car_scans").insert(data).execute()
        return res.data
    except Exception as e:
        print(f"Supabase insert error: {e}")
        return None
