#!/usr/bin/env python3

print("Testing document processing imports...")

try:
    import PyPDF2
    print("✅ PyPDF2 imported successfully")
except ImportError as e:
    print(f"❌ PyPDF2 import failed: {e}")

try:
    import docx
    print("✅ python-docx imported successfully")
except ImportError as e:
    print(f"❌ python-docx import failed: {e}")

try:
    import fitz  # PyMuPDF
    print("✅ PyMuPDF imported successfully")
except ImportError as e:
    print(f"❌ PyMuPDF import failed: {e}")

try:
    import pdfplumber
    print("✅ pdfplumber imported successfully")
except ImportError as e:
    print(f"❌ pdfplumber import failed: {e}")

print("\nTesting FastAPI imports...")
try:
    from fastapi import FastAPI, File, UploadFile
    print("✅ FastAPI imports successful")
except ImportError as e:
    print(f"❌ FastAPI import failed: {e}")

