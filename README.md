# amazon-google-product-matching
Matching Amazon and Google products using text similarity, confidence scoring and FastAPI


A product matching system that identifies identical products 
across Amazon and Google catalogues using text similarity.

## Features
- Smart candidate generation using blocking
- Multi-signal scoring (title, price)
- Confidence levels (HIGH/MEDIUM/LOW/NONE)
- Evaluated against ground truth — F1: 0.617
- Interactive similarity checker
- REST API built with FastAPI

## Files
- `Ggl_vs_Azm.ipynb` — full pipeline
- `app.py` — FastAPI REST API

## How to Run
1. Install dependencies:
pip install pandas numpy scikit-learn fastapi uvicorn

2. Run the API:
uvicorn app:app --reload

3. Open browser:
http://127.0.0.1:8000/docs
