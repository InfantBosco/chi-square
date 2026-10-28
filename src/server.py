import os
import io
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.retail_engine import RetailAnalysisEngine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
PUBLIC_DIR = os.path.join(BASE_DIR, 'public')

app = FastAPI(title="The Retail Intelligence Ledger", version="2.0")

# Global engine instance
ENGINE = RetailAnalysisEngine()

# Ensure static directory exists and mount if present
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class AnalyzeRequest(BaseModel):
    target_type: str = 'high_value'
    feature_col: str = 'Description'
    threshold: int = 25

class ScenarioRequest(BaseModel):
    product_name: str
    country: str = 'India'
    quantity: int = 2
    unit_price: float = 650.0
    threshold: int = 25

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "service": "The Retail Intelligence Ledger"}

@app.get("/", response_class=HTMLResponse)
def get_index():
    for candidate in [
        os.path.join(STATIC_DIR, "index.html"),
        os.path.join(PUBLIC_DIR, "index.html"),
        os.path.join(BASE_DIR, "api", "index.html"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "api", "index.html"),
        os.path.join(os.getcwd(), "static", "index.html"),
        os.path.join(os.getcwd(), "public", "index.html"),
        os.path.join(os.getcwd(), "api", "index.html"),
    ]:
        if os.path.exists(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                return f.read()
    return "<h1>The Retail Intelligence Ledger is Initializing...</h1>"

@app.get("/api/load-demo")
def load_demo():
    demo_candidates = [
        os.path.join(BASE_DIR, 'data', 'processed', 'online_retail_demo.csv'),
        os.path.join(BASE_DIR, 'api', 'online_retail_demo.csv'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'api', 'online_retail_demo.csv'),
        os.path.join(os.getcwd(), 'data', 'processed', 'online_retail_demo.csv'),
        os.path.join(os.getcwd(), 'api', 'online_retail_demo.csv'),
    ]
    demo_path = None
    for c in demo_candidates:
        if os.path.exists(c):
            demo_path = c
            break
    if demo_path:
        ENGINE.load_data(demo_path)
    res = ENGINE.run_analysis(target_type='high_value', feature_col='Description', threshold=25)
    return {"status": "success", "message": "Official UCI Online Retail Demo Loaded", "data": res}

@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    temp_dir = tempfile.gettempdir()
    custom_path = os.path.join(temp_dir, "custom_store_upload.csv")
    
    with open(custom_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    count = ENGINE.load_data(custom_path)
    # Automatically choose a text column
    feature_col = 'Description' if 'Description' in ENGINE.df.columns else ENGINE.df.select_dtypes(include=['object']).columns[0]
    res = ENGINE.run_analysis(target_type='high_value', feature_col=feature_col, threshold=25)
    return {"status": "success", "message": f"Successfully loaded {count:,} store records", "data": res}

@app.post("/api/analyze")
def run_analysis_endpoint(req: AnalyzeRequest):
    res = ENGINE.run_analysis(target_type=req.target_type, feature_col=req.feature_col, threshold=req.threshold)
    return {"status": "success", "data": res}

@app.post("/api/predict-scenario")
def predict_scenario_endpoint(req: ScenarioRequest):
    res = ENGINE.predict_scenario(
        product_name=req.product_name,
        country=req.country,
        quantity=req.quantity,
        unit_price=req.unit_price,
        threshold=req.threshold
    )
    return {"status": "success", "data": res}

@app.get("/api/product-list")
def get_product_list():
    counts = ENGINE.df['Description'].value_counts()
    top_common = counts.head(15).index.tolist()
    rare_niche = counts[counts < 10].sample(min(15, (counts < 10).sum()), random_state=42).index.tolist()
    return {
        "common_products": top_common,
        "niche_products": rare_niche,
        "all_top_50": counts.head(50).index.tolist()
    }

@app.get("/api/download-report")
def download_report(target_type: str = 'high_value', threshold: int = 25):
    res = ENGINE.run_analysis(target_type=target_type, threshold=threshold)
    pdf_bytes = ENGINE.generate_pdf_report(res)
    
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=Retail_Intelligence_Audit_N{threshold}.pdf"}
    )
