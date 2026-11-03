from fastapi import FastAPI, HTTPException, status
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from predict import predict_data, CLASS_NAMES

app = FastAPI(title="Wine Classifier API", description="FastAPI lab using the sklearn Wine dataset", docs_url=None)

CUSTOM_CSS = """
<style>
body { background: linear-gradient(135deg, #1a0010 0%, #4a0e2b 45%, #2b0a3d 100%) fixed; min-height: 100vh; }
.swagger-ui, .swagger-ui * { color: #ff4d4d !important; fill: #ff4d4d !important; }
.swagger-ui .topbar { background: rgba(0,0,0,.5); border-bottom: 2px solid #ff1a1a; }
.swagger-ui .info .title::before { content: "\1F377  "; }
.swagger-ui .opblock, .swagger-ui section.models, .swagger-ui .model-box,
.swagger-ui .opblock-tag, .swagger-ui table, .swagger-ui .responses-inner,
.swagger-ui .highlight-code, .swagger-ui .microlight {
  background: rgba(0,0,0,.45) !important; border: 2px solid #ff1a1a !important; border-radius: 10px; }
.swagger-ui .opblock .opblock-section-header { background: rgba(0,0,0,.6) !important; border-bottom: 1px solid #ff1a1a; }
.swagger-ui input, .swagger-ui textarea, .swagger-ui select, .swagger-ui .btn {
  background: #1a0008 !important; border: 2px solid #ff1a1a !important; border-radius: 6px; }
.swagger-ui .btn.execute { background: #ff1a1a !important; color: #fff !important; }
.swagger-ui .scheme-container { background: transparent; box-shadow: none; }
</style>
"""


@app.get("/docs", include_in_schema=False)
async def custom_docs():
    html = get_swagger_ui_html(openapi_url=app.openapi_url, title="Wine Classifier API").body.decode()
    return HTMLResponse(html.replace("</head>", CUSTOM_CSS + "</head>"))


class WineData(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "alcohol": 13.2, "malic_acid": 1.78, "ash": 2.14, "alcalinity_of_ash": 11.2,
                "magnesium": 100.0, "total_phenols": 2.65, "flavanoids": 2.76,
                "nonflavanoid_phenols": 0.26, "proanthocyanins": 1.28, "color_intensity": 4.38,
                "hue": 1.05, "od280_od315_of_diluted_wines": 3.4, "proline": 1050.0,
            }
        }
    }


class WineResponse(BaseModel):
    response: int
    class_name: str
    confidence: float


@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}


@app.post("/predict", response_model=WineResponse)
async def predict_wine(wine: WineData):
    try:
        features = [list(wine.model_dump().values())]
        pred, proba = predict_data(features)
        label = int(pred[0])
        return WineResponse(response=label, class_name=CLASS_NAMES[label],
                            confidence=round(float(proba[0].max()), 4))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
