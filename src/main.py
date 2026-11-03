from fastapi import FastAPI, HTTPException, status
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from predict import predict_data, CLASS_NAMES

app = FastAPI(title="Wine Classifier API", description="FastAPI lab using the sklearn Wine dataset", docs_url=None)

CUSTOM_CSS = """
<style>
body { background: linear-gradient(135deg, #1a0010 0%, #4a0e2b 45%, #2b0a3d 100%) fixed; min-height: 100vh; }
.swagger-ui { color: #f3e6ec; }
.swagger-ui .topbar { background: rgba(0,0,0,.35); border-bottom: 2px solid #c2185b; }
.swagger-ui .info .title, .swagger-ui .info p, .swagger-ui .opblock-tag,
.swagger-ui .opblock .opblock-summary-path, .swagger-ui .opblock .opblock-summary-description,
.swagger-ui table thead tr th, .swagger-ui .parameter__name, .swagger-ui .response-col_status,
.swagger-ui .tab li, .swagger-ui .opblock-section-header h4, .swagger-ui label, .swagger-ui .model-title { color: #f8e1ea !important; }
.swagger-ui .info .title::before { content: "\1F377  "; }
.swagger-ui .opblock { background: rgba(255,255,255,.06) !important; backdrop-filter: blur(8px);
  border-radius: 12px; box-shadow: 0 6px 24px rgba(0,0,0,.4); }
.swagger-ui .opblock .opblock-section-header { background: rgba(0,0,0,.25); }
.swagger-ui section.models { background: rgba(255,255,255,.05); border-radius: 12px; }
.swagger-ui .btn.execute { background: #c2185b; border-color: #c2185b; }
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
