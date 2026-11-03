# FastAPI Lab — Wine Classifier

Adaptation of the [MLOps FastAPI Lab](https://github.com/raminmohammadi/MLOps/tree/main/Labs/API_Labs/FastAPI_Labs) using the **sklearn Wine dataset** (178 samples, 13 chemical features, 3 cultivar classes) instead of Iris, with a **RandomForestClassifier** instead of a Decision Tree.

## Structure
```
LAB3/
├── model/wine_model.pkl   # trained model (created by train.py)
├── src/
│   ├── data.py            # load + split Wine dataset
│   ├── train.py           # train RandomForest, save model
│   ├── predict.py         # load model, predict
│   └── main.py            # FastAPI app
├── requirements.txt
└── README.md
```

## Setup
```bash
python -m venv venv
venv\Scripts\activate          # Windows  (source venv/bin/activate on mac/linux)
pip install -r requirements.txt
```

## Train
```bash
cd src
python train.py
```

## Serve
```bash
uvicorn main:app --reload
```
Open http://127.0.0.1:8000/docs for the Swagger UI.

## Endpoints
| Method | Path       | Description                          |
|--------|------------|--------------------------------------|
| GET    | `/`        | Health check                         |
| POST   | `/predict` | Predict wine class from 13 features  |

### Example request
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"alcohol":13.2,"malic_acid":1.78,"ash":2.14,"alcalinity_of_ash":11.2,"magnesium":100,"total_phenols":2.65,"flavanoids":2.76,"nonflavanoid_phenols":0.26,"proanthocyanins":1.28,"color_intensity":4.38,"hue":1.05,"od280_od315_of_diluted_wines":3.4,"proline":1050}'
```
Response:
```json
{"response": 0, "class_name": "class_0", "confidence": 0.99}
```
