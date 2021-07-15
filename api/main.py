from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

import pickle
import numpy as np

import dataloading.transform as T

app = FastAPI()

with open('checkpoints/LogisticRegression.pkl','rb') as f:
    model = pickle.load(f)

templates = Jinja2Templates(directory="./api/")    

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Index page containing a simple file upload form.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/")
async def predict(file: UploadFile = File(...)):
    """Prediction endpoint expecting a 3-axis time-series as a text file.
    Time-series is integrated and a polynomial is fit to the data to feed into the model.
    Prediction is returned as single-value array containing a class index.
    """
    # Test with curl http://0.0.0.0:80/ -F "file=@A_Template_Acceleration1-1.txt"
    
    try:
        data = np.loadtxt(file.file)
        if data.shape[1] != 3 or len(data) < 3:
            return {"status": "failed", "detail": "invalid data"}

        data = np.cumsum(data, axis=0) # Integrate
        data = np.polynomial.polynomial.polyfit(np.arange(len(data)), data, deg=2)

        prediction = model.predict(data.reshape((-1, model.W.shape[0])))

        return {"status": "success", "predictions": prediction.tolist()}
    except Exception as e:
        return {"status": "failed", "detail": str(e)}


