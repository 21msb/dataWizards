from fastapi import FastAPI, HTTPException
import joblib

app = FastAPI()
# GET request
@app.get("/")
def read_root():
    return {"message": "Welcome to Tuwaiq Academy"}
# get request
@app.get("/items/{item_id}")
async def read_item(item_id):
    return {"item_id": item_id}

model = joblib.load('knn_model.joblib')
scaler = joblib.load('Models/scaler.joblib')



from pydantic import BaseModel
# Define a Pydantic model for input data validation
class InputFeatures(BaseModel):
    appearance: int
    games_injured: int
    award: int
    highest_value: int
    goals_total: int
    assists_total: int
    yellow_cards_total: int
    

def preprocessing(input_features: InputFeatures):
    dict_f = {
        'appearance': input_features.appearance,
        'games_injured': input_features.games_injured,
        'award': input_features.award,
        'highest_value': input_features.highest_value, 
        'goals_total': input_features.goals_total,
        'assists_total': input_features.assists_total,
        'yellow_cards_total': input_features.yellow_cards_total
        
}
    features_list = [dict_f[key] for key in sorted(dict_f)]
# Scale the input features
    scaled_features = scaler.transform([list(dict_f.values
    ())])
    return scaled_features



@app.post("/predict")
async def predict(input_features: InputFeatures):
    data = preprocessing(input_features)
    y_pred = model.predict(data)
    return {"pred": y_pred.tolist()[0]}

