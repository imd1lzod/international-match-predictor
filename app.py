import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn 
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
        data_pack = joblib.load("football_model_pack.joblib")
        model = data_pack["model"]
        elo_ratings = data_pack["elo_ratings"]
        feature_cols = data_pack["featured_cols"]
        INITIAL_ELO = data_pack["initial_elo"]
except Exception as e:
        print(f"Faylni yuklashda xatolik - {e}")

class MatchRequest(BaseModel):
    home_team: str
    away_team: str
    neutral: bool = True
    is_world_cup: bool = True

def get_team_elo(team_name: str):
    return elo_ratings.get(team_name, INITIAL_ELO)


@app.post("/predict")
def predict_match(match: MatchRequest):
    home_elo = get_team_elo(match.home_team)
    away_elo = get_team_elo(match.away_team)
    
    elo_diff = home_elo - away_elo
    
    input_data = pd.DataFrame(
        [[elo_diff, match.neutral, match.is_world_cup]], 
        columns=feature_cols
    )
    
    try:
        probabilities = model.predict_proba(input_data)[0]
        result_dict = dict(zip(model.classes_, probabilities))
        
        return {
            "status": "success",
            "teams": {
                "home": match.home_team,
                "away": match.away_team
            },
            "pre_match_elo": {
                "home_elo": round(home_elo, 2),
                "away_elo": round(away_elo, 2),
                "diff": round(elo_diff, 2)
            },
            "probabilities": {
                "home_win": round(float(result_dict.get('home_win', 0)), 4),
                "away_win": round(float(result_dict.get('away_win', 0)), 4),
                "draw": round(float(result_dict.get('draw', 0)), 4)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bashorat qilishda xatolik yuz berdi")


@app.get("/")
def hello():
        return "Match Predictor ishlamoqda"

if __name__ == "__main__":
        uvicorn.run(app)