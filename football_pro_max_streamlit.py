
import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os
from datetime import datetime

API_KEY = "d8f1aacd76bc6359df94855c06b105eb"
HEADERS = {"x-apisports-key": API_KEY}
DATA_FILE = "historical_games_pro_max.csv"
LEAGUES_IDS = [39, 140, 61, 78, 135]

def normalize(value, min_val, max_val):
    return 0 if max_val - min_val == 0 else (value - min_val)/(max_val - min_val)

def get_fixtures_for_league(league_id, season=2026):
    url = "https://v3.football.api-sports.io/fixtures"
    params = {"league": league_id, "season": season}
    r = requests.get(url, headers=HEADERS, params=params)
    return r.json()["response"]

def get_last_fixtures(team_id, last=5):
    url = "https://v3.football.api-sports.io/fixtures"
    params = {"team": team_id, "last": last}
    r = requests.get(url, headers=HEADERS, params=params)
    return r.json()["response"]

def basic_stats(fixtures, team_id):
    goals_for = goals_against = wins = 0
    for match in fixtures:
        home = match["teams"]["home"]["id"]
        away = match["teams"]["away"]["id"]
        gf = match["goals"]["home"] if team_id == home else match["goals"]["away"]
        ga = match["goals"]["away"] if team_id == home else match["goals"]["home"]
        goals_for += gf
        goals_against += ga
        if gf > ga: wins += 1
    games = len(fixtures)
    return {"goals_for_avg": goals_for/games if games>0 else 0,
            "goals_against_avg": goals_against/games if games>0 else 0,
            "win_rate": wins/games if games>0 else 0}

def xg_stats(fixtures, team_id):
    xg_for = xg_against = 0
    games_with_xg = 0
    for match in fixtures:
        fixture_id = match["fixture"]["id"]
        url = "https://v3.football.api-sports.io/fixtures/statistics"
        params = {"fixture": fixture_id}
        r = requests.get(url, headers=HEADERS, params=params)
        stats_response = r.json()["response"]
        if not stats_response: continue
        for team in stats_response:
            is_team = team["team"]["id"] == team_id
            for item in team["statistics"]:
                if item["type"] == "Expected Goals" and item["value"] is not None:
                    if is_team: xg_for += float(item["value"])
                    else: xg_against += float(item["value"])
        games_with_xg += 1
    if games_with_xg == 0: return {"xg_for_avg":0,"xg_against_avg":0,"xg_diff":0}
    return {"xg_for_avg": xg_for/games_with_xg,
            "xg_against_avg": xg_against/games_with_xg,
            "xg_diff": (xg_for-xg_against)/games_with_xg}

def team_rating_advanced(team_id, home=False):
    fixtures = get_last_fixtures(team_id)
    basic = basic_stats(fixtures, team_id)
    xg = xg_stats(fixtures, team_id)
    goal_diff = basic["goals_for_avg"] - basic["goals_against_avg"]
    home_bonus = 0.1 if home else 0
    score = 0.4*normalize(xg["xg_diff"],-1.5,1.5) + 0.3*normalize(goal_diff,-2,2) + 0.2*basic["win_rate"] + home_bonus
    return score, basic, xg

def draw_probability(home_score, away_score):
    diff = abs(home_score-away_score)
    return max(0.05,0.35-diff)

def match_probabilities_advanced(home_id, away_id):
    home_score, home_basic, home_xg = team_rating_advanced(home_id, home=True)
    away_score, away_basic, away_xg = team_rating_advanced(away_id, home=False)
    diff = home_score-away_score
    home_win = 1/(1+math.exp(-diff))
    away_win = 1-home_win
    draw = draw_probability(home_score, away_score)
    total = home_win+away_win+draw
    return {"1": round(home_win/total,2), "X": round(draw/total,2), "2": round(away_win/total,2),
            "home_basic": home_basic, "away_basic": away_basic, "home_xg": home_xg, "away_xg": away_xg}

def build_dataset(fixtures_list):
    rows=[]
    for match in fixtures_list:
        home_id = match["teams"]["home"]["id"]
        away_id = match["teams"]["away"]["id"]
        home_score = team_rating_advanced(home_id,home=True)[0]
        away_score = team_rating_advanced(away_id,home=False)[0]
        result_diff = match["goals"]["home"] - match["goals"]["away"]
        result = "1" if result_diff>0 else "2" if result_diff<0 else "X"
        rows.append({"home_score":home_score,"away_score":away_score,"result":result})
    return pd.DataFrame(rows)

def train_model(df):
    X = df[["home_score","away_score"]]
    y = LabelEncoder().fit_transform(df["result"])
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)
    model = LogisticRegression(multi_class="multinomial", max_iter=1000)
    model.fit(X_train,y_train)
    return model,X_test,y_test

def predict_match_ml(model, home_id, away_id):
    home_score = team_rating_advanced(home_id,home=True)[0]
    away_score = team_rating_advanced(away_id,home=False)[0]
    probs = model.predict_proba([[home_score,away_score]])[0]
    return {"1":round(probs[0],2),"X":round(probs[1],2),"2":round(probs[2],2)}

st.title("Pro-Max Football Dashboard ⚽")

num_games_history = st.slider("Historical Games to Fetch per Team", 5,50,10)

if st.button("Fetch All Games & Update ML"):
    all_fixtures=[]
    for league_id in LEAGUES_IDS:
        league_games=get_fixtures_for_league(league_id)
        all_fixtures.extend(league_games)

    new_data=build_dataset(all_fixtures)
    new_data["timestamp"]=datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if os.path.exists(DATA_FILE):
        historical_df=pd.read_csv(DATA_FILE)
        historical_df=pd.concat([historical_df,new_data]).drop_duplicates(subset=["home_score","away_score","result"],keep='last').reset_index(drop=True)
    else:
        historical_df=new_data
    historical_df.to_csv(DATA_FILE,index=False)
    st.write(f"Dataset updated! Total games: {len(historical_df)}")

    if len(historical_df)>=2:
        model,X_test,y_test=train_model(historical_df)
        preds=model.predict(X_test)
        st.write("ML trained. Accuracy on historical test set:", round(accuracy_score(y_test,preds),2))
    else:
        st.warning("Not enough data for ML model!")

    st.subheader("Sample Predictions for Recent Games")
    for match in all_fixtures[:5]:
        home_id=match["teams"]["home"]["id"]
        away_id=match["teams"]["away"]["id"]
        ml_pred=predict_match_ml(model,home_id,away_id)
        st.write(f"{match['teams']['home']['name']} vs {match['teams']['away']['name']}: {ml_pred}")
