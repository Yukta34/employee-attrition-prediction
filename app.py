"""
Employee Attrition Predictor + Retention Simulator (standalone app).

Needs in the same folder:  attrition_model.pkl  (created by the last cells of the notebook)
Run locally:               pip install -r requirements.txt gradio  &&  python app.py
Hugging Face Spaces:       upload app.py, attrition_model.pkl, requirements.txt (+ README.md with the Space card)
"""
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import gradio as gr

art = joblib.load("attrition_model.pkl")
model      = art["model"]              # calibrated model
explainer  = art["explainer_model"]    # raw tuned pipeline (for risk drivers)
thr, t_high = art["threshold"], art["t_high"]
columns    = art["columns"]
defaults   = art["defaults"]
role_dept  = art["role_dept"]
risk_all   = np.asarray(art["risk_all"])
choices    = art["choices"]

prep_fit   = explainer.named_steps["prep"]
booster    = explainer.named_steps["model"].get_booster()
feat_names = [n.split("__", 1)[1] for n in prep_fit.get_feature_names_out()]


def add_features(d):
    """Same engineered features as the notebook."""
    d = d.copy()
    d["IncomePerYearExp"] = d["MonthlyIncome"] / (d["TotalWorkingYears"] + 1)
    d["TenureRatio"] = d["YearsAtCompany"] / (d["TotalWorkingYears"] + 1)
    d["PromotionGap"] = d["YearsSinceLastPromotion"] / (d["YearsAtCompany"] + 1)
    d["SatisfactionScore"] = d[["JobSatisfaction", "EnvironmentSatisfaction",
                                "RelationshipSatisfaction", "WorkLifeBalance"]].mean(axis=1)
    return d


def risk(row):
    d = add_features(pd.DataFrame([row]))[columns]
    return float(model.predict_proba(d)[0, 1])


def top_drivers(row, k=3):
    Xr = add_features(pd.DataFrame([row]))[columns]
    c = booster.predict(xgb.DMatrix(prep_fit.transform(Xr)), pred_contribs=True)[0][:-1]
    idx = np.argsort(c)[::-1][:k]
    return [feat_names[i] for i in idx if c[i] > 0]


ACTIONS = {
    "Eliminate overtime":              lambda r: {**r, "OverTime": "No"},
    "Add stock options (+1 level)":    lambda r: {**r, "StockOptionLevel": min(3, r["StockOptionLevel"] + 1)},
    "15% salary raise":                lambda r: {**r, "MonthlyIncome": r["MonthlyIncome"] * 1.15},
    "Improve work-life balance (+1)":  lambda r: {**r, "WorkLifeBalance": min(4, r["WorkLifeBalance"] + 1)},
    "Improve job satisfaction (+1)":   lambda r: {**r, "JobSatisfaction": min(4, r["JobSatisfaction"] + 1)},
    "Promotion (reset promotion gap)": lambda r: {**r, "YearsSinceLastPromotion": 0},
    "Cut frequent travel":             lambda r: {**r, "BusinessTravel": "Travel_Rarely" if r["BusinessTravel"] == "Travel_Frequently" else r["BusinessTravel"]},
    "Reduce commute / remote option":  lambda r: {**r, "DistanceFromHome": min(r["DistanceFromHome"], 5)},
}


def what_if(row):
    base = risk(row)
    out = [(name, base, risk(f(row))) for name, f in ACTIONS.items()]
    tbl = pd.DataFrame(out, columns=["Action", "Current risk", "Risk after action"])
    tbl["Risk reduction"] = tbl["Current risk"] - tbl["Risk after action"]
    return base, tbl.sort_values("Risk reduction", ascending=False).reset_index(drop=True)


def build_row(age, income, overtime, years_company, total_years, job_sat, env_sat, wlb, rel_sat,
              distance, stock, marital, travel, role, since_promo, job_level):
    total_years = max(total_years, years_company)
    total_years = min(total_years, max(age - 18, 0))
    years_company = min(years_company, total_years)
    r = dict(defaults)
    r.update(Age=age, MonthlyIncome=income, OverTime=overtime, YearsAtCompany=years_company,
             TotalWorkingYears=total_years, JobSatisfaction=job_sat, EnvironmentSatisfaction=env_sat,
             WorkLifeBalance=wlb, RelationshipSatisfaction=rel_sat, DistanceFromHome=distance,
             StockOptionLevel=stock, MaritalStatus=marital, BusinessTravel=travel, JobRole=role,
             JobLevel=job_level, Department=role_dept[role],
             YearsSinceLastPromotion=min(since_promo, years_company),
             YearsInCurrentRole=min(defaults["YearsInCurrentRole"], years_company),
             YearsWithCurrManager=min(defaults["YearsWithCurrManager"], years_company))
    return r


def app_predict(*inputs):
    row = build_row(*inputs)
    base, wi = what_if(row)
    pct = (risk_all < base).mean() * 100
    drivers = top_drivers(row)
    if base >= t_high:
        verdict = "HIGH RISK: likely to leave"
    elif base >= thr:
        verdict = "MEDIUM RISK: likely to leave, act soon"
    else:
        verdict = "LOW RISK: likely to stay"
    summary = (f"{verdict}\n"
               f"Attrition probability: {base * 100:.1f}%\n"
               f"Action threshold: {thr * 100:.0f}%  (cost-optimised)\n"
               f"Riskier than {pct:.0f}% of employees in the dataset\n"
               f"Main factors pushing risk up: {', '.join(drivers) or 'none'}")
    table = wi.copy()
    for c in ["Current risk", "Risk after action", "Risk reduction"]:
        table[c] = (table[c] * 100).round(1).astype(str) + "%"
    return summary, table


demo = gr.Interface(
    fn=app_predict,
    inputs=[
        gr.Slider(18, 60, value=30, step=1, label="Age"),
        gr.Number(value=3000, label="Monthly Income"),
        gr.Radio(["Yes", "No"], value="No", label="OverTime"),
        gr.Slider(0, 40, value=2, step=1, label="Years at Company"),
        gr.Slider(0, 40, value=5, step=1, label="Total Working Years"),
        gr.Slider(1, 4, value=3, step=1, label="Job Satisfaction"),
        gr.Slider(1, 4, value=3, step=1, label="Environment Satisfaction"),
        gr.Slider(1, 4, value=3, step=1, label="Work-Life Balance"),
        gr.Slider(1, 4, value=3, step=1, label="Relationship Satisfaction"),
        gr.Slider(1, 30, value=10, step=1, label="Distance From Home"),
        gr.Slider(0, 3, value=1, step=1, label="Stock Option Level"),
        gr.Dropdown(choices["MaritalStatus"], value="Single", label="Marital Status"),
        gr.Dropdown(choices["BusinessTravel"], value="Travel_Rarely", label="Business Travel"),
        gr.Dropdown(choices["JobRole"], value="Sales Representative", label="Job Role"),
        gr.Slider(0, 15, value=1, step=1, label="Years Since Last Promotion"),
        gr.Slider(1, 5, value=1, step=1, label="Job Level"),
    ],
    outputs=[gr.Textbox(label="Risk assessment", lines=6),
             gr.Dataframe(label="Retention What-If Simulator (sorted by impact)")],
    title="Employee Attrition Predictor + Retention Simulator",
    description=("Calibrated attrition risk, a cost-optimised action threshold, the main risk drivers, and HR actions "
                 "ranked by modelled risk reduction. Trained on the synthetic IBM HR dataset; the simulator shows "
                 "model sensitivity, not proven cause and effect."),
)

if __name__ == "__main__":
    demo.launch()
