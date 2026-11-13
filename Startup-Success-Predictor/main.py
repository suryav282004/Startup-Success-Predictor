import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss

st.set_page_config(page_title="Startup Success Probability Prediction", page_icon="", layout="wide")
st.title(" Startup Success Probability Prediction and Business Performance Analytics")
st.subheader("Using Ensemble Machine Learning")
st.write("Estimate startup success probability using funding, industry, revenue, team size, growth and business indicators.")

INDUSTRIES = ["Technology / SaaS", "FinTech", "Healthcare", "E-commerce", "Education", "Clean Energy", "Food & Beverage", "Media & Entertainment", "Agriculture / AgriTech", "Logistics"]
NUM = ["Funding_Cr", "Annual_Revenue_Cr", "Team_Size", "Annual_Growth_Pct", "Customer_Growth_Pct", "Customer_Retention_Pct", "Monthly_Burn_Lakh", "Runway_Months", "Startup_Age_Years", "Market_Size_Cr"]
CAT = ["Industry"]
FEATURES = NUM + CAT
EFFECT = {"Technology / SaaS":.25,"FinTech":.20,"Healthcare":.18,"E-commerce":.05,"Education":.08,"Clean Energy":.15,"Food & Beverage":-.05,"Media & Entertainment":-.02,"Agriculture / AgriTech":.10,"Logistics":.06}

@st.cache_data
def make_data(n=8000, seed=42):
    r=np.random.default_rng(seed); industry=r.choice(INDUSTRIES,n)
    funding=np.clip(r.lognormal(1,1,n),.05,500)
    revenue=np.clip(r.lognormal(-.2,1.1,n),0,1000)
    team=np.clip(r.lognormal(3,.7,n).astype(int),2,1500)
    growth=np.clip(r.normal(35,35,n),-60,250)
    cust_growth=np.clip(r.normal(30,30,n),-70,220)
    retention=np.clip(r.normal(78,15,n),20,100)
    burn=np.clip(r.lognormal(3.5,.9,n),1,3000) # INR lakh/month
    runway=np.clip(r.normal(18,10,n),1,60)
    age=np.clip(r.gamma(2.2,2,n),.2,20)
    market=np.clip(r.lognormal(3,1,n),1,100000) # INR crore
    ind=np.array([EFFECT[x] for x in industry])
    score=(.55*np.log1p(funding)+.75*np.log1p(revenue)+.012*growth+.010*cust_growth+.025*(retention-70)+.035*runway+.12*np.log1p(market)+.08*np.log1p(team)-.0007*burn-.025*np.maximum(age-8,0)+ind+r.normal(0,1.2,n))
    prob=1/(1+np.exp(-score/2.3)); y=r.binomial(1,prob)
    return pd.DataFrame({"Funding_Cr":funding,"Annual_Revenue_Cr":revenue,"Team_Size":team,"Annual_Growth_Pct":growth,"Customer_Growth_Pct":cust_growth,"Customer_Retention_Pct":retention,"Monthly_Burn_Lakh":burn,"Runway_Months":runway,"Startup_Age_Years":age,"Market_Size_Cr":market,"Industry":industry,"Success":y})

def preprocessor():
    num_pipe=Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())])
    cat_pipe=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("encode",OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("numeric",num_pipe,NUM),("category",cat_pipe,CAT)])

@st.cache_resource
def train():
    df=make_data(); X=df[FEATURES]; y=df.Success
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    rf=Pipeline([("prep",preprocessor()),("model",RandomForestClassifier(n_estimators=220,max_depth=12,min_samples_split=5,min_samples_leaf=2,class_weight="balanced",random_state=42,n_jobs=-1))])
    et=Pipeline([("prep",preprocessor()),("model",ExtraTreesClassifier(n_estimators=220,max_depth=12,min_samples_split=5,min_samples_leaf=2,class_weight="balanced",random_state=42,n_jobs=-1))])
    lr=Pipeline([("prep",preprocessor()),("model",LogisticRegression(max_iter=1200,class_weight="balanced"))])
    ens=VotingClassifier([("rf",rf),("et",et),("lr",lr)],voting="soft",weights=[3,3,2])
    models={"Random Forest":rf,"Extra Trees":et,"Logistic Regression":lr,"Soft Voting Ensemble":ens}
    metrics={}
    for name,m in models.items():
        m.fit(Xtr,ytr); pred=m.predict(Xte); p=m.predict_proba(Xte)[:,1]
        metrics[name]={"Accuracy":accuracy_score(yte,pred),"F1":f1_score(yte,pred),"ROC-AUC":roc_auc_score(yte,p),"Log Loss":log_loss(yte,p)}
    return models,metrics,rf,df

with st.spinner("Training machine learning models..."):
    models,metrics,rf,training_data=train()

st.sidebar.header("Indian Startup Profile")
st.sidebar.caption("Funding/revenue/market size: ₹ Crore | Monthly burn: ₹ Lakh")
industry=st.sidebar.selectbox("Industry",INDUSTRIES)
funding=st.sidebar.number_input("Total Funding (₹ Crore)",0.0,10000.0,5.0,step=.5)
revenue=st.sidebar.number_input("Annual Revenue (₹ Crore)",0.0,10000.0,2.0,step=.5)
team=st.sidebar.number_input("Team Size (employees)",1,10000,25)
growth=st.sidebar.slider("Annual Revenue Growth (%)",-100,500,40)
cust_growth=st.sidebar.slider("Annual Customer Growth (%)",-100,500,35)
retention=st.sidebar.slider("Customer Retention (%)",0,100,80)
burn=st.sidebar.number_input("Monthly Operating Burn (₹ Lakh)",.1,100000.0,25.0,step=1.0)
runway=st.sidebar.slider("Cash Runway (months)",1,60,18)
age=st.sidebar.slider("Startup Age (years)",0.0,30.0,3.0,.5)
market=st.sidebar.number_input("Estimated Market Size (₹ Crore)",1.0,1000000.0,5000.0,step=100.0)
model_name=st.sidebar.selectbox("Prediction Model",list(models.keys()),index=3)
predict=st.sidebar.button("Predict Startup Success",use_container_width=True)

st.subheader("Model Performance Comparison")
st.dataframe(pd.DataFrame([{"Model":n,"Accuracy":f"{v['Accuracy']*100:.2f}%","F1 Score":f"{v['F1']:.3f}","ROC-AUC":f"{v['ROC-AUC']:.3f}","Log Loss ↓":f"{v['Log Loss']:.3f}"} for n,v in metrics.items()]),use_container_width=True,hide_index=True)

if predict:
    row=pd.DataFrame([{"Funding_Cr":funding,"Annual_Revenue_Cr":revenue,"Team_Size":team,"Annual_Growth_Pct":growth,"Customer_Growth_Pct":cust_growth,"Customer_Retention_Pct":retention,"Monthly_Burn_Lakh":burn,"Runway_Months":runway,"Startup_Age_Years":age,"Market_Size_Cr":market,"Industry":industry}],columns=FEATURES)
    p=models[model_name].predict_proba(row)[0,1]; low=1-p
    st.divider(); st.header("Startup Prediction Result")
    a,b,c=st.columns(3)
    a.metric("Higher Success Probability",f"{p*100:.1f}%")
    b.metric("Lower Success Probability",f"{low*100:.1f}%")
    c.metric("Model Assessment","Higher probability" if p>=.5 else "Lower probability")
    st.progress(int(p*100))
    if p>=.70: st.success("The model assigns a higher probability of success.")
    elif p>=.40: st.warning("The model assessment is mixed. Review the business indicators.")
    else: st.error("The model assigns a lower probability of success.")
    st.caption(f"Model: {model_name}. This output is based on synthetic demonstration data.")
    st.subheader("Business Indicator Review")
    notes=[]
    if runway<9: notes.append("Cash runway is short. Review spending and funding plans.")
    elif runway>=18: notes.append("Cash runway is relatively strong based on the entered value.")
    if retention<60: notes.append("Customer retention is low. Investigate churn and product-market fit.")
    elif retention>=85: notes.append("Customer retention is a positive signal.")
    if growth<0: notes.append("Annual growth is negative. Review revenue trends.")
    elif growth>=50: notes.append("Annual revenue growth is strong.")
    funding_lakh=funding*100 # 1 crore = 100 lakh
    notes.append(f"Funding-to-burn estimate: {funding_lakh/burn:.1f} months at the current burn rate, excluding revenue and other cash flows.")
    for note in notes: st.info(note)

st.divider(); st.subheader("Important Startup Success Factors (Random Forest)")
try:
    prep=rf.named_steps["prep"]; clf=rf.named_steps["model"]
    names=prep.get_feature_names_out()
    imp=pd.DataFrame({"Feature":names,"Importance":clf.feature_importances_}).sort_values("Importance",ascending=False)
    st.bar_chart(imp.head(12).set_index("Feature"))
except Exception:
    st.info("Feature importance is unavailable.")

with st.expander("View Sample Training Dataset"):
    st.dataframe(training_data.head(25),use_container_width=True)

st.divider(); st.subheader("About This Project")
st.write("""Startup Success Predictor – India Edition is a supervised machine learning application that estimates whether a startup has a higher or lower predicted probability of success using business and financial indicators. The application accepts funding, annual revenue and estimated market size in Indian Rupees (INR Crore), and monthly operating burn in INR Lakh. It compares Random Forest, Extra Trees and Logistic Regression, and combines their probability outputs using a Soft Voting Ensemble. The dashboard presents model evaluation metrics, startup success probability, business indicator observations and feature importance. The project demonstrates data generation, preprocessing, classification, model evaluation and interactive dashboard development.""")
st.caption("Technologies: Python | Streamlit | Pandas | NumPy | Scikit-learn | Currency: INR")
