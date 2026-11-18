"""Training of the five mandated binary candidate-ranking classifiers."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
MODEL_FAMILIES={"logistic_regression":"linear_and_probabilistic","knn":"instance_based","random_forest":"ensemble","hist_gradient_boosting":"boosting","rbf_svm":"kernel_based"}
def make_pipeline(name,seed=42,**params):
    if name=="logistic_regression": model=LogisticRegression(**{"max_iter":1000,"class_weight":"balanced","random_state":seed,**params}); scale=True
    elif name=="knn": model=KNeighborsClassifier(**{"n_neighbors":15,"weights":"distance",**params}); scale=True
    elif name=="random_forest": model=RandomForestClassifier(**{"n_estimators":150,"class_weight":"balanced_subsample","n_jobs":-1,"random_state":seed,**params}); scale=False
    elif name=="hist_gradient_boosting": model=HistGradientBoostingClassifier(**{"max_iter":150,"learning_rate":.08,"random_state":seed,**params}); scale=False
    elif name=="rbf_svm": model=SVC(**{"C":2.,"kernel":"rbf","probability":True,"class_weight":"balanced","random_state":seed,**params}); scale=True
    else: raise ValueError(name)
    steps=[("imputer",SimpleImputer(strategy="median"))]+([("scaler",StandardScaler())] if scale else [])+[("model",model)]
    return Pipeline(steps)
def stratified_trip_sample(df,max_rows,seed=42):
    if len(df)<=max_rows:return df
    rng=np.random.default_rng(seed); chunks=[]
    for _,g in df.groupby("trip_id"):
        n=max(20,round(max_rows*len(g)/len(df))); pos=g[g.label==1]; neg=g[g.label==0]; take_pos=min(len(pos),max(1,n//2)); take_neg=min(len(neg),max(1,n-take_pos)); chunks += [pos.iloc[rng.choice(len(pos),take_pos,replace=False)],neg.iloc[rng.choice(len(neg),take_neg,replace=False)]]
    return pd.concat(chunks,ignore_index=True).sample(frac=1,random_state=seed).head(max_rows)
def train_five_models(df,feature_cols,output_dir,seed=42,n_splits=5,max_rows=60000):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); sampled=stratified_trip_sample(df,max_rows,seed); X,y,g=sampled[feature_cols],sampled.label.astype(int),sampled.trip_id
    cv=GroupKFold(n_splits=min(n_splits,g.nunique())); rows=[]
    for name,family in MODEL_FAMILIES.items():
        scores=[]
        for tr,va in cv.split(X,y,g):
            m=make_pipeline(name,seed);m.fit(X.iloc[tr],y.iloc[tr]);scores.append(f1_score(y.iloc[va],m.predict(X.iloc[va]),average="macro",zero_division=0))
        m=make_pipeline(name,seed); start=time.perf_counter();m.fit(X,y);train_s=time.perf_counter()-start; start=time.perf_counter();m.predict_proba(X.iloc[:min(5000,len(X))]);infer=(time.perf_counter()-start)/min(5000,len(X));dump(m,out/f"{name}.joblib")
        rows.append({"model":name,"algorithm_family":family,"training_rows":len(sampled),"cv_macro_f1_mean":np.mean(scores),"cv_macro_f1_std":np.std(scores),"training_seconds":train_s,"inference_seconds_per_row":infer})
    result=pd.DataFrame(rows).sort_values("cv_macro_f1_mean",ascending=False);result.to_csv(out/"training_cv_results.csv",index=False);return result
