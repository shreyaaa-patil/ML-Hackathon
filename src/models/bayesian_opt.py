"""Genuine Optuna TPE optimisation of the selected candidate-ranking model."""
from pathlib import Path
import json,time
import optuna
from joblib import dump
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from src.models.train import make_pipeline,stratified_trip_sample
def tune_selected(df,feature_cols,model_name,output_dir,n_trials=30,n_splits=5,seed=42,max_rows=50000):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True); d=stratified_trip_sample(df,max_rows,seed);X,y,g=d[feature_cols],d.label.astype(int),d.trip_id; cv=GroupKFold(n_splits=min(n_splits,g.nunique()))
    def params(t):
        if model_name=="logistic_regression":return {"C":t.suggest_float("C",1e-3,100,log=True)}
        if model_name=="knn":return {"n_neighbors":t.suggest_int("n_neighbors",3,35),"weights":t.suggest_categorical("weights",["uniform","distance"]),"p":t.suggest_int("p",1,2)}
        if model_name=="random_forest":return {"n_estimators":t.suggest_int("n_estimators",100,300),"max_depth":t.suggest_int("max_depth",5,35),"min_samples_leaf":t.suggest_int("min_samples_leaf",1,8),"max_features":t.suggest_categorical("max_features",["sqrt","log2"])}
        if model_name=="hist_gradient_boosting":return {"max_iter":t.suggest_int("max_iter",80,300),"learning_rate":t.suggest_float("learning_rate",.01,.2,log=True),"max_leaf_nodes":t.suggest_int("max_leaf_nodes",15,63),"l2_regularization":t.suggest_float("l2_regularization",1e-6,10,log=True)}
        return {"C":t.suggest_float("C",.1,100,log=True),"gamma":t.suggest_float("gamma",1e-4,1,log=True)}
    def objective(t):
        start=time.perf_counter();s=[]; trial_params=params(t)
        for tr,va in cv.split(X,y,g): m=make_pipeline(model_name,seed,**trial_params);m.fit(X.iloc[tr],y.iloc[tr]);s.append(f1_score(y.iloc[va],m.predict(X.iloc[va]),average="macro",zero_division=0))
        t.set_user_attr("duration_seconds",time.perf_counter()-start);return float(sum(s)/len(s))
    study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=seed));study.optimize(objective,n_trials=n_trials);study.trials_dataframe().to_csv(out/"trial_history.csv",index=False);(out/"best_params.json").write_text(json.dumps(study.best_params,indent=2));dump(study,out/"optuna_study.joblib");m=make_pipeline(model_name,seed,**study.best_params);m.fit(X,y);dump(m,out/"tuned_model.joblib");return study,m
