"""Candidate-level and observation-level metrics; no hidden route information used."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,log_loss,f1_score
def evaluate_model(model,df,feature_cols):
    x=df[feature_cols]; start=time.perf_counter(); proba=model.predict_proba(x)[:,list(model.classes_).index(1)];elapsed=time.perf_counter()-start; pred=(proba>=.5).astype(int); p,r,f,_=precision_recall_fscore_support(df.label,pred,average="binary",zero_division=0); pm,rm,fm,_=precision_recall_fscore_support(df.label,pred,average="macro",zero_division=0)
    scored=df[["trip_id","observation_id","candidate_arc_id","label_arc_id","label"]].copy();scored["score"]=proba; top=scored.sort_values(["trip_id","observation_id","score"],ascending=[True,True,False]).groupby(["trip_id","observation_id"],as_index=False).first(); top_acc=(top.candidate_arc_id==top.label_arc_id).mean(); top_macro_f1=f1_score(top.label_arc_id,top.candidate_arc_id,average="macro",zero_division=0); continuity=[]
    for _,g in top.groupby("trip_id"): continuity.append((g.candidate_arc_id.to_numpy()[1:]!=g.candidate_arc_id.to_numpy()[:-1]).mean() if len(g)>1 else 1.)
    return {"candidate_accuracy":accuracy_score(df.label,pred),"candidate_precision":p,"candidate_recall":r,"candidate_f1":f,"candidate_macro_f1":fm,"log_loss":log_loss(df.label,proba,labels=[0,1]),"student_top1_accuracy":top_acc,"student_top1_macro_f1":top_macro_f1,"route_continuity_change_rate":float(np.mean(continuity)),"inference_seconds_per_row":elapsed/max(len(df),1)},scored
def save_comparison(rows,out_dir):
    Path(out_dir).mkdir(parents=True,exist_ok=True);d=pd.DataFrame(rows);d.to_csv(Path(out_dir)/"model_comparison.csv",index=False);return d
