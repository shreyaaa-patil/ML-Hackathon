"""Spatial-indexed candidate road-segment generation."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from src.data.geo_utils import local_projector,point_to_segments_m
def _geometry(nodes,arcs,p):
    valid=arcs[arcs.from_node_id.between(0,len(nodes)-1)&arcs.to_node_id.between(0,len(nodes)-1)].copy().reset_index(drop=True)
    x,y=p.transform(nodes.longitude,nodes.latitude); xy=np.c_[x,y]
    return valid,xy[valid.from_node_id.to_numpy()],xy[valid.to_node_id.to_numpy()]
def generate_candidates(track,nodes,arcs,radius_m=75.,max_candidates=8,trip_id=None):
    """Use a KD-tree then exact point-to-segment distance; preserve source IDs."""
    if not len(nodes) or not len(arcs): raise ValueError("Road network is empty")
    p=local_projector(nodes.longitude,nodes.latitude); valid,start,end=_geometry(nodes,arcs,p); mid=(start+end)/2; half=np.linalg.norm(end-start,axis=1)/2; tree=cKDTree(mid); gx,gy=p.transform(track.longitude,track.latitude); rows=[]
    for obs,(x,y) in enumerate(zip(gx,gy)):
        if not np.isfinite(x+y): continue
        ids=np.asarray(tree.query_ball_point((x,y),r=float(radius_m+half.max())),dtype=int)
        if not len(ids): continue
        d,f=point_to_segments_m(np.repeat([[x,y]],len(ids),axis=0),start[ids],end[ids]); ids,d,f=ids[d<=radius_m],d[d<=radius_m],f[d<=radius_m]
        for rank,j in enumerate(np.argsort(d)[:max_candidates]):
            i=ids[j]; rows.append({"trip_id":str(trip_id) if trip_id is not None else None,"observation_id":obs,"candidate_arc_id":int(valid.arc_id.iloc[i]),"candidate_rank":rank,"distance_to_segment_m":float(d[j]),"segment_length_m":float(np.linalg.norm(end[i]-start[i])),"projection_fraction":float(f[j])})
    return pd.DataFrame(rows)
def candidate_diagnostics(candidates,labels):
    n=len(labels); hit=set(candidates.loc[candidates.candidate_arc_id.eq(candidates.label_arc_id),"observation_id"]) if "label_arc_id" in candidates else set()
    observed=set(candidates.observation_id) if not candidates.empty else set()
    return {"observations":n,"matched_candidates":len(hit),"unmatched_observations":n-len(observed),"candidate_recall":len(hit)/n if n else 0.,"candidate_miss_rate":1-len(hit)/n if n else 1.,"mean_candidates_per_observation":len(candidates)/max(len(observed),1)}
