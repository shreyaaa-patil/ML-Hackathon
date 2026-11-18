"""Leakage-free candidate features and validated reference-label association."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from src.data.geo_utils import haversine_m,bearing_deg,angular_difference_deg,local_projector,point_to_segments_m
FEATURE_COLUMNS=["longitude","latitude","timestamp","gps_interval_s","gps_speed_mps","gps_heading_deg","gps_step_distance_m","distance_to_segment_m","segment_length_m","road_bearing_deg","heading_difference_deg","distance_to_start_m","distance_to_end_m","candidate_rank","incoming_degree","outgoing_degree","is_self_loop"]
def _arc_geometry(nodes,arcs):
    p=local_projector(nodes.longitude,nodes.latitude); x,y=p.transform(nodes.longitude,nodes.latitude); xy=np.c_[x,y]; valid=arcs.set_index("arc_id"); valid=valid[valid.from_node_id.between(0,len(nodes)-1)&valid.to_node_id.between(0,len(nodes)-1)]
    return p,xy,valid,xy[valid.from_node_id.to_numpy()],xy[valid.to_node_id.to_numpy()]
def align_route_labels(track,nodes,arcs,route,max_label_distance_m=100.):
    """Nearest reference-route arc with a nondecreasing chronological occurrence.
    Far or unprogressable observations are retained as unmatched, never fabricated.
    """
    p,xy,valid,start,end=_arc_geometry(nodes,arcs); route_ids=route.arc_id.to_numpy(); unique=np.unique(route_ids[np.isin(route_ids,valid.index)])
    if not len(unique): return pd.DataFrame(columns=["observation_id","label_arc_id","label_distance_m","route_index","label_status"])
    pos={a:np.flatnonzero(route_ids==a) for a in unique}; ai=np.array([valid.index.get_loc(a) for a in unique]); tree=cKDTree((start[ai]+end[ai])/2); gx,gy=p.transform(track.longitude,track.latitude); current=0; rows=[]
    for obs,(x,y) in enumerate(zip(gx,gy)):
        _,near=tree.query((x,y),k=min(8,len(unique))); choices=[]
        for j in np.atleast_1d(near):
            arc=int(unique[int(j)]); forward=pos[arc][pos[arc]>=current]
            if len(forward):
                k=valid.index.get_loc(arc); d,_=point_to_segments_m(np.array([[x,y]]),start[[k]],end[[k]]); choices.append((float(d[0]),int(forward[0]),arc))
        if not choices: rows.append({"observation_id":obs,"label_arc_id":np.nan,"label_distance_m":np.nan,"route_index":np.nan,"label_status":"unmatched_progression"}); continue
        d,ri,arc=min(choices); ok=d<=max_label_distance_m
        if ok: current=ri
        rows.append({"observation_id":obs,"label_arc_id":arc if ok else np.nan,"label_distance_m":d,"route_index":ri,"label_status":"matched" if ok else "unmatched_distance"})
    return pd.DataFrame(rows)
def extract_features(track,nodes,arcs,candidates):
    if candidates.empty: return candidates.copy()
    t=track.reset_index(drop=True); dt=t.timestamp.diff(); step=haversine_m(t.longitude.shift(),t.latitude.shift(),t.longitude,t.latitude); speed=step/dt.replace(0,np.nan); heading=bearing_deg(t.longitude.shift(),t.latitude.shift(),t.longitude,t.latitude)
    p,xy,valid,start,end=_arc_geometry(nodes,arcs); lookup=valid.reindex(candidates.candidate_arc_id.to_numpy()); si=np.array([valid.index.get_loc(i) for i in candidates.candidate_arc_id]); obs=candidates.observation_id.to_numpy(); gx,gy=p.transform(t.longitude.iloc[obs],t.latitude.iloc[obs]); _,frac=point_to_segments_m(np.c_[gx,gy],start[si],end[si]); length=candidates.segment_length_m.to_numpy()
    rb=bearing_deg(nodes.longitude.iloc[lookup.from_node_id.to_numpy()].to_numpy(),nodes.latitude.iloc[lookup.from_node_id.to_numpy()].to_numpy(),nodes.longitude.iloc[lookup.to_node_id.to_numpy()].to_numpy(),nodes.latitude.iloc[lookup.to_node_id.to_numpy()].to_numpy()); indeg=arcs.groupby("to_node_id").size(); outdeg=arcs.groupby("from_node_id").size(); result=candidates.copy()
    vals={"longitude":t.longitude.iloc[obs],"latitude":t.latitude.iloc[obs],"timestamp":t.timestamp.iloc[obs],"gps_interval_s":dt.iloc[obs],"gps_speed_mps":speed.iloc[obs],"gps_heading_deg":heading.iloc[obs],"gps_step_distance_m":step.iloc[obs],"road_bearing_deg":rb,"heading_difference_deg":angular_difference_deg(heading.iloc[obs],rb),"distance_to_start_m":frac*length,"distance_to_end_m":(1-frac)*length,"incoming_degree":lookup.from_node_id.map(indeg).fillna(0).to_numpy(),"outgoing_degree":lookup.to_node_id.map(outdeg).fillna(0).to_numpy(),"is_self_loop":(lookup.from_node_id.to_numpy()==lookup.to_node_id.to_numpy()).astype(int)}
    for col,v in vals.items(): result[col]=np.asarray(v)
    return result.replace([np.inf,-np.inf],np.nan)
