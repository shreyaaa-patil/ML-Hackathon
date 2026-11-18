"""Validated loaders for the headerless PathSnap raw trip files."""
from pathlib import Path
import numpy as np
import pandas as pd
EXTENSIONS={"track":".track","nodes":".nodes","arcs":".arcs","route":".route"}
def _read_numeric(path,expected_cols):
    path=Path(path)
    if not path.is_file(): raise FileNotFoundError(path)
    df=pd.read_csv(path,sep=r"\s+",header=None,comment="#",engine="python")
    if df.shape[1]!=expected_cols: raise ValueError(f"{path}: expected {expected_cols} columns, found {df.shape[1]}")
    if df.empty: raise ValueError(f"{path}: empty file")
    return df
def load_track(path):
    df=_read_numeric(path,3); df.columns=["longitude","latitude","timestamp"]
    return df.astype(float).assign(observation_id=lambda x:np.arange(len(x),dtype=int))
def load_nodes(path):
    df=_read_numeric(path,2); df.columns=["longitude","latitude"]
    return df.astype(float).assign(node_id=lambda x:np.arange(len(x),dtype=int))
def load_arcs(path):
    df=_read_numeric(path,2); df.columns=["from_node_id","to_node_id"]
    return df.astype("int64").assign(arc_id=lambda x:np.arange(len(x),dtype=int))
def load_route(path):
    df=_read_numeric(path,1); df.columns=["arc_id"]
    return df.astype("int64").assign(sequence_index=lambda x:np.arange(len(x),dtype=int))
def discover_trip_dirs(partition_dir):
    p=Path(partition_dir)
    if not p.is_dir(): raise FileNotFoundError(p)
    return sorted((x for x in p.iterdir() if x.is_dir()),key=lambda x:x.name)
def _single_file(trip_dir,suffix):
    matches=list(Path(trip_dir).glob(f"*{suffix}"))
    if len(matches)!=1: raise FileNotFoundError(f"Expected exactly one {suffix} in {trip_dir}; found {len(matches)}")
    return matches[0]
def validate_trip(track,nodes,arcs,route):
    return {"invalid_coordinates":int((~track.longitude.between(-180,180)|~track.latitude.between(-90,90)).sum()),"missing_track_values":int(track[["longitude","latitude","timestamp"]].isna().any(axis=1).sum()),"duplicate_observations":int(track.duplicated(["longitude","latitude","timestamp"]).sum()),"non_increasing_timestamps":int((track.timestamp.diff().dropna()<=0).sum()),"invalid_arc_nodes":int((~arcs.from_node_id.between(0,len(nodes)-1)|~arcs.to_node_id.between(0,len(nodes)-1)).sum()),"invalid_route_arcs":int((~route.arc_id.between(0,len(arcs)-1)).sum())}
def load_trip(trip_dir):
    p=Path(trip_dir); files={k:_single_file(p,e) for k,e in EXTENSIONS.items()}; trip={"trip_id":p.name,"track":load_track(files["track"]),"nodes":load_nodes(files["nodes"]),"arcs":load_arcs(files["arcs"]),"route":load_route(files["route"]),"files":files}
    trip["validation"]=validate_trip(trip["track"],trip["nodes"],trip["arcs"],trip["route"]); return trip
