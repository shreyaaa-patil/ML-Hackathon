"""Actual-data inventory, quality analysis, plots and Folium map generation."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from src.data.loader import discover_trip_dirs,load_trip
from src.data.geo_utils import haversine_m
from src.visualization.folium_maps import create_trip_map
def analyse_trip(trip,partition):
    t,n,a,r=trip["track"],trip["nodes"],trip["arcs"],trip["route"];dt=t.timestamp.diff();dist=haversine_m(t.longitude.shift(),t.latitude.shift(),t.longitude,t.latitude);g=nx.DiGraph();g.add_nodes_from(n.node_id);valid=a[a.from_node_id.between(0,len(n)-1)&a.to_node_id.between(0,len(n)-1)];g.add_edges_from(valid[["from_node_id","to_node_id"]].itertuples(index=False,name=None))
    return {"partition":partition,"trip_id":trip["trip_id"],"gps_observations":len(t),"nodes":len(n),"arcs":len(a),"route_entries":len(r),"track_bytes":trip["files"]["track"].stat().st_size,"nodes_bytes":trip["files"]["nodes"].stat().st_size,"arcs_bytes":trip["files"]["arcs"].stat().st_size,"route_bytes":trip["files"]["route"].stat().st_size,"missing_files":0,"invalid_coordinates":trip["validation"]["invalid_coordinates"],"missing_coordinates":trip["validation"]["missing_track_values"],"duplicate_observations":trip["validation"]["duplicate_observations"],"non_increasing_timestamps":trip["validation"]["non_increasing_timestamps"],"median_interval_s":dt.median(),"max_interval_s":dt.max(),"gps_gaps_over_5s":(dt>5).sum(),"trajectory_length_m":dist.sum(),"duration_s":t.timestamp.max()-t.timestamp.min(),"median_speed_mps":(dist/dt.replace(0,np.nan)).median(),"duplicate_arcs":a.duplicated(["from_node_id","to_node_id"]).sum(),"self_loops":(a.from_node_id==a.to_node_id).sum(),"isolated_nodes":sum(d==0 for _,d in g.degree()),"connected_components":nx.number_weakly_connected_components(g),"invalid_arc_references":trip["validation"]["invalid_arc_nodes"],"invalid_route_arc_ids":trip["validation"]["invalid_route_arcs"],"route_duplicate_consecutive":(r.arc_id.diff()==0).sum()}
def _plot(df,col,path,title):
    fig,ax=plt.subplots(figsize=(7,4));ax.hist(df[col].dropna(),bins=40,color="#3478bf");ax.set(title=title,xlabel=col,ylabel="Trips");fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)
def run_eda(data_root,output_dir,partitions=("train","student_test")):
    out=Path(output_dir);plots=out/"plots";plots.mkdir(parents=True,exist_ok=True);maps=out.parent/"maps";maps.mkdir(parents=True,exist_ok=True);rows=[];first=None
    for part in partitions:
        for d in discover_trip_dirs(Path(data_root)/part):
            trip=load_trip(d);rows.append(analyse_trip(trip,part));first=first or trip
    inv=pd.DataFrame(rows);inv.to_csv(out/"dataset_inventory.csv",index=False)
    for c,title in [("gps_observations","GPS observations per trip"),("median_interval_s","GPS sampling intervals"),("trajectory_length_m","GPS trajectory lengths"),("duration_s","Trip durations"),("duplicate_observations","Duplicate observations"),("connected_components","Network connectivity")]:_plot(inv,c,plots/f"{c}.png",title)
    if first:
        t=first["track"];fig,ax=plt.subplots(figsize=(7,6));ax.plot(t.longitude,t.latitude,".",ms=1);ax.set(title="GPS trajectory",xlabel="longitude",ylabel="latitude");fig.tight_layout();fig.savefig(plots/"gps_trajectory.png",dpi=140);plt.close(fig);fig,ax=plt.subplots(figsize=(7,5));ax.scatter(t.longitude,t.latitude,s=2);ax.set(title="Longitude-latitude distribution",xlabel="longitude",ylabel="latitude");fig.tight_layout();fig.savefig(plots/"lon_lat_distribution.png",dpi=140);plt.close(fig);create_trip_map(t,first["nodes"],first["arcs"],first["route"].arc_id,maps/f"{first['trip_id']}_eda_map.html")
    (out/"eda_summary.md").write_text("# PathSnap EDA summary\n\nAnalysed %d permitted trips. GPS observations: %d. Invalid coordinates: %d. Distances use geodesic metres and local projected geometry."%(len(inv),int(inv.gps_observations.sum()),int(inv.invalid_coordinates.sum())))
    return inv
