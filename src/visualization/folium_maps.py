
import folium
from folium import FeatureGroup
from pathlib import Path

def create_trip_map(track, nodes, arcs, route_arc_ids, output_path):
    center=[float(track.latitude.mean()),float(track.longitude.mean())]
    m=folium.Map(location=center,zoom_start=15,control_scale=True)
    gps=FeatureGroup(name="Original GPS observations",show=True)
    for row in track.itertuples():
        folium.CircleMarker([row.latitude,row.longitude],radius=2,color="blue",fill=True).add_to(gps)
    gps.add_to(m)
    net=FeatureGroup(name="Road network",show=False)
    for a in arcs.itertuples():
        if 0<=a.from_node_id<len(nodes) and 0<=a.to_node_id<len(nodes):
            p1=nodes.iloc[a.from_node_id]; p2=nodes.iloc[a.to_node_id]
            folium.PolyLine([[p1.latitude,p1.longitude],[p2.latitude,p2.longitude]],
                            color="gray",weight=2,opacity=.5).add_to(net)
    net.add_to(m)
    ref=FeatureGroup(name="Reference route",show=True)
    arc_lookup=arcs.set_index("arc_id")
    for arc_id in route_arc_ids:
        if arc_id in arc_lookup.index:
            a=arc_lookup.loc[arc_id]; p1=nodes.iloc[int(a.from_node_id)]; p2=nodes.iloc[int(a.to_node_id)]
            folium.PolyLine([[p1.latitude,p1.longitude],[p2.latitude,p2.longitude]],
                            color="green",weight=4).add_to(ref)
    ref.add_to(m)
    folium.LayerControl().add_to(m)
    Path(output_path).parent.mkdir(parents=True,exist_ok=True)
    m.save(str(output_path))
    return m
