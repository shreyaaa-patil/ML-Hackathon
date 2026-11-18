import json
import os
from urllib.parse import quote

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="PathSnap | Trip maps", layout="wide")


def api(method, path, **kwargs):
    response = requests.request(method, server + path, timeout=(5, 180), **kwargs)
    if not response.ok:
        try:
            detail = response.json()["detail"]
        except (ValueError, KeyError, TypeError):
            detail = response.reason
        raise ValueError(f"API returned {response.status_code}: {detail}")
    return response.json()


def draw_map(data, prediction):
    gps, nodes, arcs = data["gps"], data["nodes"], data["arcs"]
    points = gps or nodes
    view = folium.Map(location=points[0] if points else [0, 0], zoom_start=2,
                      tiles="OpenStreetMap", prefer_canvas=True, control_scale=True)
    lines = [[nodes[start], nodes[end]] for start, end in arcs]
    if lines:
        roads = folium.FeatureGroup(name="Road network").add_to(view)
        folium.PolyLine(lines, color="#687783", opacity=0.55, weight=2).add_to(roads)
    observations = folium.FeatureGroup(name="GPS observations").add_to(view)
    for point in gps:
        folium.CircleMarker(point, radius=3, color="#285bcc", weight=1,
                            fill=True, fill_opacity=0.85).add_to(observations)
    if prediction and prediction["route_arc_ids"]:
        layer = folium.FeatureGroup(name="Predicted route").add_to(view)
        route = [lines[index] for index in prediction["route_arc_ids"]]
        folium.PolyLine(route, color="#00836a", weight=5).add_to(layer)
    if points:
        view.fit_bounds(points, max_zoom=17)
    folium.LayerControl(collapsed=False).add_to(view)
    return view


st.title("PathSnap")
settings, workspace = st.columns([1, 2.7], gap="large")
with settings:
    dataset = st.radio("Dataset", ["Student", "Evaluator"], horizontal=True).lower()
    with st.expander("API connection"):
        server = st.text_input("Server URL", os.getenv("PATHSNAP_API_URL", "http://localhost:8000")).strip().rstrip("/")
        refresh = st.button("Reload trips")
    source = (server, dataset)
    state = st.session_state
    if source != state.get("source") or refresh:
        state.source = source
        state.ids = []
        state.geometry = None
        state.prediction = None
        state.trip = None
        state.problem = None
        try:
            state.ids = api("GET", f"/datasets/{dataset}/trips")["trip_ids"]
        except (requests.RequestException, ValueError, KeyError) as error:
            state.problem = str(error)
    if state.problem:
        st.error(state.problem)
    elif dataset == "evaluator" and len(state.ids) != 10:
        st.warning(f"Received {len(state.ids)} reserved trips. Final evaluation expects 10.")
    trip = st.selectbox("Trip", state.ids, disabled=not state.ids)
    if trip is not None and trip != state.trip:
        state.geometry = None
        state.prediction = None
        try:
            state.geometry = api("GET", f"/datasets/{dataset}/trips/{quote(trip, safe='')}/map")
            state.trip = trip
        except (requests.RequestException, ValueError) as error:
            state.trip = None
            st.error(str(error))
    if dataset == "evaluator" and st.button("Reconstruct route", type="primary", disabled=state.geometry is None):
        state.prediction = None
        try:
            with st.spinner("Reconstructing route…"):
                result = api("POST", "/predict", json={"trip_id": trip})
                if result["trip_id"] != trip or any(
                    index < 0 or index >= len(state.geometry["arcs"]) for index in result["route_arc_ids"]
                ):
                    raise ValueError("Prediction does not match this trip’s road network.")
                state.prediction = result
        except (requests.RequestException, ValueError, KeyError) as error:
            st.error(str(error))
    if state.geometry:
        st.caption(f'{len(state.geometry["gps"]):,} GPS observations · {len(state.geometry["arcs"]):,} road segments')
    if state.prediction:
        result = state.prediction
        accuracy = result["top1_accuracy"]
        st.metric("Top-1 accuracy", "Unavailable" if accuracy is None else f"{accuracy:.2%}")
        st.metric("Processing time", f'{result["elapsed_seconds"]:.2f} s')
        st.caption(f'{result["matched_observations"]} of {result["total_observations"]} observations scored')
        st.download_button("Download results", data=json.dumps(result, indent=2),
                           file_name="pathsnap_results.json", mime="application/json")

with workspace:
    st.subheader("Trip map")
    if state.geometry is None:
        st.info("Load a trip from the API to see its GPS observations and road network.")
    else:
        view = draw_map(state.geometry, state.prediction)
        st_folium(view, height=550, use_container_width=True, returned_objects=[],
                  key=f"{server}/{dataset}/{trip}/{bool(state.prediction)}")
