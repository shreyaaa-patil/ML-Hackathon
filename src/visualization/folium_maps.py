
import folium
from folium import FeatureGroup
from pathlib import Path


def create_trip_map(
    track,
    nodes,
    arcs,
    route_arc_ids,
    output_path,
    tiles="CartoDB positron"
):
    """
    Generate an interactive PathSnap map.

    Layers:
        1. Original GPS observations
        2. Road network
        3. Reference route

    tiles:
        "CartoDB positron" - Light basemap (recommended)
        "CartoDB dark_matter" - Dark basemap
        None - No external basemap
    """

    # --------------------------------------------------
    # 1. Validate input data
    # --------------------------------------------------

    if track.empty:
        raise ValueError("GPS track is empty.")

    required_track = {"latitude", "longitude"}
    required_nodes = {"latitude", "longitude"}
    required_arcs = {"from_node_id", "to_node_id", "arc_id"}

    if not required_track.issubset(track.columns):
        raise ValueError("Track must contain latitude and longitude.")

    if not required_nodes.issubset(nodes.columns):
        raise ValueError("Nodes must contain latitude and longitude.")

    if not required_arcs.issubset(arcs.columns):
        raise ValueError("Arcs must contain from_node_id, to_node_id and arc_id.")

    # Remove invalid GPS coordinates for visualization only
    valid_track = track.dropna(subset=["latitude", "longitude"])

    if valid_track.empty:
        raise ValueError("No valid GPS coordinates available.")

    # --------------------------------------------------
    # 2. Calculate map centre
    # --------------------------------------------------

    center = [
        float(valid_track.latitude.mean()),
        float(valid_track.longitude.mean())
    ]

    # --------------------------------------------------
    # 3. Initialize map
    # --------------------------------------------------

    m = folium.Map(
        location=center,
        zoom_start=15,
        control_scale=True,
        tiles=tiles
    )

    # --------------------------------------------------
    # 4. Original GPS observations
    # --------------------------------------------------

    gps = FeatureGroup(
        name="Original GPS observations",
        show=True
    )

    gps_coordinates = []

    for row in valid_track.itertuples():

        lat = float(row.latitude)
        lon = float(row.longitude)

        gps_coordinates.append([lat, lon])

        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color="blue",
            weight=1,
            fill=True,
            fill_color="blue",
            fill_opacity=0.8,
            tooltip="Original GPS observation"
        ).add_to(gps)

    # Connect consecutive GPS observations
    if len(gps_coordinates) > 1:
        folium.PolyLine(
            gps_coordinates,
            color="blue",
            weight=2,
            opacity=0.6,
            tooltip="GPS trajectory"
        ).add_to(gps)

    gps.add_to(m)

    # --------------------------------------------------
    # 5. Create node lookup
    # --------------------------------------------------

    # Use actual node IDs if available.
    # Otherwise, use row positions as IDs.

    if "node_id" in nodes.columns:
        node_lookup = nodes.set_index("node_id")
    else:
        node_lookup = nodes.copy()
        node_lookup.index = range(len(node_lookup))

    # --------------------------------------------------
    # 6. Road network layer
    # --------------------------------------------------

    net = FeatureGroup(
        name="Road network",
        show=False
    )

    for arc in arcs.itertuples():

        from_id = arc.from_node_id
        to_id = arc.to_node_id

        if from_id not in node_lookup.index:
            continue

        if to_id not in node_lookup.index:
            continue

        p1 = node_lookup.loc[from_id]
        p2 = node_lookup.loc[to_id]

        if getattr(p1, "latitude", None) is None:
            continue

        coordinates = [
            [float(p1.latitude), float(p1.longitude)],
            [float(p2.latitude), float(p2.longitude)]
        ]

        folium.PolyLine(
            coordinates,
            color="gray",
            weight=2,
            opacity=0.45
        ).add_to(net)

    net.add_to(m)

    # --------------------------------------------------
    # 7. Reference route layer
    # --------------------------------------------------

    ref = FeatureGroup(
        name="Reference route",
        show=True
    )

    arc_lookup = arcs.set_index("arc_id")

    route_coordinates = []

    for arc_id in route_arc_ids:

        if arc_id not in arc_lookup.index:
            continue

        arc = arc_lookup.loc[arc_id]

        # Skip ambiguous duplicate arc IDs
        if hasattr(arc, "ndim") and arc.ndim != 1:
            continue

        from_id = arc.from_node_id
        to_id = arc.to_node_id

        if from_id not in node_lookup.index:
            continue

        if to_id not in node_lookup.index:
            continue

        p1 = node_lookup.loc[from_id]
        p2 = node_lookup.loc[to_id]

        segment = [
            [float(p1.latitude), float(p1.longitude)],
            [float(p2.latitude), float(p2.longitude)]
        ]

        folium.PolyLine(
            segment,
            color="green",
            weight=5,
            opacity=0.9,
            tooltip=f"Reference arc: {arc_id}"
        ).add_to(ref)

        route_coordinates.extend(segment)

    ref.add_to(m)

    # --------------------------------------------------
    # 8. Layer controls
    # --------------------------------------------------

    folium.LayerControl(
        collapsed=False
    ).add_to(m)

    # --------------------------------------------------
    # 9. Fit map to GPS trajectory
    # --------------------------------------------------

    if gps_coordinates:
        m.fit_bounds(gps_coordinates)

    # --------------------------------------------------
    # 10. Save map
    # --------------------------------------------------

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    m.save(str(output_path))

    print(f"Map successfully saved: {output_path}")

    return m