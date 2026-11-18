# Road graph and HMM route decoder

`graph.py` loads a trip's headerless `.nodes` and `.arcs` files into a directed
NetworkX multigraph. It preserves source `arc_id` values and attaches metric
arc lengths and bearings. `RoadGraph.transition()` finds a shortest directed
connector between two candidate arcs and returns its distance and arc sequence.

`hmm_viterbi.py` combines each classifier probability (emission likelihood)
with directed network reachability and agreement between network travel distance
and GPS displacement (transition likelihood). It uses log probabilities and
Viterbi dynamic programming, then expands the chosen candidate sequence through
intermediate road arcs to produce a continuous directed route.

## Run the student-test route decoder

After features and model artifacts exist, run from the project root:

```powershell
python run_pipeline.py --stage decode --data-root "C:\path\to\Student folder"
```

The decoder uses `artifacts/tuned_model.joblib` when present. Otherwise it uses
the model named in `outputs/model_comparison/model_selection_report.md`.
It writes per-trip observation choices, expanded route arcs, and
`outputs/routes/student_test/decode_summary.csv`. The stage does not load the
evaluator partition.

## Use from Python

```python
from src.routing import decode_trip, load_road_graph

graph = load_road_graph(nodes, arcs)  # loader DataFrames with node_id/arc_id
result = decode_trip(model, track, candidate_features, graph, FEATURE_COLUMNS)
result.observations.to_csv("decoded_observations.csv", index=False)
result.route.to_csv("continuous_route.csv", index=False)
```

`candidate_features` must contain the classifier features, `observation_id`,
`candidate_arc_id`, and optionally `projection_fraction` or
`distance_to_start_m`/`segment_length_m`. The track should have GPS columns and
an index or `observation_id` matching candidate observations. A directed
network break raises a clear `ValueError` instead of returning a disconnected
route.

## HMM details and evaluation

- **Hidden state:** one candidate directed road arc at each observation that
  has at least one generated candidate.
- **Emission:** the fitted classifier's positive-class probability for that
  candidate, normalized across candidates for the observation.
- **Transition:** a directed shortest-path connection must exist. Its log score
  is a Gaussian penalty on the difference between network travel distance and
  straight-line GPS displacement, with uncertainty
  `max(35 m, 0.35 × GPS displacement, 5 m/s × elapsed GPS time)`. Scores are
  normalized across reachable next states before Viterbi decoding.
- **Missing candidates:** observations with no generated candidates do not
  become HMM states. The transition uses the GPS displacement between the
  remaining consecutive candidate observations, so a longer spatial or time
  gap widens the transition uncertainty. The decoder does not invent a
  candidate for a missing observation.
- **Disconnected roads:** if no directed path connects any consecutive state
  pair, decoding raises a `ValueError`; the pipeline records a decode failure.

The `decode` stage reports classifier-only versus HMM Top-1 Accuracy and Macro
F1 on the same successfully decoded student-test trips, alongside matched-label
coverage. It writes paired per-trip metrics to
`outputs/routes/student_test/decode_summary.csv`, a generated comparison to
`route_method_comparison.md`, and plots under `outputs/routes/student_test/plots/`.
Interactive Folium maps compare the GPS track, road network, and reference or
decoded route under `outputs/maps/`. HMM improvement is therefore determined
from the measured student-test results; it is not assumed in advance.
