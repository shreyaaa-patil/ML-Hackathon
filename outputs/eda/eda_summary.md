# PathSnap – Comprehensive Exploratory Data Analysis (EDA) Report
**Course:** UE24CS352A Machine Learning | **Project:** PathSnap Map-Matching & Route Reconstruction<br>
**Authors:** PathSnap Team | **Date:** November 2026

---

## 1. Executive Summary & Purpose
This document provides a rigorous empirical investigation of the PathSnap dataset. The core objective of PathSnap is to accurately reconstruct vehicle routes by matching noisy, discretely sampled GPS trajectory observations onto a physical road network. This exploratory data analysis investigates spatial coordinate validity, temporal sampling fidelity, road network graph connectivity, and reference route properties across all permitted partitions without data leakage.

---

## 2. Dataset Partition Overview & Inventory
The dataset is strictly partitioned into three mutually exclusive sets to ensure absolute evaluation integrity:

| Partition | Trip Count | Observation Count | Purpose | Access Restrictions |
|---|---:|---:|---|---|
| **Training** | 80 trips | 196,412 | Model fitting, cross-validation, feature engineering | Full access |
| **Student Testing** | 10 trips | 31,165 | Development-stage model evaluation and comparison | Unlabeled feature evaluation |
| **Evaluator Testing** | 10 trips | Reserved | Final live demonstration and scoring | **Strictly held out server-side** |
| **Total Permitted** | **90 trips** | **227,577** | Exploration & Development | Zero inter-partition leakage |

### Key Integrity Rules Verified:
1. **Group Isolation:** Individual vehicle trajectories are preserved as discrete groups. No observations from any trip are split across partitions.
2. **Evaluator Isolation:** Evaluator-testing data and ground-truth reference routes remain untouched on the server and are never inspected during development.

---

## 3. Raw File Formats and Structure
Each trip is stored in an independent directory containing four whitespace-separated, headerless data files:

* **`.track` (GPS Observations):** Contains 3 whitespace-separated columns:
  * Column 0: `longitude` (angular degrees, $-180^\circ$ to $+180^\circ$)
  * Column 1: `latitude` (angular degrees, $-90^\circ$ to $+90^\circ$)
  * Column 2: `timestamp` (continuous elapsed seconds, nominally 1 Hz)
* **`.nodes` (Road Junctions & Vertices):** Contains 2 whitespace-separated columns:
  * Column 0: `longitude`
  * Column 1: `latitude`
  * Row index ($0$-based) represents the unique `node_id`.
* **`.arcs` (Directed Road Segments):** Contains 2 integer columns:
  * Column 0: `from_node_id`
  * Column 1: `to_node_id`
  * Row index ($0$-based) serves as the unique `arc_id`.
* **`.route` (Reference Ground Truth Sequence):** Contains 1 integer column:
  * Chronological sequence of traversed `arc_id` values. Note: Number of route entries does not equal the number of GPS observations.

---

## 4. Geospatial & Geodesic Considerations
* **Angular Coordinates:** Longitude and latitude represent angles on an ellipsoidal Earth.
* **Rejection of Euclidean Distance:** Ordinary Euclidean distance ($\sqrt{\Delta x^2 + \Delta y^2}$) in raw degrees is mathematically invalid. Because a degree of longitude shrinks proportionally with $\cos(\text{latitude})$, Euclidean distance introduces severe non-linear spatial distortion.
* **Geodesic Implementation:** All physical distance computations, candidate radiuses, and perpendicular offsets are computed in **metres** using the **Haversine formula** (mean Earth radius $R = 6,371,000\text{ m}$) or projected local coordinate transformations.

---

## 5. Temporal Sampling Fidelity & Gap Analysis
* **Nominal Frequency:** GPS trajectories are recorded at nominally 1 Hz.
* **Empirical 1 Hz Proportion:**
  * Median sampling interval across all 90 permitted trips is exactly **$1.0\text{ s}$**.
  * Approximately **97.2%** of all consecutive timestamps exhibit $\Delta t = 1.0\text{ s}$.
* **Timestamp Consistency:**
  * Non-increasing or reversed timestamps ($\Delta t \le 0$): **0 occurrences** (timestamps are strictly monotonic).
  * Duplicate timestamps: **0 occurrences**.
* **GPS Anomalies and Large Gaps:**
  * Several trips exhibit substantial GPS dropouts ($\Delta t > 5\text{ s}$) due to urban canyons, underpasses, and physical signal obstruction.
  * *Notable Anomalies:* Trip `00000021` contains a maximum gap of **$2,161\text{ s}$**; Trip `00000011` exhibits a gap of **$909\text{ s}$**; Trip `00000002` contains 78 distinct gaps exceeding $5\text{ s}$.
  * *HMM Design Decision:* Transition probabilities must not over-penalize large temporal gaps; when $\Delta t > 3\text{ s}$, transition distance constraints must be dynamically scaled.

---

## 6. Road Network Graph Topology & Connectivity
Using NetworkX directed graphs ($\mathcal{G} = (\mathcal{V}, \mathcal{E})$) constructed from `.nodes` and `.arcs`:
* **Network Scale:** Varies by geographical extent:
  * Smallest trip network: 732 nodes, 1,510 arcs (Trip `00000021`).
  * Largest trip network: 155,003 nodes, 307,392 arcs (Trip `00000013`).
* **Connected Components:**
  * Road networks consist of multiple weakly connected components (ranging from 2 to 175 components per trip), reflecting bridges, highway cutoffs, and bounding-box border effects.
* **Self-Loops & Duplicate Arcs:**
  * Self-loops (`from_node == to_node`): Extremely rare ($<0.01\%$).
  * Duplicate directed arcs: Handled during candidate indexing to avoid redundant emission states.

---

## 7. Data Quality & Missing Value Summary
* **Invalid Coordinates:** **0** (all coordinates lie within plausible geographical bounds).
* **Missing Values:** **0** nulls across `.track`, `.nodes`, `.arcs`, and `.route`.
* **Duplicate Observations:** **0** duplicate GPS records.
* **Candidate Generation Diagnostic:**
  * Spatial query radius: $75\text{ m}$ (max 6 candidates per observation).
  * **Candidate Miss Rate:** Ground-truth arc is captured within the candidate set for **$\approx 98.4\%$** of all valid observations.

---

## 8. Summary of Assumptions Influencing Map-Matching Design
1. **Sensor Noise:** GPS points deviate from true road centerlines by up to $15\text{–}30\text{ m}$ in dense areas; perpendicular distance alone is insufficient to distinguish parallel roadways.
2. **Heading Alignment:** The directional difference between vehicle instantaneous velocity vector and road arc bearing is a crucial discriminator.
3. **Sequential Regularity:** Consecutive GPS points must map to topologically reachable road segments. Independent classifier predictions cause unphysical route discontinuities, proving the strict necessity of an **HMM / Log-Viterbi decoder**.
