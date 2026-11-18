From the repository root, install requirements.txt and frontend/requirements.txt.
Set PATHSNAP_DATA_ROOT to the parent of student_test and evaluator_test on the
API laptop. Start python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000.
On the UI laptop run python -m streamlit run frontend/app.py. Enter the server
LAN URL under API connection. Select a dataset and trip to load its map.
Evaluator prediction requires artifacts/tuned_model.joblib. Use student trips
during development and the evaluator partition during final evaluation.
