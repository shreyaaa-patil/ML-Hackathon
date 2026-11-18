# FastAPI evaluator server

Keep the evaluator partition on the server laptop. Set its path and start the
API from the repository root in PowerShell:

```powershell
$env:PATHSNAP_DATA_ROOT = "D:\ML Hackathon\Student folder"
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

The data root must contain `evaluator_test/`, and the server must have
`artifacts/tuned_model.joblib` from the Optuna optimization stage. Uvicorn logs
requests, accuracy, and timing in the server terminal.

- `GET /trips` returns evaluator trip IDs only.
- `POST /predict` accepts `{"trip_id":"00000000"}` and returns aggregate
  top-1 accuracy plus the predicted route arc IDs. Reference labels stay on the
  server and are never included in the JSON response.

The interactive API docs are available at `http://localhost:8000/docs` on the
server laptop. The teammate's laptop should call `http://<server-LAN-IP>:8000`.
