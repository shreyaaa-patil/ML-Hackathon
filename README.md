# PathSnap

PathSnap matches GPS observations to candidate road segments, compares five classifiers, tunes the selected model with Optuna, and reconstructs continuous routes over a directed road network.

## Data

Keep raw trip data outside Git. Set `--data-root` to a directory containing `train/` and `student_test/`, with one `.track`, `.nodes`, `.arcs`, and `.route` file per trip. Evaluator data is not read by these stages.

## Run

Install dependencies with `python -m pip install -r requirements.txt`, then run from the repository root:

```powershell
python run_pipeline.py --stage eda --data-root "C:\path\to\Student folder"
python run_pipeline.py --stage features --data-root "C:\path\to\Student folder"
python run_pipeline.py --stage train --data-root "C:\path\to\Student folder"
python run_pipeline.py --stage optimize --data-root "C:\path\to\Student folder"
python run_pipeline.py --stage decode --data-root "C:\path\to\Student folder"
```

Optimization writes `outputs/optimization/tuned_vs_untuned.csv` and `artifacts/tuned_model.joblib`. Route decoding uses the tuned artifact when present and writes per-trip candidate choices and expanded continuous route arcs under `outputs/routes/student_test/`.
