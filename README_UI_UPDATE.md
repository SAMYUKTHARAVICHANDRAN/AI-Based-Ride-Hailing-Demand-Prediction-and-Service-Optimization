# NYC Taxi Intelligence Studio — Latest notebooks + redesigned UI

This smaller package contains the dark-themed Streamlit `app.py`, four notebooks copied **unchanged** from the latest supplied `notebooks(2).zip`, existing training/export script, and Streamlit dependencies. The Parquet dataset and generated model artifact are **not included** to keep downloading easy. Keep your existing `D:\ml_review2_ui\data` directory.

**Notebook verification:** Regression 32 Markdown cells, Classification A 4, Classification B 0, Clustering 0. The four main notebook files in this upload have identical SHA-256 hashes to the prior `notebooks(1).zip` (although the ZIP itself may differ). Do not assume later unsaved local edits were captured.

## Install on Windows (PowerShell)

Back up `D:\ml_review2_ui\app.py` and the four `.ipynb` files before copying. Extract this archive and copy `app.py`, `train_export.py`, `requirements_UI_LOCAL.txt`, `README_UI_UPDATE.md` and the four `notebooks/*.ipynb` files into `D:\ml_review2_ui`, retaining the relative paths. **Do not delete the existing `data/` directory.**

```powershell
cd D:\ml_review2_ui
python -m pip install -r requirements_UI_LOCAL.txt
python -m streamlit run app.py
```

Unlike the old app, this version reads the four `.ipynb` files directly from `notebooks/`, so added Markdown cells and newly saved outputs appear after Streamlit refresh; you do **not** need to rebuild `notebooks.zip`. It does not execute notebook code on the dashboard. The Live Prediction page needs a genuine trained model: `python train_export.py`, using the local NYC Taxi Parquet dataset. Predictions are not available unless export succeeds.

## GitHub update (do not push large data)

The previous work was merged into `main`. From PowerShell in `D:\ml_review2_ui`, first run `git status` and `git branch --show-current`. If the current branch is still `feature/streamlit-dashboard`, avoid resetting or force-pushing; create a fresh branch only after fetching/updating from main safely. Stage explicit paths only (`app.py`, the four notebooks, `train_export.py`, and `requirements_UI_LOCAL.txt`), commit, push a feature branch, and create a pull request. Avoid uploading `data/`, `notebooks.zip`, checkpoints, and local model files unintentionally.

## Limitations

The notebooks' saved outputs are historical results. This package is syntax-checked, but end-to-end Streamlit rendering, local training, and notebook reruns have not been tested in your Windows environment. Classification probabilities are uncalibrated and should not be called validated confidence scores.
