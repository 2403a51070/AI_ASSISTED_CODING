# Car Price Prediction - Mini Project (MVP)

## Files
- `train.py` : trains model and writes `model.joblib`
- `app.py` : Flask web app (serves UI and /predict)
- `data/generate_data.py` : generate synthetic dataset (creates `data/cars.csv`)
- `templates/index.html` : front-end
- `static/style.css` : styles
- `requirements.txt` : pip dependencies

## Quick start
1. Create & activate venv:
   - mac/linux:
     ```bash
     python -m venv venv
     source venv/bin/activate
     ```
   - windows:
     ```bash
     python -m venv venv
     venv\\Scripts\\activate
     ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
