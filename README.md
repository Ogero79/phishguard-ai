# PhishGuard AI
Phishing URL detection system using machine learning.

## Team
| Member | Role |
|---|---|
| Brian Ogero | Team Lead |
| Irene Kambo | Data Engineer |
| Ian Kipkurui| AI Engineer |
| Larry Mjomba | Backend |
| Agnes Mugure| Frontend |

## Project Structure
- data/raw — original downloaded dataset (not tracked by Git)
- data/processed — cleaned and split dataset files
- notebooks — Jupyter notebooks for each project stage
- models — trained model and scaler files
- plots — all saved visualisation images
- app.py — Streamlit web application
- utils.py — feature extraction and prediction pipeline

## Setup
1. Clone the repository
2. Install dependencies: pip install -r requirements.txt
3. Download the          dfsd   dataset from Kaggle and place it in data/raw/
4. Run notebooks in order starting from 02_preprocessing.ipynb

## Run the App Locally
streamlit run app.py

## Live Deployment
https://phishguard-ai-2026.streamlit.app/

## Dataset
PhiUSIIL Phishing URL Dataset — Kaggle
