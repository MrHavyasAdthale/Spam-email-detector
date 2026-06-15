# Smart Spam Email Detection using Gmail Integration

## Overview
This beginner‑friendly Python project connects to a user's Gmail account, fetches recent emails, and predicts whether each email is **Spam** or **Not Spam** using a Multinomial Naïve Bayes classifier trained on a TF‑IDF representation of email bodies.

The project includes:
- Gmail OAuth2 authentication (read‑only scope)
- Email fetching and parsing script
- Data preprocessing (lower‑casing, punctuation removal, stop‑word filtering)
- Model training script (`train_model.py`)
- Prediction utilities (`predict.py`)
- A clean Streamlit web interface (`app.py`) with login, fetch, and results table (spam rows highlighted in red)
- Full documentation and step‑by‑step guide for creating Gmail API credentials

## Project Structure
```
Smart_Spam_Email_Detection/
│
├── dataset/
│   └── spam.csv               # Example dataset (you can replace with your own)
├── models/
│   ├── model.pkl              # Trained Naïve Bayes model (generated after training)
│   └── vectorizer.pkl         # TF‑IDF vectorizer (generated after training)
├── train_model.py              # Training script
├── gmail_fetch.py              # Gmail API authentication & fetching
├── predict.py                  # Prediction helper
├── app.py                      # Streamlit UI
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── .gitignore                 # Ignored files
```

## 📧 Gmail API Credential Setup
1. Open the **Google Cloud Console**: https://console.cloud.google.com/
2. Create a new project or select an existing one.
3. Navigate to **APIs & Services → Library** and enable the **Gmail API**.
4. Go to **APIs & Services → Credentials** → **Create Credentials → OAuth client ID**.
5. Choose **Desktop app** as the application type and give it a name (e.g., *Smart Spam Detector*).
6. Click **Create**. In the dialog, click **Download JSON** – this file is `credentials.json`.
7. Place `credentials.json` in the project root (`Smart_Spam_Email_Detection/`). **Do NOT commit this file** to version control.
8. The first run of the app will open a browser window asking you to grant read‑only Gmail access. After consent, a `token.json` file is saved locally for subsequent runs.

## 🛠️ Installation
```bash
# Clone the repo (or copy the folder) and navigate into it
cd Smart_Spam_Email_Detection

# Create a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# or source venv/bin/activate on macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## 📚 Usage
### 1️⃣ Train the Model
```bash
python train_model.py --data_path dataset/spam.csv
```
The script will create `models/model.pkl` and `models/vectorizer.pkl`.

### 2️⃣ Run the Streamlit App
```bash
streamlit run app.py
```
- Click **Login with Google** to authenticate.
- Click **Fetch Emails** to retrieve the latest 20 messages from your inbox.
- The table will display **Sender**, **Subject**, **Prediction**, and **Confidence**. Spam rows appear in red.

## 🚀 Future Enhancements (Optional)
- **Auto‑label Spam**: Move detected spam messages to a custom Gmail label called *AI Spam* using the Gmail `modify` endpoint.
- **Statistics Dashboard**: Show total emails processed, spam count, and spam percentage (pie chart).
- **Model Retraining Pipeline**: Periodically retrain the model with newly labeled data.

## 📖 References
- [Google Gmail API Python Quickstart](https://developers.google.com/gmail/api/quickstart/python)
- [Scikit‑learn Naïve Bayes Documentation](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [Streamlit Documentation](https://docs.streamlit.io/)

---
*Happy coding and enjoy building your AI‑powered spam detector!*

## Backend / Server Configuration
If the Gmail integration is called from a backend process instead of the local Streamlit desktop flow, you can provide OAuth data through environment variables:

```bash
GMAIL_CREDENTIALS_JSON='{"installed": ...}'
GMAIL_TOKEN_JSON='{"token": ...}'
GMAIL_OAUTH_INTERACTIVE=0
GMAIL_OAUTH_OPEN_BROWSER=0
```

`GMAIL_CREDENTIALS_JSON` replaces `credentials.json`, and `GMAIL_TOKEN_JSON` replaces `token.json`. Backend code can call `scan_inbox()` from `gmail_fetch.py`; it will fail fast when OAuth is not configured instead of trying to open a browser from the server process.
