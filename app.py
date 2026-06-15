"""Streamlit web application for the Smart Spam Email Detector.

Provides two modes:
    1. **Manual Mode** — Paste any text and classify it instantly.
    2. **Gmail Mode** — Authenticate via Google OAuth2, fetch inbox
       emails, and classify each one. Spam rows are highlighted red.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Spam Email Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for premium look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---------- Global ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---------- Header banner ---------- */
    .hero-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .hero-banner h1 {
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
    }
    .hero-banner p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.4rem;
    }

    /* ---------- Stat cards ---------- */
    .stat-card {
        background: linear-gradient(135deg, #1e1e2f, #2a2a40);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    .stat-card h3 {
        font-size: 2rem;
        margin: 0;
        font-weight: 700;
    }
    .stat-card p {
        margin: 0.2rem 0 0;
        font-size: 0.9rem;
        opacity: 0.7;
    }
    .stat-ham h3 { color: #2ecc71; }
    .stat-spam h3 { color: #e74c3c; }
    .stat-total h3 { color: #667eea; }
    .stat-pct h3 { color: #f39c12; }

    /* ---------- Result badge ---------- */
    .badge-spam {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-ham {
        background: linear-gradient(135deg, #2ecc71, #27ae60);
        color: white;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    /* ---------- Table highlight ---------- */
    .spam-row {
        background-color: rgba(231, 76, 60, 0.15) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ Spam Detector")
    st.markdown("---")
    st.markdown(
        "AI-powered email spam detection using **Multinomial Naïve Bayes** "
        "trained on TF-IDF features."
    )
    st.markdown("---")

    mode = st.radio(
        "Choose mode",
        ["✍️ Manual Input", "📧 Gmail Integration"],
        index=0,
        help="Manual mode lets you paste text. Gmail mode fetches your inbox.",
    )

    st.markdown("---")
    st.markdown(
        "**How it works:**\n"
        "1. Text is preprocessed (lowercase, remove punctuation & stopwords)\n"
        "2. Converted to TF-IDF features\n"
        "3. Classified by Naïve Bayes\n"
    )
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit & scikit-learn")


# ---------------------------------------------------------------------------
# Hero banner
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <h1>🛡️ Smart Spam Email Detector</h1>
        <p>Detect spam emails instantly using Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Manual Input Mode
# ---------------------------------------------------------------------------
if mode == "✍️ Manual Input":
    st.markdown("### ✍️ Paste text to classify")

    user_text = st.text_area(
        "Enter email or message text below:",
        height=180,
        placeholder="e.g. Congratulations! You have won a $1000 gift card. Click here to claim now!",
    )

    col_btn, col_space = st.columns([1, 3])
    with col_btn:
        classify_btn = st.button("🔍 Classify", type="primary", use_container_width=True)

    if classify_btn and user_text.strip():
        from predict import predict

        with st.spinner("Analysing…"):
            label, confidence = predict(user_text)

        st.markdown("---")
        st.markdown("### Result")

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            badge_class = "badge-spam" if label == "spam" else "badge-ham"
            emoji = "🚫" if label == "spam" else "✅"
            st.markdown(
                f'<span class="{badge_class}">{emoji} {label.upper()}</span>',
                unsafe_allow_html=True,
            )
        with res_col2:
            st.metric("Confidence", f"{confidence}%")

        # Visual confidence bar
        bar_color = "#e74c3c" if label == "spam" else "#2ecc71"
        st.markdown(
            f"""
            <div style="background:#2a2a40; border-radius:10px; overflow:hidden; height:12px; margin-top:0.5rem;">
                <div style="width:{confidence}%; background:{bar_color}; height:100%; border-radius:10px;
                            transition: width 0.6s ease;"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif classify_btn:
        st.warning("Please enter some text to classify.")


# ---------------------------------------------------------------------------
# Gmail Integration Mode
# ---------------------------------------------------------------------------
elif mode == "📧 Gmail Integration":
    st.markdown("### 📧 Gmail Inbox Scanner")

    import os
    from pathlib import Path

    credentials_path = Path(__file__).resolve().parent / "credentials.json"
    has_credentials = credentials_path.exists() or os.getenv("GMAIL_CREDENTIALS_JSON")

    if not has_credentials:
        st.warning(
            "⚠️ **credentials.json not found.**  \n"
            "To use Gmail integration, follow the steps in the README to "
            "download your OAuth2 credentials from Google Cloud Console and "
            "place `credentials.json` in the project root.",
            icon="🔑",
        )
        st.info(
            "💡 **Tip:** You can still use **Manual Input** mode to classify "
            "any text without Gmail credentials.",
            icon="💡",
        )
    else:
        col_login, col_fetch = st.columns(2)

        with col_login:
            login_btn = st.button("🔐 Login with Google", use_container_width=True)

        with col_fetch:
            max_emails = st.number_input(
                "Emails to fetch", min_value=1, max_value=50, value=20
            )

        if login_btn or st.session_state.get("gmail_service"):
            try:
                from gmail_fetch import authenticate, fetch_and_classify

                if "gmail_service" not in st.session_state:
                    with st.spinner("Authenticating with Google…"):
                        st.session_state["gmail_service"] = authenticate(
                            interactive=True
                        )
                    st.success("✅ Logged in successfully!")

                fetch_btn = st.button(
                    "📥 Fetch Emails", type="primary", use_container_width=True
                )

                if fetch_btn:
                    with st.spinner(f"Fetching {max_emails} emails…"):
                        emails = fetch_and_classify(
                            st.session_state["gmail_service"],
                            max_results=max_emails,
                        )

                    if not emails:
                        st.info("No emails found in your inbox.")
                    else:
                        results = []
                        for em in emails:
                            results.append(
                                {
                                    "Sender": em["sender"],
                                    "Subject": em["subject"],
                                    "Prediction": em["prediction"].upper(),
                                    "Confidence": f"{em['confidence']}%",
                                }
                            )

                        df = pd.DataFrame(results)

                        # Stats
                        total = len(df)
                        spam_count = (df["Prediction"] == "SPAM").sum()
                        ham_count = total - spam_count
                        spam_pct = round(spam_count / total * 100, 1) if total else 0

                        st.markdown("---")
                        st.markdown("### 📊 Statistics")

                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            st.markdown(
                                f'<div class="stat-card stat-total"><h3>{total}</h3><p>Total Emails</p></div>',
                                unsafe_allow_html=True,
                            )
                        with c2:
                            st.markdown(
                                f'<div class="stat-card stat-ham"><h3>{ham_count}</h3><p>Legitimate</p></div>',
                                unsafe_allow_html=True,
                            )
                        with c3:
                            st.markdown(
                                f'<div class="stat-card stat-spam"><h3>{spam_count}</h3><p>Spam</p></div>',
                                unsafe_allow_html=True,
                            )
                        with c4:
                            st.markdown(
                                f'<div class="stat-card stat-pct"><h3>{spam_pct}%</h3><p>Spam Rate</p></div>',
                                unsafe_allow_html=True,
                            )

                        st.markdown("---")
                        st.markdown("### 📋 Results")

                        # Style spam rows red
                        def highlight_spam(row):
                            if row["Prediction"] == "SPAM":
                                return [
                                    "background-color: rgba(231,76,60,0.18); color: #e74c3c; font-weight: 600"
                                ] * len(row)
                            return [""] * len(row)

                        styled_df = df.style.apply(highlight_spam, axis=1)
                        st.dataframe(
                            styled_df,
                            use_container_width=True,
                            height=500,
                        )

            except FileNotFoundError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Error: {e}")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; opacity:0.5; font-size:0.85rem;'>"
    "Smart Spam Email Detector &bull; Powered by scikit-learn &amp; Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
