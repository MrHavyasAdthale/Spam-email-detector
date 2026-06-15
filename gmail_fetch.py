"""Gmail API integration for fetching and classifying inbox emails.

The module supports the local Streamlit desktop flow and backend-style
configuration through environment variables:

- GMAIL_CREDENTIALS_JSON: full OAuth client JSON content.
- GMAIL_TOKEN_JSON: full authorized user token JSON content.
- GMAIL_OAUTH_INTERACTIVE=0: fail fast instead of starting OAuth.
- GMAIL_OAUTH_OPEN_BROWSER=0: print the consent URL instead of opening a browser.
- GMAIL_OAUTH_PORT: fixed callback port, otherwise Google auth chooses one.
"""

import base64
import html
import json
import os
import re
from pathlib import Path
from typing import Optional, Union

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
BASE_DIR = Path(__file__).resolve().parent
CREDENTIALS_FILE = BASE_DIR / "credentials.json"
TOKEN_FILE = BASE_DIR / "token.json"


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean-like environment flag."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _load_credentials_from_env() -> Optional[Credentials]:
    """Load an OAuth token from GMAIL_TOKEN_JSON for backend deployments."""
    token_json = os.getenv("GMAIL_TOKEN_JSON")
    if not token_json:
        return None
    return Credentials.from_authorized_user_info(json.loads(token_json), SCOPES)


def _load_client_config(credentials_file: Path) -> Optional[dict]:
    """Load OAuth client config from GMAIL_CREDENTIALS_JSON or disk."""
    credentials_json = os.getenv("GMAIL_CREDENTIALS_JSON")
    if credentials_json:
        return json.loads(credentials_json)
    if credentials_file.exists():
        with credentials_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


def authenticate(
    credentials_file: Union[str, os.PathLike] = CREDENTIALS_FILE,
    token_file: Union[str, os.PathLike] = TOKEN_FILE,
    *,
    interactive: Optional[bool] = None,
    open_browser: Optional[bool] = None,
):
    """Authenticate with Gmail and return a Gmail API service object.

    Local app usage can keep ``credentials.json`` in the project root. Backend
    usage can provide ``GMAIL_CREDENTIALS_JSON`` and ``GMAIL_TOKEN_JSON`` so the
    server never depends on relative files or an interactive browser session.
    """
    credentials_path = Path(credentials_file)
    token_path = Path(token_file)
    client_config = _load_client_config(credentials_path)

    if not client_config:
        raise FileNotFoundError(
            f"'{credentials_path}' not found and GMAIL_CREDENTIALS_JSON is not set.\n"
            "Download it from Google Cloud Console -> APIs & Services -> "
            "Credentials -> OAuth 2.0 Client ID -> Download JSON.\n"
            "Place it in the project root directory or provide it through the "
            "GMAIL_CREDENTIALS_JSON environment variable."
        )

    creds = _load_credentials_from_env()

    if not creds and token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            should_interact = (
                _env_flag("GMAIL_OAUTH_INTERACTIVE", True)
                if interactive is None
                else interactive
            )
            if not should_interact:
                raise RuntimeError(
                    "Gmail OAuth token is missing or invalid. Provide a valid "
                    "token.json file or set GMAIL_TOKEN_JSON for backend use."
                )

            flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
            should_open_browser = (
                _env_flag("GMAIL_OAUTH_OPEN_BROWSER", True)
                if open_browser is None
                else open_browser
            )
            port = int(os.getenv("GMAIL_OAUTH_PORT", "0"))
            creds = flow.run_local_server(
                port=port,
                open_browser=should_open_browser,
                prompt="consent",
            )

        if not os.getenv("GMAIL_TOKEN_JSON"):
            token_path.write_text(creds.to_json(), encoding="utf-8")

    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def _decode_base64url(raw: str) -> str:
    """Decode Gmail's base64url body strings."""
    if not raw:
        return ""
    padding = "=" * (-len(raw) % 4)
    return base64.urlsafe_b64decode(raw + padding).decode("utf-8", errors="replace")


def _html_to_text(value: str) -> str:
    """Convert a small HTML email body into readable plain text."""
    value = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", value)
    value = re.sub(r"(?i)<br\s*/?>", "\n", value)
    value = re.sub(r"(?i)</p\s*>", "\n", value)
    value = re.sub(r"(?s)<.*?>", " ", value)
    return html.unescape(re.sub(r"[ \t\r\f\v]+", " ", value)).strip()


def _decode_body(payload: dict) -> str:
    """Recursively extract the best readable body from a Gmail payload."""
    mime_type = payload.get("mimeType", "")
    raw = payload.get("body", {}).get("data", "")
    if raw:
        text = _decode_base64url(raw)
        return _html_to_text(text) if mime_type == "text/html" else text

    plain_text = ""
    html_text = ""
    for part in payload.get("parts", []):
        part_mime_type = part.get("mimeType", "")
        part_raw = part.get("body", {}).get("data", "")

        if part_mime_type == "text/plain" and part_raw:
            plain_text = _decode_base64url(part_raw)
            break
        if part_mime_type == "text/html" and part_raw and not html_text:
            html_text = _html_to_text(_decode_base64url(part_raw))
            continue
        if part_mime_type.startswith("multipart/"):
            nested = _decode_body(part)
            if nested and not plain_text:
                plain_text = nested

    return plain_text or html_text


def _get_header(headers: list[dict], name: str) -> str:
    """Extract a header value by name from a list of Gmail headers."""
    for header in headers:
        if header.get("name", "").lower() == name.lower():
            return header.get("value", "")
    return ""


def fetch_emails(service, max_results: int = 20) -> list[dict]:
    """Fetch recent inbox emails from Gmail."""
    max_results = max(1, min(int(max_results), 100))
    results = (
        service.users()
        .messages()
        .list(userId="me", maxResults=max_results, labelIds=["INBOX"])
        .execute()
    )

    messages = results.get("messages", [])
    if not messages:
        return []

    emails = []
    for msg_meta in messages:
        msg = (
            service.users()
            .messages()
            .get(userId="me", id=msg_meta["id"], format="full")
            .execute()
        )
        payload = msg.get("payload", {})
        headers = payload.get("headers", [])
        body = _decode_body(payload)

        if len(body) > 2000:
            body = body[:2000] + "..."

        emails.append(
            {
                "id": msg.get("id", ""),
                "sender": _get_header(headers, "From"),
                "subject": _get_header(headers, "Subject"),
                "body": body,
                "snippet": msg.get("snippet", ""),
                "internal_date": msg.get("internalDate", ""),
            }
        )

    return emails


def fetch_and_classify(service, max_results: int = 20) -> list[dict]:
    """Fetch inbox emails and append spam predictions in one backend call."""
    from predict import predict_batch

    emails = fetch_emails(service, max_results=max_results)
    texts = [f"{email.get('subject', '')} {email.get('body', '')}" for email in emails]
    predictions = predict_batch(texts)

    for email, (label, confidence) in zip(emails, predictions):
        email["prediction"] = label
        email["confidence"] = confidence

    return emails


def scan_inbox(max_results: int = 20, *, interactive: bool = False) -> list[dict]:
    """Backend-friendly Gmail scanner.

    This is the function a backend route/job should call. It never imports
    Streamlit, and by default it fails fast when OAuth has not already been
    configured instead of trying to open a browser from the server process.
    """
    service = authenticate(interactive=interactive, open_browser=interactive)
    return fetch_and_classify(service, max_results=max_results)
