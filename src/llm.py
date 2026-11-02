from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import logging
import os
from typing import Any
import json
import subprocess
from threading import Lock
from google.auth import default
import google.auth.transport.requests
from datetime import datetime, timedelta, timezone
_vertex_request = google.auth.transport.requests.Request()
load_dotenv()

_vertex_credentials_lock = Lock()
_vertex_credentials = None

# Suppress the automatic function calling warning from google-genai
logging.getLogger("google_genai").setLevel(logging.ERROR)

def _vertex_token_needs_refresh(credentials):
    if not credentials.token:
        return True

    expiry = getattr(credentials, "expiry", None)
    if expiry is None:
        return True

    if expiry.tzinfo is None:
        now = datetime.utcnow()
    else:
        now = datetime.now(timezone.utc)

    return expiry <= now + timedelta(minutes=5)

def _get_vertex_access_token():
    global _vertex_credentials

    with _vertex_credentials_lock:
        if _vertex_credentials is None:
            _vertex_credentials,_ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])

        if _vertex_token_needs_refresh(_vertex_credentials):
            _vertex_credentials.refresh(_vertex_request)

        return _vertex_credentials.token

api_key = _get_vertex_access_token()

client = OpenAI(
    base_url=os.environ["GEMINI_BASE_URL"],
    api_key=api_key
)
# client = OpenAI(
#     api_key=os.environ["GEMINI_API_KEY"],
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
# )

def get_git_status():
    """
    Use this tool to get the current status of the git repository (staged, unstaged, and untracked files).
    """
    try:
        result = subprocess.run(["git", "status"], capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            return f"Failed to get git status: {result.stderr.strip()}"
        return result.stdout.strip() or "Git status is clean."
    except Exception as e:
        return f"Failed to get git status: {e}"

def get_git_diff(file_path: str = None, staged: bool = False):
    """
    Use this tool to retrieve git diff output.

    Args:
        file_path: Optional path to a specific file to check diff for.
        staged: If True, shows staged changes (--cached). Defaults to False (unstaged working directory changes).
    Returns:
        Diff output string or status message.
    """
    try:
        command = ["git", "diff"]
        if staged:
            command.append("--cached")
        if file_path:
            command.append(file_path)

        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            return f"Failed to get git diff: {result.stderr.strip()}"

        diff_output = (result.stdout or "").strip()
        if not diff_output:
            target = f"for '{file_path}'" if file_path else ""
            diff_type = "staged" if staged else "unstaged"
            return f"No {diff_type} changes found {target}.".strip()
        return diff_output
    except Exception as e:
        return f"Failed to get git diff: {e}"

def structured_call(query:str,output_structure:BaseModel):
    response = client.chat.completions.parse(
        # model="gemini-3.6-flash",
        model="google/gemini-3.6-flash",
        messages=[{"role": "user", "content": query}],
        response_format=output_structure
    )

    return response.choices[0].message.parsed

def call(messages:list[dict[str, Any]],tools:list[dict[str, Any]]):
    response = client.chat.completions.create(
            # model="gemini-3.6-flash",
            model="google/gemini-3.6-flash",
            messages=messages,
            tools=tools or None
        )
    
    return json.loads(response.choices[0].message.to_json())

