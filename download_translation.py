#!/usr/bin/env python3
"""
Download and persist Gladia transcription/translation results locally.

Usage:
    python3 download_translation.py             # fetch latest job
    python3 download_translation.py --id <uuid> # fetch specific job
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List


KEY_FILE = Path("/Users/aliimran/Sehar/gladia/keys")
API_BASE = "https://api.gladia.io/v2/transcription"
OUTPUT_DIR = Path("/Users/aliimran/Sehar/gladia/gladia_downloads")


def load_api_key() -> str:
    try:
        content = KEY_FILE.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(f"Could not read key file at {KEY_FILE}") from exc

    match = re.search(r"gladia\s*=\s*['\"]([^'\"]+)['\"]", content)
    if not match:
        raise RuntimeError("API key not found in keys file; expected format `gladia = '...'`")

    return match.group(1).strip()


def api_request(endpoint: str, api_key: str) -> Dict[str, Any]:
    headers = {
        "x-gladia-key": api_key,
        "Accept": "application/json",
    }
    request = urllib.request.Request(endpoint, headers=headers, method="GET")

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as http_error:
        error_body = http_error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Gladia API returned HTTP {http_error.code}: {error_body}"
        ) from http_error
    except urllib.error.URLError as url_error:
        raise RuntimeError(f"Network error while contacting Gladia: {url_error}") from url_error

    try:
        return json.loads(payload)
    except json.JSONDecodeError as decode_error:
        raise RuntimeError(f"Unexpected non-JSON response from Gladia: {payload}") from decode_error


def fetch_latest_job_id(api_key: str) -> str:
    query = urllib.parse.urljoin(API_BASE, "?limit=1&offset=0")
    data = api_request(query, api_key)
    items: List[Dict[str, Any]] = data.get("items", [])
    if not items:
        raise RuntimeError("No transcription jobs found in your Gladia account.")
    return items[0]["id"]


def fetch_job_details(job_id: str, api_key: str) -> Dict[str, Any]:
    endpoint = f"{API_BASE}/{job_id}"
    return api_request(endpoint, api_key)


def extract_plain_text(job: Dict[str, Any]) -> str:
    result = job.get("result", {})
    transcription = result.get("transcription", {})
    utterances: List[Dict[str, Any]] = transcription.get("utterances", [])
    if utterances:
        return "\n".join(segment.get("text", "") for segment in utterances if segment.get("text"))

    translation = result.get("translation", {})
    translated_text = translation.get("translated_text")
    if translated_text:
        return str(translated_text)

    return ""


def save_results(job: Dict[str, Any]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    job_id = job.get("id", "unknown")

    json_path = OUTPUT_DIR / f"{job_id}.json"
    text_path = OUTPUT_DIR / f"{job_id}.txt"

    json_path.write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")

    plain_text = extract_plain_text(job)
    if plain_text:
        text_path.write_text(plain_text, encoding="utf-8")
    else:
        text_path.write_text("No transcription or translation text found.", encoding="utf-8")

    return json_path


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and store Gladia transcription results.")
    parser.add_argument(
        "--id",
        dest="job_id",
        help="Specific transcription/translation job UUID to download. Defaults to the latest job.",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    try:
        api_key = load_api_key()
        job_id = args.job_id or fetch_latest_job_id(api_key)
        job_details = fetch_job_details(job_id, api_key)
        save_results(job_details)
        print(f"Saved Gladia job {job_id} to {OUTPUT_DIR}")
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

