#!/usr/bin/env python3
"""
Generate sentiment and emotion over-time plots from a Gladia transcription JSON export.

Example:
    python3 plot_sentiment.py gladia_downloads/<job_id>.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List


SENTIMENT_ORDER: Dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}

SOURCE_COLOR_MAP: Dict[str, str] = {
    "gladia": "#1f77b4",  # blue
    "voice": "#ff7f0e",   # orange
}


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot sentiment and emotion analysis results over time from a Gladia JSON file."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the Gladia transcription JSON file.",
    )
    parser.add_argument(
        "--sentiment-output",
        type=Path,
        help="Optional output image path for sentiment plot (defaults to '<input_stem>_sentiment.png').",
    )
    parser.add_argument(
        "--emotion-output",
        type=Path,
        help="Optional output image path for emotion plot (defaults to '<input_stem>_emotion.png').",
    )
    return parser.parse_args(list(argv))


def _validate_times(start: float | int | None, end: float | int | None) -> tuple[float, float] | None:
    if start is None or end is None:
        return None
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return None
    return float(start), float(end)


def _load_gladia_entries(data: Dict[str, object]) -> List[Dict[str, float | str]]:
    try:
        sentiment_results = data["result"]["sentiment_analysis"]["results"]  # type: ignore[index]
    except KeyError as exc:
        raise RuntimeError(
            "Sentiment analysis data not found. "
            "Ensure the JSON contains 'result.sentiment_analysis.results'."
        ) from exc
    points: List[Dict[str, float | str]] = []
    for entry in sentiment_results:  # type: ignore[assignment]
        sentiment = entry.get("sentiment", "unknown")
        emotion = entry.get("emotion", "unknown")
        validated = _validate_times(entry.get("start"), entry.get("end"))
        if validated is None:
            continue
        start, end = validated
        midpoint = (start + end) / 2.0
        duration = abs(end - start)
        points.append(
            {
                "midpoint": midpoint,
                "duration": duration,
                "sentiment": sentiment,
                "emotion": emotion,
                "source": "gladia",
                "sentiment_match": True,
                "emotion_match": True,
            }
        )

    return points


def _load_mapped_entries(entries: List[Dict[str, object]]) -> List[Dict[str, float | str]]:
    points: List[Dict[str, float | str]] = []
    for entry in entries:
        validated = _validate_times(entry.get("start"), entry.get("end"))
        if validated is None:
            continue
        start, end = validated
        midpoint = (start + end) / 2.0
        duration = abs(end - start)

        gladia_sentiment = entry.get("gladia_sentiment", "unknown")
        voice_sentiment = entry.get("voice_sentiment", "unknown")
        gladia_emotion = entry.get("gladia_emotion", "unknown")
        voice_emotion = entry.get("voice_emotion_label", entry.get("voice_emotion", "unknown"))
        same_sentiment = voice_sentiment == gladia_sentiment
        same_emotion = voice_emotion == gladia_emotion

        points.append(
            {
                "midpoint": midpoint,
                "duration": duration,
                "sentiment": gladia_sentiment,
                "emotion": gladia_emotion,
                "source": "gladia",
                "sentiment_match": same_sentiment,
                "emotion_match": same_emotion,
            }
        )
        points.append(
            {
                "midpoint": midpoint,
                "duration": duration,
                "sentiment": voice_sentiment,
                "emotion": voice_emotion,
                "source": "voice",
                "sentiment_match": same_sentiment,
                "emotion_match": same_emotion,
            }
        )

    return points


def load_sentiment_entries(json_path: Path) -> List[Dict[str, float | str]]:
    try:
        raw_content = json.loads(json_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Input JSON file not found: {json_path}") from exc

    if isinstance(raw_content, list):
        points = _load_mapped_entries(raw_content)
    else:
        points = _load_gladia_entries(raw_content)

    if not points:
        raise RuntimeError("No usable sentiment entries found in the JSON file.")

    return points


def collect_categories(points: Iterable[Dict[str, float | str]], field: str) -> List[str]:
    unique = {str(point.get(field, "unknown")) for point in points}
    if field == "sentiment":
        return sorted(unique, key=lambda s: (SENTIMENT_ORDER.get(s, math.inf), s))
    return sorted(unique)


def ensure_matplotlib():
    try:
        import matplotlib  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to draw the sentiment graph. "
            "Install it with `pip install matplotlib`."
        ) from exc


def plot_category(points: List[Dict[str, float | str]], field: str, output_path: Path) -> None:
    ensure_matplotlib()
    import matplotlib.pyplot as plt  # imported only after ensuring availability
    from matplotlib.lines import Line2D

    categories = collect_categories(points, field)
    y_mapping = {category: index for index, category in enumerate(categories)}

    x_values = [float(point["midpoint"]) for point in points]
    y_values = [y_mapping.get(str(point.get(field, "unknown")), math.nan) for point in points]
    sizes = [max(float(point["duration"]) * 40.0, 20.0) for point in points]
    match_flags = [bool(point.get(f"{field}_match", True)) for point in points]

    MATCH_COLOR = "#7f7f7f"
    MISMATCH_COLOR = "#d62728"
    marker_map = {"gladia": "o", "voice": "x"}
    present_sources = sorted({str(point.get("source", "gladia")) for point in points})

    plt.figure(figsize=(12, 4.5))

    for source in present_sources:
        indices = [idx for idx, point in enumerate(points) if str(point.get("source", "gladia")) == source]
        if not indices:
            continue
        xs = [x_values[idx] for idx in indices]
        ys = [y_values[idx] for idx in indices]
        sc_sizes = [sizes[idx] for idx in indices]
        sc_colors = [
            MATCH_COLOR if match_flags[idx] else MISMATCH_COLOR
            for idx in indices
        ]
        plt.scatter(
            xs,
            ys,
            c=sc_colors,
            s=sc_sizes,
            alpha=0.75,
            edgecolors="k",
            linewidths=0.5,
            marker=marker_map.get(source, "o"),
        )

    plt.yticks(range(len(categories)), categories)
    plt.xlabel("Time (seconds)")
    plt.title(f"{field.title()} over time")
    plt.grid(axis="x", linestyle="--", alpha=0.4)

    shape_handles = [
        Line2D(
            [0], [0], marker=marker_map.get(source, "o"), linestyle="",
            markerfacecolor="white", markeredgecolor="black", label=source, markersize=8,
        )
        for source in present_sources
    ]

    color_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=MATCH_COLOR, markeredgecolor="black",
               label="match", markersize=8),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=MISMATCH_COLOR, markeredgecolor="black",
               label="mismatch", markersize=8),
    ]

    legend1 = plt.legend(handles=shape_handles, title="Source", loc="upper right")
    plt.gca().add_artist(legend1)
    plt.legend(handles=color_handles, title="Comparison", loc="lower right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)

    sentiment_output = args.sentiment_output or args.input_path.with_name(
        f"{args.input_path.stem}_sentiment.png"
    )
    emotion_output = args.emotion_output or args.input_path.with_name(
        f"{args.input_path.stem}_emotion.png"
    )

    try:
        points = load_sentiment_entries(args.input_path)
        plot_category(points, "sentiment", sentiment_output)
        print(f"Sentiment plot saved to {sentiment_output}")

        plot_category(points, "emotion", emotion_output)
        print(f"Emotion plot saved to {emotion_output}")
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

