#!/usr/bin/env python3
"""
Run local inference with the SER Odyssey WavLM multi-attribute model.

Usage:
    python emotion_recognition.py --audio /Users/aliimran/Sehar/gladia/Rocky.mp3
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
from transformers import AutoModelForAudioClassification

from vad2gladia import map_vad_to_gladia


LOGGER = logging.getLogger(__name__)

MODEL_ID = "3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes"


def load_audio(path: Path, target_rate: int) -> Tuple[np.ndarray, int]:
    """Load audio as mono signal resampled to the target rate."""
    waveform, sampling_rate = librosa.load(
        path.as_posix(),
        sr=target_rate,
        mono=True,
    )
    LOGGER.debug("Loaded audio: shape=%s, sampling_rate=%s", waveform.shape, sampling_rate)
    return waveform, sampling_rate


@dataclass
class SegmentResult:
    index: int
    start: float
    end: float
    speaker: Optional[int]
    channel: Optional[int]
    text: str
    gladia_sentiment: Optional[str]
    gladia_emotion: Optional[str]
    voice_sentiment: str
    voice_emotion_label: str
    voice_emotion_prob: float
    normalized_vad: Dict[str, float]


@dataclass
class InferenceResult:
    audio_path: str
    device: str
    segment_results: List[SegmentResult]
    emotion_counts: Dict[str, int]
    clip_scores: Optional[Dict[str, float]] = None
    embedding_preview: Optional[List[float]] = None
    clip_voice_sentiment: Optional[str] = None
    clip_voice_emotion: Optional[Dict[str, Any]] = None


def load_gladia_segments(json_path: Optional[Path]) -> List[Dict]:
    if not json_path:
        return []

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Gladia JSON not found: {json_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gladia JSON is not valid JSON: {json_path}") from exc

    result = data.get("result", {})
    sentiment_block = result.get("sentiment_analysis", {}) or {}
    sentiment_results = sentiment_block.get("results", []) or []

    if sentiment_results:
        return sentiment_results

    # Fallback to transcription utterances if sentiment results are missing.
    transcription = result.get("transcription", {}) or {}
    utterances = transcription.get("utterances", []) or []
    return [
        {
            "text": utterance.get("text", ""),
            "start": float(utterance.get("start", 0.0)),
            "end": float(utterance.get("end", 0.0)),
            "speaker": utterance.get("speaker"),
            "channel": utterance.get("channel"),
        }
        for utterance in utterances
    ]


def slice_waveform(
    waveform: np.ndarray,
    sampling_rate: int,
    start: float,
    end: float,
) -> np.ndarray:
    start_sample = max(int(start * sampling_rate), 0)
    end_sample = min(int(end * sampling_rate), waveform.shape[-1])
    if end_sample <= start_sample:
        return np.zeros((1,), dtype=waveform.dtype)
    return waveform[start_sample:end_sample]


def run_inference(
    audio_path: Path,
    device: torch.device,
    gladia_segments: Optional[List[Dict]] = None,
) -> InferenceResult:
    LOGGER.info("Loading SER model %s on device %s", MODEL_ID, device)
    model = AutoModelForAudioClassification.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    config = model.config
    target_rate = int(getattr(config, "sampling_rate", 16_000))
    mean = float(getattr(config, "mean", 0.0))
    std = float(getattr(config, "std", 1.0))
    eps = 1e-6
    LOGGER.info("Model params: sampling_rate=%s mean=%.6f std=%.6f", target_rate, mean, std)

    waveform, sampling_rate = load_audio(audio_path, target_rate)
    LOGGER.info("Loaded audio waveform with %s samples (original sr=%s)", waveform.shape[-1], sampling_rate)
    emotion_counts: Dict[str, int] = {}

    segment_results: List[SegmentResult] = []

    if gladia_segments:
        total_segments = len(gladia_segments)
        LOGGER.info("Processing %s Gladia segments", total_segments)
        for index, segment in enumerate(gladia_segments, start=1):
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", start))
            segment_waveform = slice_waveform(waveform, target_rate, start, end)

            if segment_waveform.size <= 1:
                LOGGER.warning(
                    "Skipping segment %s due to insufficient audio samples (start=%s, end=%s)",
                    index,
                    start,
                    end,
                )
                continue

            norm_segment = (segment_waveform - mean) / (std + eps)
            segment_values = torch.tensor(
                norm_segment,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            segment_mask = torch.ones(
                (1, segment_values.shape[-1]),
                dtype=torch.bool,
                device=device,
            )

            LOGGER.info(
                "Segment %03d/%03d | span %.3fs-%.3fs | %d samples",
                index,
                total_segments,
                start,
                end,
                segment_values.shape[-1],
            )
            seg_start_time = time.perf_counter()
            with torch.no_grad():
                segment_outputs = model(segment_values, segment_mask)
            seg_elapsed = time.perf_counter() - seg_start_time
            arousal, dominance, valence = segment_outputs.detach().cpu().numpy().squeeze().tolist()
            LOGGER.debug(
                "Segment %03d logits (arousal=%.4f, dominance=%.4f, valence=%.4f) in %.2fs",
                index,
                arousal,
                dominance,
                valence,
                seg_elapsed,
            )
            segment_mapping = map_vad_to_gladia(
                valence=valence,
                arousal=arousal,
                dominance=dominance,
                input_range="zero_one",
            )
            voice_sentiment = segment_mapping["sentiment"]
            voice_emotion_label = segment_mapping["emotion"]["label"]
            voice_emotion_prob = float(segment_mapping["emotion"]["confidence"])
            normalized_vad = segment_mapping["normalized_vad"]
            emotion_counts[voice_emotion_label] = emotion_counts.get(voice_emotion_label, 0) + 1
            segment_results.append(
                SegmentResult(
                    index=index,
                    start=start,
                    end=end,
                    speaker=segment.get("speaker"),
                    channel=segment.get("channel"),
                    text=segment.get("text", ""),
                    gladia_sentiment=segment.get("sentiment"),
                    gladia_emotion=segment.get("emotion"),
                    voice_sentiment=voice_sentiment,
                    voice_emotion_label=voice_emotion_label,
                    voice_emotion_prob=voice_emotion_prob,
                    normalized_vad={
                        "valence": float(normalized_vad["valence"]),
                        "arousal": float(normalized_vad["arousal"]),
                        "dominance": float(normalized_vad["dominance"]),
                    },
                )
            )

    LOGGER.info("Finished inference for %s segments", len(segment_results))
    LOGGER.info("Emotion counts: %s", emotion_counts)

    return InferenceResult(
        audio_path=str(audio_path),
        device=str(device),
        segment_results=segment_results,
        emotion_counts=emotion_counts,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio",
        type=Path,
        default=Path("/Users/aliimran/Sehar/gladia/Rocky.mp3"),
        help="Path to the audio file (default: Rocky.mp3 in repository root).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use (default: cuda if available else cpu).",
    )
    parser.add_argument(
        "--gladia-json",
        type=Path,
        help="Optional path to a Gladia transcription JSON file for per-segment analysis.",
    )
    parser.add_argument(
        "--segments-output",
        type=Path,
        default=Path("mapped_emotions.json"),
        help="Path to save segment emotion results as JSON (default: mapped_emotions.json).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (default: INFO).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    LOGGER.debug("Arguments: %s", args)

    audio_path = args.audio.expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    LOGGER.info("Running inference for audio file %s", audio_path)

    device = torch.device(args.device)
    LOGGER.info("Using device: %s", device)
    gladia_segments = load_gladia_segments(args.gladia_json)
    if gladia_segments:
        LOGGER.info("Loaded %s segments from %s", len(gladia_segments), args.gladia_json)
    else:
        LOGGER.info("No Gladia segments provided; clip-level analysis only")
    result = run_inference(audio_path, device, gladia_segments)

    print(f"Audio file: {result.audio_path}")
    print(f"Device: {result.device}")
    if result.segment_results:
        print()
        print(f"Computed segment scores for {len(result.segment_results)} segments.")
        for segment in result.segment_results:
            print(
                f"[{segment.index:03d}] {segment.start:7.3f}s -> {segment.end:7.3f}s "
                f"Speaker {segment.speaker if segment.speaker is not None else 'N/A'} "
                f"Text: {segment.text!r}"
            )
            print(
                f"      Gladia sentiment={segment.gladia_sentiment or 'N/A'} "
                f"emotion={segment.gladia_emotion or 'N/A'}"
            )
            print(
                f"      Voice sentiment/emotion: sentiment={segment.voice_sentiment}, "
                f"emotion={segment.voice_emotion_label} "
                f"(confidence={segment.voice_emotion_prob:.3f})"
            )
            print(
                f"      Normalized VAD: valence={segment.normalized_vad['valence']:.3f}, "
                f"arousal={segment.normalized_vad['arousal']:.3f}, "
                f"dominance={segment.normalized_vad['dominance']:.3f}"
            )

    if result.emotion_counts:
        print()
        print("Emotion counts:")
        for label, count in sorted(result.emotion_counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {label}: {count}")

    if args.segments_output:
        output_path = args.segments_output.expanduser().resolve()
        segments_payload = [
            {
                "index": segment.index,
                "start": segment.start,
                "end": segment.end,
                "speaker": segment.speaker,
                "channel": segment.channel,
                "text": segment.text,
                "gladia_sentiment": segment.gladia_sentiment,
                "gladia_emotion": segment.gladia_emotion,
                "voice_sentiment": segment.voice_sentiment,
                "voice_emotion_label": segment.voice_emotion_label,
                "voice_emotion_prob": segment.voice_emotion_prob,
                "normalized_vad": segment.normalized_vad,
            }
            for segment in result.segment_results
        ]
        output_path.write_text(
            json.dumps(segments_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print()
        print(f"Wrote segment results to {output_path}")


if __name__ == "__main__":
    main()

