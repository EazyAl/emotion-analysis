"""
vad2gladia.py

Map Valence–Arousal–Dominance (VAD) scores to Gladia-style
Sentiments and Emotions with probabilities.

Inputs are typically in [0,1]. We normalize to [-1,1],
score against emotion centroids in PAD space, then:
  - choose a discrete emotion via softmax over (negative) distance
  - choose a sentiment from valence with thresholds
  - return 'unknown' emotion if everything is too far from any centroid
  - provide an optional utterance-level sentiment aggregator that can output 'mixed'

No external deps beyond numpy.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable
import math
import numpy as np


# ---------------------------
# Config & Centroids
# ---------------------------

# Gladia sentiment label set
SENTIMENT_LABELS = ("positive", "negative", "neutral", "mixed", "unknown")

# Emotion centroids in [-1,1]^3 (Valence, Arousal, Dominance)
# These are well-motivated heuristics from PAD/circumplex literature.
# Calibrate per your data by nudging values or learning them from a dev set.
DEFAULT_CENTROIDS: Dict[str, Tuple[float, float, float]] = {
  # Positive-ish
  "adoration":         (+0.85, +0.45, -0.10),
  "amusement":         (+0.70, +0.55, +0.10),
  "awe":               (+0.20, +0.80, -0.60),
  "contentment":       (+0.70, -0.30, +0.10),
  "desire":            (+0.60, +0.60, +0.35),
  "ecstatic":          (+0.95, +0.95, +0.50),
  "elation":           (+0.85, +0.80, +0.40),
  "interest":          (+0.35, +0.45,  0.00),
  "relief":            (+0.60, -0.40, +0.10),
  "positive_surprise": (+0.40, +0.85, -0.30),
  "triumph":           (+0.80, +0.70, +0.90),
  "sympathy":          (-0.10, +0.20, -0.40),

  # Neutral-ish
  "neutral":           ( 0.00,  0.00,  0.00),
  "realization":       (+0.05, +0.05, +0.05),
  "confusion":         (-0.15, +0.25, -0.50),

  # Negative-ish
  "anger":             (-0.60, +0.75, +0.55),
  "contempt":          (-0.55, +0.15, +0.65),   # colder, high control
  "disgust":           (-0.70, +0.40, +0.20),
  "distress":          (-0.70, +0.60, -0.60),
  "embarrassment":     (-0.40, +0.50, -0.70),
  "fear":              (-0.80, +0.85, -0.70),
  "pain":              (-0.70, +0.60, -0.60),
  "sadness":           (-0.80, -0.40, -0.50),
  "disappointment":    (-0.50, -0.20, -0.40),
  "negative_surprise": (-0.40, +0.85, -0.40),
}


@dataclass
class VAD2GladiaConfig:
    # Axis weights for the distance metric (Valence carries sentiment, Dominance separates anger/fear)
    w_valence: float = 0.6
    w_arousal: float = 0.25
    w_dominance: float = 0.15

    # Softmax temperature (higher alpha => sharper distribution)
    alpha: float = 3.0

    # Sentiment thresholds on centered valence ([-1,1])
    t_pos: float = +0.15
    t_neg: float = -0.15

    # Speech/VAD confidence threshold below which we return 'unknown' sentiment
    speech_conf_min: float = 0.4

    # Emotion "unknown" threshold: if the min normalized distance exceeds this, predict 'unknown'
    # Distances are normalized by the theoretical max (see _max_possible_distance()).
    # Start at 0.55–0.65 and tune.
    unknown_dist_threshold: float = 0.55

    # Emotion centroids
    centroids: Dict[str, Tuple[float, float, float]] = field(default_factory=lambda: DEFAULT_CENTROIDS)


# ---------------------------
# Core utilities
# ---------------------------

def _center_01_to_pm1(x: float) -> float:
    """Map a single [0,1] value to [-1,1]."""
    return 2.0 * x - 1.0


def normalize_vad(valence: float, arousal: float, dominance: float, input_range: str = "zero_one") -> Tuple[float, float, float]:
    """
    Normalize incoming VAD to the common PAD cube [-1,1]^3.

    Args:
        valence, arousal, dominance: floats either in [0,1] or [-1,1]
        input_range: "zero_one" (default) or "pm_one" (already [-1,1])

    Returns:
        (v, a, d) in [-1,1]
    """
    if input_range not in ("zero_one", "pm_one"):
        raise ValueError("input_range must be 'zero_one' or 'pm_one'")

    if input_range == "zero_one":
        return (_center_01_to_pm1(valence),
                _center_01_to_pm1(arousal),
                _center_01_to_pm1(dominance))
    else:
        # already centered
        return (float(valence), float(arousal), float(dominance))


def _max_possible_distance(cfg: VAD2GladiaConfig) -> float:
    """
    The maximum weighted Euclidean distance between any two points in [-1,1]^3.
    Each axis can differ by at most 2, so:
        max_dist = sqrt( wV*(2)^2 + wA*(2)^2 + wD*(2)^2 ) = 2 * sqrt(wV + wA + wD)
    With default weights summing to 1, max_dist = 2.
    """
    wsum = cfg.w_valence + cfg.w_arousal + cfg.w_dominance
    return 2.0 * math.sqrt(wsum)


def _weighted_distance(v: float, a: float, d: float,
                       vc: float, ac: float, dc: float,
                       cfg: VAD2GladiaConfig) -> float:
    """Weighted Euclidean distance in PAD space."""
    dv = v - vc
    da = a - ac
    dd = d - dc
    return math.sqrt(
        cfg.w_valence   * dv * dv +
        cfg.w_arousal   * da * da +
        cfg.w_dominance * dd * dd
    )


def emotion_probabilities(v: float, a: float, d: float, cfg: VAD2GladiaConfig) -> Tuple[List[Tuple[str, float]], Dict[str, float]]:
    """
    Compute softmax probabilities over emotions based on distances to centroids.

    Returns:
        ranked: list of (emotion, prob) sorted desc
        dist_norm: dict of emotion -> normalized distance in [0,1]
    """
    maxd = _max_possible_distance(cfg)
    dists = {}
    for emo, (vc, ac, dc) in cfg.centroids.items():
        dist = _weighted_distance(v, a, d, vc, ac, dc, cfg)
        dists[emo] = dist / maxd  # normalize so alpha is stable across configs

    # Softmax over NEGATIVE normalized distance
    emos, norms = zip(*dists.items())  # deterministic order
    logits = np.array([-cfg.alpha * x for x in norms], dtype=np.float64)
    logits -= logits.max()  # numerical stability
    probs = np.exp(logits)
    probs /= probs.sum()

    ranked = sorted(zip(emos, probs.tolist()), key=lambda x: x[1], reverse=True)
    return ranked, dists


def predict_emotion(v: float, a: float, d: float, cfg: VAD2GladiaConfig) -> Tuple[str, float, float]:
    """
    Predict a single emotion with probability, or 'unknown' if too far.

    Returns:
        label: top-1 emotion or 'unknown'
        prob: probability of label (1.0 if 'unknown')
        min_norm_dist: the smallest normalized distance to any centroid (for diagnostics)
    """
    ranked, dist_norm = emotion_probabilities(v, a, d, cfg)
    top_emo, top_p = ranked[0]
    min_norm_dist = min(dist_norm.values())

    if min_norm_dist > cfg.unknown_dist_threshold:
        confidence = 0.0
        return "unknown", 1.0, confidence, min_norm_dist

    confidence = max(0.0, min(1.0, 1.0 - min_norm_dist))
    return top_emo, float(top_p), confidence, min_norm_dist


def predict_sentiment(v: float, a: float, d: float, cfg: VAD2GladiaConfig,
                      speech_confidence: float | None = 1.0) -> str:
    """
    Map centered valence ([-1,1]) to Gladia sentiments.

    Rules:
      - 'unknown'  if speech_confidence is low
      - 'positive' if v >= t_pos
      - 'negative' if v <= t_neg
      - 'neutral'  otherwise
    """
    if speech_confidence is not None and speech_confidence < cfg.speech_conf_min:
        return "unknown"
    if v >= cfg.t_pos:
        return "positive"
    if v <= cfg.t_neg:
        return "negative"
    return "neutral"


def aggregate_sentiment(segment_sentiments: Iterable[str],
                        min_frac_each: float = 0.20) -> str:
    """
    Utterance-/clip-level sentiment with 'mixed' support.
    If >= min_frac_each are positive AND negative, return 'mixed';
    else return majority (ties broken by presence of pos/neg over neutral).
    'unknown' votes are ignored.
    """
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    total = 0
    for s in segment_sentiments:
        if s not in counts:
            continue
        counts[s] += 1
        total += 1
    if total == 0:
        return "unknown"

    pos_frac = counts["positive"] / total
    neg_frac = counts["negative"] / total
    if pos_frac >= min_frac_each and neg_frac >= min_frac_each:
        return "mixed"

    # majority among pos/neg/neutral
    return max(counts.items(), key=lambda kv: kv[1])[0]


# ---------------------------
# High-level API
# ---------------------------

def map_vad_to_gladia(
    valence: float,
    arousal: float,
    dominance: float,
    *,
    input_range: str = "zero_one",
    speech_confidence: float | None = 1.0,
    cfg: VAD2GladiaConfig | None = None,
) -> Dict:
    """
    One-stop mapping from VAD to Gladia-style outputs.

    Returns dict with:
      - normalized_vad: (v,a,d) in [-1,1]
      - sentiment: str in {'positive','negative','neutral','unknown'}
      - emotion: {'label': str, 'prob': float, 'min_norm_dist': float}
      - emotion_probs: list[(emotion, prob)] sorted desc
      - distances: dict[emotion] -> normalized distance [0,1]
    """
    cfg = cfg or VAD2GladiaConfig()
    v, a, d = normalize_vad(valence, arousal, dominance, input_range=input_range)

    # Sentiment
    sentiment = predict_sentiment(v, a, d, cfg, speech_confidence=speech_confidence)

    # Emotions (with 'unknown' fallback based on distance)
    ranked, dist_norm = emotion_probabilities(v, a, d, cfg)
    emo_label, emo_prob, emo_confidence, min_norm_dist = predict_emotion(v, a, d, cfg)

    return {
        "normalized_vad": {"valence": v, "arousal": a, "dominance": d},
        "sentiment": sentiment,
        "emotion": {
            "label": emo_label,
            "prob": emo_prob,
            "confidence": emo_confidence,
            "min_norm_dist": min_norm_dist,
        },
        "emotion_probs": ranked,            # list of (emotion, probability)
        "distances": dist_norm,             # emotion -> normalized distance [0,1]
        "config": cfg,
    }


# ---------------------------
# Example usage / smoke test
# ---------------------------

if __name__ == "__main__":
    # Example from your message (values looked like [0,1])
    val01, aro01, dom01 = 0.3854922056, 0.6586869955, 0.6556775570

    out = map_vad_to_gladia(
        valence=val01,
        arousal=aro01,
        dominance=dom01,
        input_range="zero_one",   # set to "pm_one" if already in [-1,1]
        speech_confidence=1.0,    # set from your VAD/ASR conf if available
        cfg=VAD2GladiaConfig(
            w_valence=0.5, w_arousal=0.3, w_dominance=0.2,
            alpha=3.0,
            t_pos=+0.15, t_neg=-0.15,
            speech_conf_min=0.4,
            unknown_dist_threshold=0.60,  # tune on a dev set
        )
    )

    print("Normalized VAD:", out["normalized_vad"])
    print("Sentiment:", out["sentiment"])
    print("Emotion:", out["emotion"])
    print("Top-5 emotions:", out["emotion_probs"][:5])