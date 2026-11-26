"""Shared scoring, candidate selection and ordering for the playlist pilot."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlaylistWeights:
    preference: float = 0.2
    negative: float = 0.2
    position: float = 2.0
    step: float = 2.0
    local_va: float = 0.5
    content: float = 0.1


def trajectory(length, current, target):
    """Return finite anchors, their observation mask, and whether to match directed steps."""
    anchors = np.zeros((length, 2))
    observed = np.zeros(length, dtype=bool)
    if length == 1 and target is not None:
        anchors[0], observed[0] = target, True
    elif current is not None and target is not None:
        fraction = np.linspace(0.0, 1.0, length)[:, None]
        anchors[:] = current * (1.0 - fraction) + target * fraction
        observed[:] = True
    elif target is not None:
        anchors[:] = target
        observed[:] = True
    elif current is not None:
        anchors[0], observed[0] = current, True
    return anchors, observed, length > 1 and current is not None and target is not None


def candidate_pools(position_scores, budget):
    """Compare highest individual fit anywhere with round-robin position coverage."""
    length, count = position_scores.shape
    if not length <= budget <= count:
        raise ValueError("Candidate budget must be between playlist length and catalog size")
    trackwise = np.argsort(-position_scores.max(axis=0), kind="stable")[:budget]
    by_position = np.argsort(-position_scores, axis=1, kind="stable")
    used = set()
    offsets = [0] * length
    # Cycle over all positions before revisiting one, skipping previously selected IDs.
    while len(used) < budget:
        for position, ordered in enumerate(by_position):
            offset = offsets[position]
            while int(ordered[offset]) in used:
                offset += 1
            used.add(int(ordered[offset]))
            offsets[position] = offset + 1
            if len(used) == budget:
                break
    # Fix tie-breaking order in the planner independently of selection order.
    return {"trackwise_top": np.sort(trackwise), "position_coverage": np.array(sorted(used))}


def beam_order(unary, edges, width):
    """Find a unique-track sequence using a relaxed suffix bound and a shared beam width."""
    length, count = unary.shape
    if not 1 <= length <= count or width < 1:
        raise ValueError("Invalid ordering dimensions or beam width")
    suffix = np.zeros_like(unary)
    for k in range(length - 2, -1, -1):
        suffix[k] = (edges[k] + unary[k + 1] + suffix[k + 1]).max(axis=1)

    initial = np.argsort(-(unary[0] + suffix[0]), kind="stable")[:width]
    paths = [(int(i),) for i in initial]
    scores = unary[0, initial]
    for k in range(1, length):
        endings = [path[-1] for path in paths]
        extension = scores[:, None] + edges[k - 1, endings] + unary[k]
        for row, path in enumerate(paths):
            extension[row, list(path)] = -np.inf
        priorities = (extension + suffix[k]).ravel()
        best = np.argsort(-priorities, kind="stable")[:width]
        parents, next_tracks = np.divmod(best[np.isfinite(priorities[best])], count)
        paths = [paths[parent] + (int(track),) for parent, track in zip(parents, next_tracks, strict=True)]
        scores = extension[parents, next_tracks]
    return np.asarray(paths[int(scores.argmax())]), float(scores.max())


def position_scores(va, preference, negative, anchors, observed, weights=PlaylistWeights()):
    position_error = np.square(va[None, :, :] - anchors[:, None, :]).sum(axis=-1) / 8.0
    position_cost = weights.position * position_error * observed[:, None]
    unary = weights.preference * preference - weights.negative * negative - position_cost
    return unary, position_error


def edge_scores(va, audio, anchors, directed, weights=PlaylistWeights()):
    length = len(anchors)
    song_delta = va[None, :, :] - va[:, None, :]
    if directed:
        residual = song_delta[None, :, :, :] - np.diff(anchors, axis=0)[:, None, None, :]
        va_cost = weights.step * np.square(residual).sum(axis=-1) / 32.0
    else:
        va_cost = weights.local_va * np.square(song_delta).sum(axis=-1) / 8.0
    content_distance = (1.0 - np.clip(audio @ audio.T, -1.0, 1.0)) / 2.0
    scores = -(va_cost + weights.content * content_distance) / max(length - 1, 1)
    return np.broadcast_to(scores, (length - 1, len(va), len(va)))


def compare_pools(va, audio, preference, negative, anchors, observed, directed, budget, width,
                  weights=PlaylistWeights()):
    """Score both pools with the same frozen predictions, objective and beam search."""
    length = len(anchors)
    unary, position_error = position_scores(va, preference, negative, anchors, observed, weights)
    pools = candidate_pools(unary, budget)
    outputs = {}
    for name, pool in pools.items():
        edges = edge_scores(va[pool], audio[pool], anchors, directed, weights)
        order, score = beam_order(unary[:, pool] / length, edges, width)
        tracks = pool[order]
        chosen_error = position_error[np.arange(length), tracks][observed]
        chosen_delta = np.diff(va[tracks], axis=0)
        step_error = np.square(chosen_delta - np.diff(anchors, axis=0)).sum(axis=-1) / 32.0 if directed else []
        outputs[name] = {
            "tracks": tracks.tolist(),
            "score": score,
            "mean_preference": float(preference[tracks].mean()),
            "mean_position_error": float(chosen_error.mean()) if chosen_error.size else None,
            "mean_step_error": float(np.mean(step_error)) if directed else None,
            "position_error": [float(position_error[k, i]) if observed[k] else None for k, i in enumerate(tracks)],
            "step_error": np.asarray(step_error).tolist(),
            "best_available_position_error": position_error[:, pool].min(axis=1)[observed].tolist(),
        }
    return outputs, pools
