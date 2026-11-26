"""Bounded playlist search with exact completion targets for learned node priorities."""

from dataclasses import dataclass
from functools import cache
import heapq
from itertools import count
from time import perf_counter

import numpy as np


TOLERANCE = 1e-10


@dataclass(frozen=True)
class CompletionModel:
    """A ridge regressor for the gap between a relaxed bound and a legal completion."""

    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    maximum: float

    @classmethod
    def fit(cls, features, gaps):
        mean, scale = features.mean(axis=0), features.std(axis=0).clip(1e-8)
        x = np.column_stack((np.ones(len(features)), (features - mean) / scale))
        penalty = np.eye(x.shape[1]) * (0.01 * len(features))
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(x.T @ x + penalty, x.T @ gaps)
        return cls(mean, scale, coefficients, float(gaps.max()))

    def predict(self, features):
        if not len(features):
            return np.empty(0)
        prediction = self.coefficients[0] + (features - self.mean) / self.scale @ self.coefficients[1:]
        # A relaxed path with no conflicts already attains the bound.
        return np.where(features[:, 1] + features[:, 2] > 0, prediction.clip(0, self.maximum), 0.0)


@dataclass(frozen=True)
class SearchResult:
    order: tuple[int, ...]
    score: float
    upper_bound: float
    expanded: int
    generated: int


class PrefixProblem:
    """One request's position and transition scores on a shared candidate set."""

    def __init__(self, unary, edges):
        self.unary = np.asarray(unary, dtype=np.float64)
        self.edges = np.asarray(edges, dtype=np.float64)
        self.length, self.size = self.unary.shape
        if (not 1 <= self.length <= self.size or self.edges.shape != (self.length - 1, self.size, self.size)
                or not np.isfinite(self.unary).all() or not np.isfinite(self.edges).all()):
            raise ValueError("Invalid playlist objective")

        self.suffix = np.zeros_like(self.unary)
        self.next_values = np.empty_like(self.edges)
        self.next_order = np.empty(self.edges.shape, dtype=int)
        self.relaxed_tail = np.empty((self.length, self.size, self.length - 1), dtype=int)
        self.relaxed_repeats = np.zeros((self.length, self.size), dtype=int)
        for position in range(self.length - 2, -1, -1):
            options = self.edges[position] + self.unary[position + 1] + self.suffix[position + 1]
            np.fill_diagonal(options, -np.inf)
            order = np.argsort(-options, axis=1, kind="stable")
            best = order[:, 0]
            self.next_values[position] = options
            self.next_order[position] = order
            self.suffix[position] = options[np.arange(self.size), best]
            remaining = self.length - position - 1
            self.relaxed_tail[position, :, 0] = best
            self.relaxed_tail[position, :, 1:remaining] = self.relaxed_tail[position + 1, best, :remaining - 1]
            repeats_current = (self.relaxed_tail[position, :, :remaining] == np.arange(self.size)[:, None]).any(axis=1)
            self.relaxed_repeats[position] = self.relaxed_repeats[position + 1, best] + repeats_current

    def children(self, prefix, score, with_features=False):
        """Score legal next tracks and tighten the relaxed bound by one legal step."""
        position = len(prefix)
        used = np.zeros(self.size, dtype=bool)
        used[list(prefix)] = True
        tracks = np.flatnonzero(~used)
        immediate = self.unary[position, tracks].copy()
        if prefix:
            immediate += self.edges[position - 1, prefix[-1], tracks]
        earned = score + immediate
        remaining = self.length - position - 1
        features = np.zeros((len(tracks), 7)) if with_features else None
        if not remaining:
            return tracks, earned, earned, features

        options = self.next_order[position, tracks, :len(prefix) + 2]
        # The current track has -inf in the relaxed ordering. Excluding it also
        # handles the case where the entire candidate set is still available.
        allowed = ~used[options] & (options != tracks[:, None])
        first = options[np.arange(len(tracks)), allowed.argmax(axis=1)]
        tail_bound = self.next_values[position, tracks, first]
        if with_features:
            tail = np.column_stack((first, self.relaxed_tail[position + 1, first, :remaining - 1]))
            blocked_hits = (used[tail] | (tail == tracks[:, None])).sum(axis=1)
            repeated = self.relaxed_repeats[position + 1, first]
            features = np.column_stack((
                np.full(len(tracks), remaining), blocked_hits, repeated,
                self.suffix[position, tracks] - tail_bound, immediate,
                tail_bound / remaining, np.full(len(tracks), self.size - self.length),
            ))
        return tracks, earned, earned + tail_bound, features

    def greedy(self):
        prefix = ()
        score = 0.0
        while len(prefix) < self.length:
            tracks, earned, bounds, _ = self.children(prefix, score)
            best = int(bounds.argmax())
            prefix += (int(tracks[best]),)
            score = float(earned[best])
        return prefix, score

    def search(self, expansion_budget, model=None, milliseconds=None):
        """Use the model for queue order. The admissible bound alone controls pruning."""
        if expansion_budget < 0:
            raise ValueError("Expansion budget must be nonnegative")
        if milliseconds is not None and (not np.isfinite(milliseconds) or milliseconds <= 0):
            raise ValueError("Time budget must be finite and positive")
        deadline = perf_counter() + milliseconds / 1000.0 if milliseconds is not None else None
        best, best_score = self.greedy()
        root_bound = float((self.unary[0] + self.suffix[0]).max())
        serial = count()
        queue = [(-root_bound, next(serial), (), 0.0, root_bound)]
        expanded, generated = 0, 0
        while queue and expanded < expansion_budget:
            if deadline is not None and perf_counter() >= deadline:
                break
            _, _, prefix, score, bound = heapq.heappop(queue)
            if bound <= best_score + TOLERANCE:
                continue
            tracks, earned, bounds, features = self.children(prefix, score, model is not None)
            expanded += 1
            generated += len(tracks)
            if len(prefix) + 1 == self.length:
                best_index = int(earned.argmax())
                if earned[best_index] > best_score:
                    best, best_score = prefix + (int(tracks[best_index]),), float(earned[best_index])
                continue
            active = np.flatnonzero(bounds > best_score + TOLERANCE)
            priorities = bounds[active]
            if model is not None:
                priorities = priorities - model.predict(features[active])
            for index, priority in zip(active, priorities, strict=True):
                order = prefix + (int(tracks[index]),)
                heapq.heappush(queue, (-float(priority), next(serial), order, float(earned[index]), float(bounds[index])))
        remaining_bound = max((entry[-1] for entry in queue), default=best_score)
        return SearchResult(best, best_score, max(best_score, remaining_bound), expanded, generated)

    def exact_values(self):
        """Return a memoized exact suffix teacher for the small-pool experiment."""
        unary, edges = self.unary, self.edges
        all_tracks = (1 << self.size) - 1

        @cache
        def suffix(last, used):
            position = used.bit_count()
            if position == self.length:
                return 0.0
            remaining = all_tracks ^ used
            best = -np.inf
            while remaining:
                bit = remaining & -remaining
                j = bit.bit_length() - 1
                reward = unary[position, j] + (edges[position - 1, last, j] if position else 0.0)
                best = max(best, reward + suffix(j, used | bit))
                remaining ^= bit
            return best

        return suffix

    def teacher_labels(self, rng):
        """Mix relaxed greedy and sampled prefixes; label each legal action by exact completion."""
        suffix = self.exact_values()
        optimum = suffix(-1, 0)
        greedy, _ = self.greedy()
        prefixes = {()}
        for depth in range(1, self.length - 2):
            prefixes.add(greedy[:depth])
            prefixes.add(tuple(map(int, rng.choice(self.size, depth, replace=False))))
        features, gaps = [], []
        for prefix in sorted(prefixes):
            used = sum(1 << track for track in prefix)
            tracks, earned, bound, x = self.children(prefix, 0.0, with_features=True)
            exact = np.array([reward + suffix(int(track), used | (1 << int(track)))
                              for track, reward in zip(tracks, earned, strict=True)])
            gap = bound - exact
            if np.any(gap < -TOLERANCE):
                raise RuntimeError("Relaxed bound below exact legal completion")
            features.append(x)
            gaps.append(gap.clip(0))
        suffix.cache_clear()
        return np.concatenate(features), np.concatenate(gaps), float(optimum)
