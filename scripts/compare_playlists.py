"""Generate playlists from raw requests and compare two equal-budget candidate pools."""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emomm.data import SCHEMA, read_features, tag_display_name
from emomm.playlist import PlaylistWeights, compare_pools, trajectory
from emomm.playlist_inputs import PlaylistInputs, QueryParsingError, create_request_parser, prefix_problem
from emomm.prefix_search import CompletionModel, TOLERANCE
from emomm.train import load_yaml, sha256


def resolve(path):
    return ROOT / path


def pilot_requests(count, request_csv, config, feature_id, seed):
    rng = random.Random(seed)
    with request_csv.open(encoding="utf-8-sig", newline="") as stream:
        rows = [(row["ID"], row["query"].strip()) for row in csv.DictReader(stream)
                if 0 < len(row["query"].strip()) <= 1000]
    by_query = {query: identity for identity, query in rows}
    real = [("catalog_request", by_query[query], query)
            for query in rng.sample(list(by_query), (count + 1) // 2)]
    if count == 1:
        return real
    # Dataset labels choose varied endpoint pairs. The request passed to the model is only text.
    with Path(config["sources"]["emobank"]["csv"]).open(encoding="utf-8-sig", newline="") as stream:
        options = [row for row in csv.DictReader(stream) if row["split"] == "dev"
                   and 6 <= len(row["text"].split()) <= 22 and len(row["text"]) <= 135]
    options.sort(key=lambda row: float(row["A"]))
    group_size = len(options) // 4
    controlled_count = count // 2
    if controlled_count > group_size or group_size == 0:
        raise ValueError("Not enough EmoBank validation rows for the requested pilot size")
    low = rng.sample(options[:group_size], controlled_count)
    high = rng.sample(options[-group_size:], controlled_count)
    mtg_tags = resolve(config["outputs"]["features"]) / feature_id / "mtg_tags.npz"
    with np.load(mtg_tags, allow_pickle=False) as tags:
        genres = [tag_display_name(tag) for tag in tags["preference_labels"] if tag.startswith("genre---")]
    controlled = []
    for index, (start, end) in enumerate(zip(high, low, strict=True)):
        if index % 2:
            start, end = end, start
        query = (f'My current mood is expressed by this sentence: "{start["text"].strip()}" '
                 f'By the end, I want to feel the emotion in this sentence: "{end["text"].strip()}" '
                 f'I would prefer {rng.choice(genres)} music and a gradual transition.')
        controlled.append(("emobank_pair", f'{start["id"]}:{end["id"]}', query))
    return [item for pair in zip(real[:controlled_count], controlled, strict=True) for item in pair] + real[controlled_count:]


def evaluate(request, inputs, budget, beam_width, prefix_search=False, completion_model=None,
             expansion_budget=16, search_ms=None):
    preference, negative, (anchors, observed, directed) = inputs.score_request(request)
    outputs, pools = compare_pools(inputs.va, inputs.audio, preference, negative, anchors, observed,
                                  directed, budget, beam_width)
    trackwise = outputs["trackwise_top"]
    coverage = outputs["position_coverage"]
    added = np.setdiff1d(pools["position_coverage"], pools["trackwise_top"], assume_unique=True)
    added_in_playlist = [str(inputs.ids[i]) for i in coverage["tracks"] if i in added]
    for output in outputs.values():
        output["tracks"] = [
            {"id": str(inputs.ids[index]), "va": inputs.va[index].tolist(),
             "preference_match": float(preference[index]), "negative_match": float(negative[index])}
            for index in output["tracks"]
        ]
    result = {
        "status": "complete", "parsed": request.as_dict(),
        "anchors": [anchor.tolist() if active else None for anchor, active in zip(anchors, observed, strict=True)],
        "directed": directed, "methods": outputs,
        "score_delta": coverage["score"] - trackwise["score"],
        "pool_overlap": budget - len(added), "added_tracks_in_playlist": added_in_playlist,
    }
    if prefix_search:
        problem, indices = prefix_problem(inputs, preference, negative, anchors, observed, directed, budget)
        result["prefix_search"] = {}
        variants = [("bound", None)]
        if completion_model is not None:
            variants.append(("learned", completion_model))
        for name, model in variants:
            started = perf_counter()
            solution = problem.search(expansion_budget, model, search_ms)
            elapsed_ms = (perf_counter() - started) * 1000
            result["prefix_search"][name] = {
                **asdict(solution), "order": [str(inputs.ids[indices[i]]) for i in solution.order],
                "certificate_gap": solution.upper_bound - solution.score, "search_ms": elapsed_ms,
            }
    return result


@torch.inference_mode()
def cached_problems(inputs, config, run_id, split, count, tracks, budgets, seed):
    """Generate varied trajectory requests from cached text features and public genre names."""
    feature_folder = resolve(config["outputs"]["features"]) / inputs.feature_id
    files = load_yaml(ROOT / "runs" / "train" / run_id / "effective_config.yaml")["feature_files"]
    for name in ("text_emobank", "mtg_tags"):
        if sha256(feature_folder / f"{name}.npz") != files[name]["sha256"]:
            raise ValueError(f"Cached {name} differs from selected run")
    text = read_features(feature_folder / "text_emobank.npz")
    features = text["text_embeddings"][text["splits"] == split]
    text_va = inputs.text_va(torch.as_tensor(features, device=inputs.device)).cpu().numpy()
    with np.load(feature_folder / "mtg_tags.npz", allow_pickle=False) as tags:
        genres = np.flatnonzero(np.char.startswith(tags["preference_labels"], "genre---"))
        # Hold out a fixed subset of the request genre conditions for evaluation.
        held_out = np.arange(len(genres)) % 4 == 0
        genre_ids = genres[held_out if split != "train" else ~held_out]
        prompts = tags["preference_label_embeddings"][genre_ids]
    genre_text = inputs.retriever.encode_factor("text", "preference", torch.as_tensor(prompts, device=inputs.device))
    preference = ((1.0 + (inputs.factors["preference"] @ genre_text.T).clamp(-1, 1)) / 2.0).cpu().numpy()
    rng = np.random.default_rng(seed)
    axes = np.argsort(text_va, axis=0)
    quartile = len(text_va) // 4
    for index in range(count):
        order = axes[:, index % 2]
        start = text_va[rng.choice(order[:quartile])]
        end = text_va[rng.choice(order[-quartile:])]
        if index % 4 >= 2:
            start, end = end, start
        anchors, observed, directed = trajectory(tracks, start, end)
        positive = preference[:, rng.integers(preference.shape[1])]
        for budget in budgets:
            problem, _ = prefix_problem(inputs, positive, 0.0, anchors, observed, directed, budget)
            yield budget, problem


def save_completion_model(path, model, selection):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez(stream, **asdict(model), run_id=selection["run_id"], feature_id=selection["feature_id"])


def load_completion_model(path, selection):
    with np.load(path, allow_pickle=False) as checkpoint:
        if any(str(checkpoint[key]) != selection[key] for key in ("run_id", "feature_id")):
            raise ValueError("Completion model was fit on a different MusicRec training run")
        return CompletionModel(checkpoint["mean"], checkpoint["scale"], checkpoint["coefficients"],
                               float(checkpoint["maximum"]))


def run_prefix_pilot(args, config, selection):
    device = torch.device(args.device or config["device"])
    rng = np.random.default_rng(args.seed)
    train = PlaylistInputs(args.run_id, config, "train", device)
    train_size = len(train.ids)
    labels, features = [], []
    teacher_count = args.prefix_pilot
    started = perf_counter()
    for index, (_, problem) in enumerate(cached_problems(
            train, config, args.run_id, "train", teacher_count, args.tracks, [args.budget], args.seed), start=1):
        x, y, _ = problem.teacher_labels(rng)
        features.append(x)
        labels.append(y)
        if index % 16 == 0 or index == teacher_count:
            print(f"Exact teacher: {index}/{teacher_count}", file=sys.stderr, flush=True)
    features, labels = np.concatenate(features), np.concatenate(labels)
    model = CompletionModel.fit(features, labels)
    training_seconds = perf_counter() - started
    del train

    validation = PlaylistInputs(args.run_id, config, args.split, device)
    budgets = sorted({args.budget, max(args.tracks, args.budget // 2)})
    expansion_budgets = sorted({args.expansions, max(1, args.expansions // 4)})
    measurements = {(b, e): [] for b in budgets for e in expansion_budgets}
    label_errors = []
    for index, (budget, problem) in enumerate(cached_problems(
            validation, config, args.run_id, args.split, args.eval_requests, args.tracks, budgets, args.seed + 1), start=1):
        x, y, optimum = problem.teacher_labels(rng)
        label_errors.append((np.abs(y).sum(), np.abs(y - model.predict(x)).sum(), len(y)))
        for limit in expansion_budgets:
            pair = []
            for candidate_model in (None, model):
                started = perf_counter()
                result = problem.search(limit, candidate_model, args.search_ms)
                elapsed_ms = (perf_counter() - started) * 1000
                if not result.score - TOLERANCE <= optimum <= result.upper_bound + TOLERANCE:
                    raise RuntimeError("Search interval does not contain the exact optimum")
                pair.append((max(0, optimum - result.score), result.upper_bound - result.score,
                             result.expanded, result.generated, elapsed_ms, result.score))
            measurements[budget, limit].append(pair)
        if index % 24 == 0:
            print(f"Exact validation: {index}/{args.eval_requests * len(budgets)}", file=sys.stderr, flush=True)

    results = []
    for (budget, limit), observations in measurements.items():
        metrics = np.asarray(observations)
        delta = metrics[:, 1, 5] - metrics[:, 0, 5]
        row = {"candidates": budget, "max_expansions": limit, "requests": len(observations),
               "learned_wins": int((delta > TOLERANCE).sum()), "learned_losses": int((delta < -TOLERANCE).sum()),
               "ties": int((np.abs(delta) <= TOLERANCE).sum())}
        for i, name in enumerate(("bound", "learned")):
            values = metrics[:, i, :]
            row[name] = {"mean_exact_gap": float(values[:, 0].mean()),
                         "exact_hits": int((values[:, 0] <= TOLERANCE).sum()),
                         "certified": int((values[:, 1] <= TOLERANCE).sum()),
                         "mean_expanded": float(values[:, 2].mean()), "mean_generated": float(values[:, 3].mean()),
                         "p50_search_ms": float(np.median(values[:, 4]))}
        results.append(row)
    if args.save_search_checkpoint:
        save_completion_model(args.save_search_checkpoint, model, selection)
    plain_error, model_error, validation_labels = np.asarray(label_errors).sum(axis=0)
    print(json.dumps({
        "run_id": args.run_id, "feature_id": selection["feature_id"], "split": args.split, "seed": args.seed,
        "length": args.tracks, "train_requests": teacher_count, "train_labels": len(labels),
        "search_ms_limit": args.search_ms,
        "train_nonzero_gaps": int((labels > TOLERANCE).sum()), "training_seconds": training_seconds,
        "evaluation_requests": args.eval_requests, "train_catalog_size": train_size,
        "evaluation_catalog_size": len(validation.ids), "validation_labels": int(validation_labels),
        "bound_value_mae": plain_error / validation_labels, "learned_value_mae": model_error / validation_labels,
        "results": results,
    }, allow_nan=False, indent=2))


def summarize(rows):
    complete = [row for row in rows if row["status"] == "complete"]
    deltas = np.array([row["score_delta"] for row in complete])
    position_deltas, step_deltas = [], []
    for row in complete:
        top, covered = (row["methods"][key] for key in ("trackwise_top", "position_coverage"))
        for field, values in (("mean_position_error", position_deltas), ("mean_step_error", step_deltas)):
            if top[field] is not None:
                values.append(covered[field] - top[field])
    return {
        "requests": len(rows), "complete": len(complete), "directed": sum(row["directed"] for row in complete),
        "wins": int((deltas > 1e-8).sum()), "ties": int((np.abs(deltas) <= 1e-8).sum()),
        "losses": int((deltas < -1e-8).sum()), "mean_score_delta": float(deltas.mean()) if len(deltas) else None,
        "mean_position_error_delta": float(np.mean(position_deltas)) if position_deltas else None,
        "mean_step_error_delta": float(np.mean(step_deltas)) if step_deltas else None,
        "requests_using_added_track": sum(bool(row["added_tracks_in_playlist"]) for row in complete),
    }


def summarize_prefix(rows):
    comparisons = [row["prefix_search"] for row in rows if "learned" in row.get("prefix_search", {})]
    if not comparisons:
        return None
    deltas = np.array([c["learned"]["score"] - c["bound"]["score"] for c in comparisons])
    result = {"requests": len(comparisons), "learned_wins": int((deltas > TOLERANCE).sum()),
              "learned_losses": int((deltas < -TOLERANCE).sum()), "ties": int((np.abs(deltas) <= TOLERANCE).sum()),
              "mean_score_delta": float(deltas.mean())}
    for name in ("bound", "learned"):
        values = [c[name] for c in comparisons]
        result[name] = {"certified": sum(v["certificate_gap"] <= TOLERANCE for v in values),
                        "mean_certificate_gap": float(np.mean([v["certificate_gap"] for v in values])),
                        "mean_expanded": float(np.mean([v["expanded"] for v in values])),
                        "p50_search_ms": float(np.median([v["search_ms"] for v in values]))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--text", help="Raw natural-language request")
    mode.add_argument("--pilot-requests", type=int, help="Sample local requests and build controlled text pairs")
    mode.add_argument("--prefix-pilot", type=int, metavar="TRAIN_REQUESTS",
                      help="Fit learned search against exact completions and evaluate on held-out requests")
    parser.add_argument("--tracks", type=int, default=5)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--expansions", type=int, default=16, help="Maximum expanded prefixes per search")
    parser.add_argument("--search-ms", type=float, help="Also stop after this many search milliseconds")
    parser.add_argument("--eval-requests", type=int, default=40, help="Held-out requests for the exact pilot")
    parser.add_argument("--save-search-checkpoint", type=Path, help="Save the fitted prefix search model in pilot mode")
    parser.add_argument("--prefix-search", action="store_true", help="Also run certified bound-guided search on raw requests")
    parser.add_argument("--search-checkpoint", type=Path, help="Apply learned prefix search to raw requests")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default="20261015_131238_182630")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--parser-digest", help="Explicitly pin the installed parser model digest for this run")
    parser.add_argument("--request-csv", type=Path, default=Path("D:/AAAAAAA_dataset/audio/music_reco_intent/MusicRecoIntent_dataset.csv"))
    args = parser.parse_args()
    if not 1 <= args.tracks <= 20 or not args.tracks <= args.budget <= 400 or args.beam_width < 1:
        parser.error("Require 1 <= tracks <= 20, tracks <= budget <= 400 and a positive beam width")
    if args.pilot_requests is not None and args.pilot_requests < 1:
        parser.error("Pilot request count must be positive")
    if args.expansions < 1 or args.eval_requests < 1:
        parser.error("Expansion and evaluation limits must be positive")
    if args.search_ms is not None and (not np.isfinite(args.search_ms) or args.search_ms <= 0):
        parser.error("The search deadline must be a finite positive number")
    if args.prefix_pilot is not None:
        if args.prefix_pilot < 1 or not 3 <= args.tracks <= 5 or args.budget > 24:
            parser.error("Exact prefix pilot requires a positive training count, 3–5 tracks and at most 24 candidates")
        if args.search_checkpoint or args.prefix_search:
            parser.error("Use --prefix-search or --search-checkpoint with a natural-language input mode")
    elif args.save_search_checkpoint:
        parser.error("Use --save-search-checkpoint with --prefix-pilot")
    elif args.search_ms is not None and not (args.prefix_search or args.search_checkpoint):
        parser.error("Use --search-ms with --prefix-search or --search-checkpoint")

    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    config = load_yaml(ROOT / "config" / "data.yaml")
    selection = load_yaml(ROOT / "runs" / "train" / args.run_id / "selection.yaml")
    if selection["schema"] != SCHEMA:
        raise ValueError("Unsupported MusicRec training selection schema")
    if args.prefix_pilot is not None:
        run_prefix_pilot(args, config, selection)
        return 0
    model = load_completion_model(args.search_checkpoint, selection) if args.search_checkpoint else None
    prefix_search = args.prefix_search or model is not None
    query_parser = create_request_parser(args.parser_digest)
    parser_digest = query_parser.model_digest()
    requests = ([("user", "1", args.text)] if args.text is not None else
                pilot_requests(args.pilot_requests, args.request_csv, config, selection["feature_id"], args.seed))
    results = []
    inputs = None
    for index, (source, identity, text) in enumerate(requests, start=1):
        print(f"[{index}/{len(requests)}] {source} {identity}", file=sys.stderr, flush=True)
        try:
            request = query_parser.parse(text, args.tracks)
        except QueryParsingError as exc:
            results.append({"source": source, "id": identity, "text": text,
                            "status": "parse_error", "reason": str(exc)})
            continue
        if inputs is None:
            inputs = PlaylistInputs(args.run_id, config, args.split, torch.device(args.device or config["device"]))
            if args.budget > len(inputs.ids):
                parser.error("Candidate budget exceeds the number of eligible songs")
        results.append({"source": source, "id": identity,
                        **evaluate(request, inputs, args.budget, args.beam_width, prefix_search, model,
                                   args.expansions, args.search_ms)})

    summary = summarize(results)
    print(json.dumps({
        "run_id": args.run_id, "feature_id": selection["feature_id"], "split": args.split,
        "catalog_size": len(inputs.ids) if inputs else None, "tracks": args.tracks, "budget": args.budget,
        "beam_width": args.beam_width, "seed": args.seed, "device": args.device or config["device"],
        "prefix_expansion_budget": args.expansions if prefix_search else None,
        "prefix_search_ms_limit": args.search_ms if prefix_search else None,
        "parser_model": query_parser.model, "parser_digest": parser_digest, "weights": asdict(PlaylistWeights()),
        "trajectory_profile": "linear", "negative_conditions": "soft_scores",
        "reference_conditions": "semantic_scores", "summary": summary,
        "prefix_summary": summarize_prefix(results),
        "by_source": {source: summarize([row for row in results if row["source"] == source])
                      for source in sorted({row["source"] for row in results})},
        "results": results,
    }, ensure_ascii=False, allow_nan=False, indent=2))
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
