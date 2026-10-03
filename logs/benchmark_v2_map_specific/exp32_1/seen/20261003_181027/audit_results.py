"""Read-only acceptance checks for this Seen metric run."""

import argparse
import hashlib
import json
import math
from pathlib import Path


MAPS = "cs_agency cs_italy de_ancient de_anubis de_dust2 de_inferno de_mirage de_nuke de_overpass de_train".split()
REQUIRED = {
    "discrete": "PSNR SSIM LPIPS Boundary_F1 FID Pixel_MAE_255 CLIP".split(),
    "continuous": "PSNR SSIM LPIPS Temporal_Warping_Error Temporal_Difference_Error FVD Boundary_F1 Optical_Flow_EPE Pixel_MAE_255".split(),
}


def load(path):
    return json.loads(path.read_text())


def finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def audit(project, kind):
    root = project / "outputs_eval/benchmark_v2/exp32_1/seen" / kind
    continuous = kind == "continuous"
    count = 1280 if continuous else 2000
    split = "seen_continuous" if continuous else "seen_discrete_test"
    prefix = "benchmark_csgo_v2_conti" if continuous else "benchmark_csgo_v2"
    expected = {root / f"{prefix}_{name}.json" for name in MAPS}
    assert set(root.glob(f"{prefix}_*.json")) == expected, f"{kind}: map file set"
    assets = project / "data/csgo_benchmark_v2/minimal_dataset_report.json"
    asset_hash = hashlib.sha256(assets.read_bytes()).hexdigest()
    selected_hash = load(assets)["selected_images"]["sha256"]
    inference = load(root / "inference_manifest.json")
    assert inference["maps"] == MAPS
    assert inference["sample_count"] == count * len(MAPS)
    assert inference["seed"] == 42
    checkpoint = (project / "outputs/csgo_1b/exp32_1/model.safetensors").resolve()
    assert (project / inference["checkpoint"]).resolve() == checkpoint
    per_map = {}
    for name in MAPS:
        result = load(root / f"{prefix}_{name}.json")
        metrics = result["metrics_ordered"]
        assert result["map_name"] == name
        assert result["metric_profile"] == "benchmark_v2_core"
        assert result["benchmark_v2_split"] == split
        assert result["benchmark_v2_asset_backend"] == "minimal"
        assert result["benchmark_v2_asset_manifest_sha256"] == asset_hash
        assert result["benchmark_v2_selected_images_sha256"] == selected_hash
        assert result["inference_provenance"]["payload"] == inference
        for key in REQUIRED[kind]:
            assert finite(metrics[key]), f"{kind}/{name}: {key} invalid"
        assert metrics["Coverage_GT"] == metrics["Coverage_Pred"] == 1
        assert metrics["Common_Count"] == result["common_count"] == count
        assert result["gt_count"] == result["pred_count"] == count
        assert result["missing_pred_files_count"] == result["unmatched_pred_files_count"] == 0
        assert set(result["skipped_metrics"]) == {key for key, value in metrics.items() if value is None}
        assert not set(REQUIRED[kind]).intersection(result["skipped_metrics"])
        config = result["config"]
        assert config["paired_size"] == 448 and config["batch_size"] == 1
        if continuous:
            assert metrics["Track_Count"] == 20 and metrics["Seq_Frame_Count"] == 1280
            assert config["fvd_size"] == 224 and config["clip_length"] == config["clip_stride"] == 16
            assert result["details"]["fvd_clip_count"] == 80
            tracks = result["details"]["tracks"]
            assert tracks["source"] == "benchmark_v2_manifest"
            assert tracks["manifest_clip_count"] == tracks["track_count"] == 20
            assert tracks["dropped_incomplete_clip_count"] == tracks["dropped_short_track_count"] == 0
            assert tracks["unparsed_common_files_count"] == 0
            assert tracks["parsed_frame_count"] == 1280
            assert len(tracks["track_summaries"]) == 20
            assert all(track["length"] == 64 for track in tracks["track_summaries"])
        per_map[name] = metrics
    summary = load(root / "summary.json")
    assert summary["maps"] == MAPS and summary["per_map"] == per_map
    assert summary["split"] == split and summary["kind"] == kind
    assert summary["sample_count"] == count * len(MAPS) and summary["inference_seed"] == 42
    assert (project / summary["checkpoint"]).resolve() == checkpoint
    assert summary["benchmark_v2_asset_backend"] == "minimal"
    assert summary["benchmark_v2_asset_manifest_sha256"] == asset_hash
    assert summary["benchmark_v2_selected_images_sha256"] == selected_hash
    macro = summary["metrics_macro_map"]
    for key in REQUIRED[kind]:
        assert finite(macro[key]), f"{kind}: missing macro {key}"
        expected_mean = sum(per_map[name][key] for name in MAPS) / len(MAPS)
        assert math.isclose(macro[key], expected_mean, rel_tol=1e-12, abs_tol=1e-12)
    print(json.dumps({"kind": kind, "status": "PASS", "maps": len(MAPS), "samples": count * len(MAPS), "required_metrics_macro": {key: macro[key] for key in REQUIRED[kind]}}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["discrete", "continuous", "both"], default="both")
    args = parser.parse_args()
    for kind in REQUIRED if args.kind == "both" else [args.kind]:
        audit(Path.cwd(), kind)
