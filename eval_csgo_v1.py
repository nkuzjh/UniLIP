import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml

from benchmark_csgo_v1 import build_arg_parser as build_benchmark_arg_parser
from benchmark_csgo_v1 import run_benchmark_v1
from eval_csgo import run_csgo_generation_from_config


def resolve_eval_maps(csgo_config: Dict[str, object]) -> List[str]:
    map_names = csgo_config.get("val_maps") or csgo_config.get("test_maps")
    if not map_names:
        raise ValueError("No val_maps or test_maps found in csgo config.")
    return list(map_names)


def build_benchmark_args(csgo_config: Dict[str, object], output_dir: str, map_name: str) -> argparse.Namespace:
    data_dir = csgo_config.get("data_dir", "data/preprocessed_data")
    gt_dir = Path(data_dir) / map_name / "imgs"
    pred_dir = Path(output_dir) / "gen_imgs" / map_name

    parser = build_benchmark_arg_parser()
    args = parser.parse_args([
        "--gt", str(gt_dir),
        "--pred", str(pred_dir),
        "--data_dir", str(data_dir),
        "--map_name", map_name,
    ])

    yaml_to_benchmark_arg = {
        "benchmark_batch_size": "batch_size",
        "benchmark_device": "device",
        "benchmark_paired_size": "paired_size",
        "benchmark_edge_quantile": "edge_quantile",
        "benchmark_edge_tolerance": "edge_tolerance",
        "benchmark_pose_json": "pose_json",
        "benchmark_external_loc_repo_root": "external_loc_repo_root",
        "benchmark_external_loc_config_path": "external_loc_config_path",
        "benchmark_external_loc_checkpoint_path": "external_loc_checkpoint_path",
    }
    for yaml_key, arg_name in yaml_to_benchmark_arg.items():
        if yaml_key in csgo_config:
            setattr(args, arg_name, csgo_config[yaml_key])

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    return args


def run_eval_csgo_v1(csgo_config_path: str, *, output_dir: Optional[str] = None, seed: int = 42) -> Dict[str, object]:
    with open(csgo_config_path, "r", encoding="utf-8") as f:
        csgo_config = yaml.safe_load(f)

    if csgo_config.get("is_conti_gen", False):
        raise ValueError("eval_csgo_v1.py only supports discrete test frames. Use eval_csgo.py for gen_conti configs.")

    generation_result = run_csgo_generation_from_config(
        csgo_config_path,
        output_dir=output_dir,
        seed=seed,
        release_model=True,
    )
    csgo_config = generation_result["csgo_config"]

    benchmark_results = []
    if csgo_config.get("run_benchmark_v1", True):
        for map_name in resolve_eval_maps(csgo_config):
            benchmark_args = build_benchmark_args(csgo_config, generation_result["output_dir"], map_name)
            benchmark_results.append(run_benchmark_v1(benchmark_args))
    else:
        print("run_benchmark_v1=False, skip benchmark_csgo_v1 metrics.")

    return {
        **generation_result,
        "benchmark_results": benchmark_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Run CSGO discrete-frame generation and benchmark v1.")
    parser.add_argument("--csgo_config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = run_eval_csgo_v1(args.csgo_config, output_dir=args.output_dir, seed=args.seed)
    if result["benchmark_results"]:
        print("Benchmark JSON files:")
        for item in result["benchmark_results"]:
            print(f"  {item['map_name']}: {item['json_path']}")


if __name__ == "__main__":
    main()
