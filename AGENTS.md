# AGENTS.md

## 0. Mission

This repository is an AI / Python research codebase.

Your role is to act as a careful research engineer:
- build accurate understanding before acting
- design clean experiments
- implement minimal, controlled changes
- avoid unsupported assumptions

This file defines persistent project-level rules shared across all Codex threads.

---

## 1. Global Working Principles

### 1.1 Order of operations
Always follow:
1. Analyze
2. Plan
3. Implement (minimal)
4. Verify (if possible)
5. Report

### 1.2 Evidence over assumptions
- Ground every claim in actual repository code.
- Explicitly label:
  - confirmed facts
  - plausible hypotheses
  - unknowns

### 1.3 Minimality
- Prefer smallest valid change.
- Avoid broad refactors unless explicitly requested.
- Do not mix:
  - experiment logic
  - refactoring
  - bug fixing

### 1.4 Reproducibility mindset
- Never assume a result exists unless tied to a concrete script/config.
- Never claim “works” or “fixed” without execution evidence.

---

## 2. Repository Understanding Protocol

When analyzing the repository, you MUST:

### 2.1 Identify structure
- top-level directories
- key modules
- dependency boundaries

### 2.2 Identify entry points
Locate scripts or functions for:
- training
- evaluation
- inference
- data preprocessing

### 2.3 Trace full pipeline
Follow end-to-end flow:

- config loading
- argument parsing
- dataset construction
- dataloader
- model construction
- loss construction
- optimizer / scheduler
- training loop
- checkpointing
- evaluation pipeline

### 2.4 Experiment control surface
Identify where behavior is controlled:
- config files (yaml/json)
- CLI args
- shell scripts
- launch scripts
- notebooks

### 2.5 Baseline reconstruction
Determine:
- canonical training command
- active losses
- dataset splits
- evaluation metrics
- output artifacts

---

## 3. Experiment Design Rules

### 3.1 Always define
For any experiment:
- hypothesis
- baseline
- changed variable
- controlled variables
- evaluation protocol

### 3.2 Factor separation
Separate:
- architecture changes
- loss changes
- data changes
- optimization changes
- inference changes

### 3.3 Ablation discipline
- Prefer one-factor-at-a-time
- Rank:
  - must-run
  - high-value
  - optional

### 3.4 Confounder control
Explicitly consider:
- data leakage
- metric mismatch
- seed variance
- hidden coupling in config

---

## 4. Implementation Rules

### 4.1 Minimal edits
- Modify the fewest files possible.
- Keep diffs localized.

### 4.2 Preserve baseline
- Default behavior must remain unchanged.
- New behavior should be gated by config/flags.

### 4.3 Reuse abstractions
- Do not duplicate logic.
- Integrate with existing pipelines.

### 4.4 Interface stability
- Avoid breaking:
  - checkpoints
  - config schema
  - public APIs

---

## 5. Evaluation Rules

### 5.1 Metric integrity
- Identify exact metric implementation.
- Do not rely on naming assumptions.

### 5.2 Evaluation path
- Distinguish:
  - training-time metrics
  - validation metrics
  - test metrics

### 5.3 Reporting discipline
Always specify:
- metric name
- where computed
- dataset split

---

## 6. Debugging Protocol

When debugging:

1. Localize failure
2. Narrow to smallest component
3. Rank root causes:
   - confirmed
   - likely
   - speculative
4. Propose minimal verification steps
5. Only then propose fixes

Special attention to:
- gradient flow
- loss routing
- masking logic
- device / dtype issues

---

## 7. Multi-Thread Development Protocol

This repository is expected to be developed across multiple Codex threads.

Therefore:

### 7.1 No hidden context
- Each response must be self-contained.
- Do not rely on prior thread memory.

### 7.2 Scope clarity
Always restate:
- current objective
- boundaries

### 7.3 Task isolation
Each thread should focus on ONE:
- analysis
- experiment design
- implementation
- debugging

### 7.4 Handoff requirement
At the end of each task, produce:

Handoff Summary:
- Objective
- Confirmed findings
- Files inspected
- Files modified
- What remains
- Next step
- Risks

---

## 8. Output Format Standard

Unless otherwise specified:

1. Objective
2. Findings
3. Uncertainties
4. Proposed next step
5. Relevant files

Keep responses:
- compact
- structured
- high-signal

---

## 9. Prohibitions

- Do NOT fabricate behavior not present in code
- Do NOT claim validation without execution
- Do NOT mix unrelated changes
- Do NOT rewrite large subsystems without request
- Do NOT silently change experiment semantics

---

## 10. Preferred Scientific Workflow

Baseline
→ Reproduce
→ Instrument
→ Hypothesis
→ Minimal change
→ Run
→ Evaluate
→ Ablate
→ Summarize

---

## 11. Project-Specific Notes (To Be Filled)

### 11.1 Intended task
- Primary documented task: unified multimodal understanding, image generation, image editing, and image reconstruction for UniLIP (`README.md`, `TRAIN.md`, `EVAL.md`, `tokenizer/README_RECON.md`).
- Additional task present in this checkout: a CS2/CS:GO-specific extension for first-person-view generation from radar + pose, and camera-pose localization from FPS + radar (`train_csgo.py`, `eval_csgo.py`, `eval_csgo_loc.py`).
- Understanding benchmark evaluation is present, but its metric/runtime integration depends on external VLMEvalKit (`tokenizer/README_RECON.md`, `tokenizer/inference.py`).

### 11.2 Primary entry points
- Training:
  - Official UniLIP generation/editing:
    - `unilip/train/train_stage1.py::train`
    - `unilip/train/train_stage2.py::train`
    - `unilip/train/train_stage3.py::train`
    - Launchers: `scripts/run_unilip_{1b,2b}_stage{1,2,3}.sh`
  - Reconstruction/tokenizer:
    - `tokenizer/scripts/train_stage1.py::main`
    - `tokenizer/scripts/train_stage2.py::main`
    - Launchers: `tokenizer/{1b,2b}_stage{1,2,2_448}.sh`
  - CS2/CS:GO extension:
    - `train_csgo.py::train`
    - Local launcher: `run_unilip_1b_csgo.sh`
- Evaluation:
  - Official generation/editing benchmark drivers:
    - `eval/geneval/geneval.py::main`
    - `eval/WISE/wise.py::main`
    - `eval/ImgEdit/imgedit.py::main`
    - Launchers: `eval/*/*_{1b,2b}.sh`
  - Reconstruction:
    - `tokenizer/scripts/evaluation.py::main`
    - Launcher: `tokenizer/{1b,2b}_test.sh`
  - CS2/CS:GO:
    - `eval_csgo.py::main`
    - `eval_csgo_loc.py::main`
    - `benchmark_csgo.py::main`
    - `benchmark_csgo_video.py::main`
- Inference:
  - Official demos:
    - `scripts/inference_gen.py`
    - `scripts/inference_edit.py`
  - Reconstruction demo:
    - `tokenizer/scripts/inference.py::main`
  - Lower-level inference wrappers:
    - `unilip/pipeline_gen.py::CustomGenPipeline`
    - `unilip/pipeline_edit.py::CustomEditPipeline`

### 11.3 Baseline command
- Most defensible official training pipeline from repository docs (`TRAIN.md`):
  - `cd scripts && bash run_unilip_1b_stage1.sh`
  - `cd scripts && bash run_unilip_1b_stage2.sh`
  - `cd scripts && bash run_unilip_1b_stage3.sh`
  - Rationale: this is the only end-to-end training sequence explicitly documented as the main UniLIP workflow.
- Most defensible reconstruction/tokenizer baseline (`tokenizer/README_RECON.md`):
  - `cd tokenizer && bash 1b_stage1.sh`
  - `cd tokenizer && bash 1b_stage2.sh`
  - `cd tokenizer && bash 1b_stage2_448.sh`
- CS2/CS:GO baseline command: uncertain.
  - Evidence: `train_csgo.py` is the entry point and `run_unilip_1b_csgo.sh` exists, but many `csgo_configs/*.yaml` variants are present and no single canonical config is documented in `README.md`/`TRAIN.md`.

### 11.4 Key config files
- Official UniLIP generation/editing training is primarily controlled by shell launchers plus CLI flags:
  - `scripts/run_unilip_{1b,2b}_stage{1,2,3}.sh`
- Reconstruction/tokenizer configs:
  - `tokenizer/configs/training/InternVL3_1B_DCAE/*.yaml`
  - `tokenizer/configs/training/InternVL3_2B_DCAE/*.yaml`
- CS2/CS:GO configs:
  - `csgo_configs/*.yaml`
  - `csgo_configs/test/*.yaml`
- DeepSpeed configs:
  - `deepspeed_scripts/zero{0,1,2,3}.json`
  - `deepspeed_scripts/zero3_offload.json`
- Evaluation launch configs:
  - `eval/geneval/*_{1b,2b}.sh`
  - `eval/WISE/*_{1b,2b}.sh`
  - `eval/ImgEdit/*_{1b,2b}.sh`

### 11.5 Core model modules
- Main UniLIP language-model wrappers:
  - `unilip/model/language_model/unilip_internvl.py::UniLIP_InternVLForCausalLM`
  - `unilip/model/language_model/unilip_vae_internvl.py::UniLIP_VAE_InternVLForCausalLM`
  - `unilip/model/language_model/unified_unilip.py::Unified_UniLIP_InternVLForCausalLM`
- Supporting UniLIP model code:
  - `unilip/model/unilip_internvl.py`
  - `unilip/model/unilip_vae_internvl.py`
  - `unilip/model/builder.py::load_pretrained_model_general`
  - `unilip/model/sana.py`
  - `unilip/model/vae_modules.py`
- Training wrapper:
  - `unilip/train/nonmix_trainer.py::NonMixTrainer`
- Reconstruction/tokenizer models:
  - `tokenizer/modeling/dc_ae_vit.py::DC_AE_ViT`
  - `tokenizer/modeling/dc_ae_vit_stage2.py::DC_AE_ViT_Stage2`
  - `tokenizer/modeling/dc_ae_vit_stage2_448.py::DC_AE_ViT_Stage2_448`
- CS2/CS:GO data/model integration:
  - `csgo_datasets/unified_task_dataset.py`
  - `unilip/model/external_loc_model_loader.py`

### 11.6 Loss structure
- Official UniLIP generation/editing loss:
  - `unilip/model/language_model/unilip_internvl.py`
  - `unilip/model/language_model/unilip_vae_internvl.py`
  - Major term: diffusion-style image loss using `torch.nn.MSELoss()` between DiT noise prediction and target `noise - latents`.
- CS2/CS:GO multitask loss:
  - `unilip/model/language_model/unified_unilip.py`
  - Major terms:
    - `masked_gen_loss`: generation branch loss
    - `masked_loc_loss`: localization branch loss
    - `masked_loc_aux_loss`: optional auxiliary localization-on-generated-image loss
    - Total loss: `gen_loss + alpha_loc_loss * loc_loss + alpha_loc_aux_loss * loc_aux_loss`
  - Localization loss variants in code:
    - plain `MSELoss`
    - `_compute_codex_loc_regression_loss(...)`, which uses SmoothL1 on XY/Z and either circular or direct angle loss
    - action-DiT localization path is implemented inside `Unified_UniLIP_InternVLForCausalLM.forward`; exact formula should be read there directly
- Reconstruction/tokenizer losses:
  - Instantiated in `tokenizer/utils/train_utils_stage1.py::create_model_and_loss_module` and `tokenizer/utils/train_utils_stage2.py::create_model_and_loss_module`
  - Implemented in `tokenizer/modeling/modules/losses.py`
  - Major terms:
    - reconstruction loss (`l1` or `l2`)
    - perceptual loss
    - quantizer loss
    - GAN generator/discriminator loss
    - LeCam regularization
    - VAE KL term (stage 1 when `quantize_mode == "vae"`)
    - distillation loss (stage 2 only)
  - Loss classes:
    - `ReconstructionLoss_Stage1`
    - `ReconstructionLoss_Stage2`

### 11.7 Metrics
- Reconstruction metrics implemented in-repo:
  - `tokenizer/evaluator/evaluator.py`
  - Metrics:
    - `InceptionScore`
    - `rFID`
    - `psnr`
    - `ssim`
    - optional `CodebookUsage`
    - optional `CodebookEntropy`
- CS2/CS:GO localization metrics implemented in-repo:
  - `eval_csgo_loc.py::calculate_metrics`
  - Metrics:
    - normalized-space: `Norm_L2_XY`, `Norm_L2_5D`, `Norm_MSE_5D`, `Norm_SmoothL1_5D`
    - physical-space: `XY_Dist`, `Z_Dist`, `Pitch_Dist`, `Yaw_Dist`, `L1_*`, `L2_*`, `MSE_Loss_5D`, `SmoothL1_Loss_5D`
    - optional train-aligned summaries: `TrainAligned_LocLoss`, `TrainAligned_TotalLoss`
- CS2/CS:GO image/video post-hoc metrics:
  - `benchmark_csgo.py`
    - `FID`, `LPIPS`, `PSNR`, `SSIM`, `CLIP`, `Aesthetic`
  - `benchmark_csgo_video.py`
    - adds `FrechetVideoDistance`
    - optional `DreamSim` if installed
- Official GenEval / WISE / ImgEdit benchmark scripts:
  - `eval/geneval/geneval.py`
  - `eval/WISE/wise.py`
  - `eval/ImgEdit/imgedit.py`
  - These scripts clearly generate benchmark outputs, but final benchmark scoring logic is uncertain / not fully implemented in this repository.

### 11.8 Dataset & splits
- Official training datasets documented in `TRAIN.md`:
  - Generation pretraining:
    - `BLIP3o-Pretrain-Long-Caption`
    - `BLIP3o-Pretrain-Short-Caption`
    - `BLIP3o-Pretrain-JourneyDB`
  - Editing pretraining:
    - `GPT-Image-Edit-1.5M`
  - SFT:
    - `BLIP3o-60k`
    - `ShareGPT-4o-Image`
- Reconstruction/tokenizer dataset definitions:
  - Train shards and eval shards are defined in YAML, e.g.
    - `tokenizer/configs/training/InternVL3_1B_DCAE/internvl3_1B_stage1.yaml`
    - `tokenizer/configs/training/InternVL3_1B_DCAE/internvl3_1B_stage2.yaml`
  - Train:
    - `train_shards_path_or_url`
  - Eval:
    - `eval_shards_path_or_url` (ImageNet val shards)
- Official benchmark datasets:
  - GenEval prompts from `eval/geneval/geneval_prompt.jsonl`
  - WISE JSON files loaded from `data/WISE/data/*.json`
  - ImgEdit single-turn benchmark loaded from `data/ImgEdit/Benchmark/singleturn/singleturn.json`
- Understanding benchmark datasets:
  - handled through external VLMEvalKit; dataset definitions/splits are outside this repo (`tokenizer/README_RECON.md`, `tokenizer/inference.py`)
- CS2/CS:GO datasets and splits:
  - Defined by YAML keys in `csgo_configs/*.yaml`
  - Dataset root usually `data/preprocessed_data`
  - Per-map split files:
    - `splits_20000_5000/train_split.json`
    - `splits_20000_5000/test_split.json`
    - `splits_20000_5000/continuous_unseen_clips.json`
  - Train maps / val maps / test maps are specified by config keys:
    - `train_maps`
    - `val_maps`
    - `test_maps`
  - In `train_csgo.py`, HF `eval_dataset` is `None`; validation is handled by separate eval scripts, not by the trainer loop.

### 11.9 Known quirks / risks
- The repository mixes at least three partially separate workflows:
  - official UniLIP generation/editing
  - reconstruction/tokenizer training
  - local CS2/CS:GO multitask experiments
- There is no single repo-wide config system:
  - official UniLIP relies mainly on shell scripts + CLI args
  - tokenizer relies on OmegaConf YAML
  - CS2/CS:GO relies on YAML plus dataclass CLI args
- `train_csgo.py` explicitly overrides some parsed training args in code:
  - `training_args.deepspeed = "deepspeed_scripts/zero0.json"`
  - `training_args.lr_scheduler_kwargs = {"min_lr": 1e-5}`
- `unilip/model/builder.py::load_pretrained_model_general` hardcodes `.to('cuda:0')`; this is a non-obvious device assumption during inference loading.
- `setup_transformers.py` copies local replacements into the installed `transformers` package; runtime behavior may depend on whether this patch has been applied.
- The official benchmark driver scripts generate outputs, but exact final scoring for GenEval / WISE / ImgEdit is uncertain from in-repo code alone.
- CS2/CS:GO baseline selection is ambiguous:
  - many `csgo_configs/*.yaml` variants exist
  - no single canonical experiment is documented in `README.md`/`TRAIN.md`
- CS2 debug behavior can switch training data to test splits (`csgo_datasets/unified_task_dataset.py`, `train_csgo.py` path-specific dataset code); treat debug configs carefully.
- Understanding evaluation is not standalone in this repo; it depends on external VLMEvalKit file replacement (`tokenizer/README_RECON.md`, `tokenizer/inference.py`).
