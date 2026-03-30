import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    OptimizerNames,
    is_torch_hpu_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_xpu_available,
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    # ALL_LAYERNORM_LAYERS,
    logger,
)

ALL_LAYERNORM_LAYERS = [nn.LayerNorm]
from typing import Any, List, Optional, Union
from transformers.utils import is_torch_xla_available, is_peft_available
from transformers.trainer_utils import speed_metrics
from accelerate.utils import DistributedType
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
)
import importlib.metadata
from packaging import version

if is_peft_available():
    from peft import PeftModel

def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,)
        # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False



if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    from torch_xla import __version__ as XLA_VERSION

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(XLA_FSDPV2_MIN_VERSION)
    if IS_XLA_FSDPV2_POST_2_2:
        import torch_xla.distributed.spmd as xs
        import torch_xla.runtime as xr
else:
    IS_XLA_FSDPV2_POST_2_2 = False


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)

import math
import torch
import torch.distributed as dist
from torch.utils.data import Sampler, BatchSampler, Dataset, DataLoader
from typing import Iterator, Optional, List

class DistributedTaskTypeBatchSampler(BatchSampler):
    """
    一个支持分布式训练的BatchSampler，它首先按任务类型对数据进行分组，
    然后在每个epoch中，保证每个副本（GPU）拿到不重复的、随机打乱的批次。

    核心逻辑：
    1. 在所有副本上生成一个完全相同的、全局的批次列表（batches）。
    2. 根据副本数量（world_size）对这个全局批次列表进行填充或截断，使其长度能被整除。
    3. 每个副本（rank）根据自己的排名，从全局批次列表中切分出自己需要处理的部分。
    4. `set_epoch()` 方法确保每个epoch的随机种子都不同，从而实现不同的数据打乱顺序。
    """
    def __init__(self,
                 dataset: Dataset,
                 batch_size: int,
                 shuffle: bool = True,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 drop_last: bool = False):
        """
        Args:
            dataset (Dataset): 需要采样的数据集。
            batch_size (int): 每个批次的大小。
            shuffle (bool): 是否在每个 epoch 开始时打乱数据。
            num_replicas (int, optional): 分布式训练中的进程数。如果为None，则从 dist.get_world_size() 获取。
            rank (int, optional): 当前进程的排名。如果为None，则从 dist.get_rank() 获取。
            drop_last (bool): 如果为 True，则丢弃最后一个不完整的批次。在这里，它意味着丢弃无法被副本数整除的尾部批次。
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        self.epoch = 0

        # 与原始 TaskTypeBatchSampler 相同的逻辑，构建 type 到 indices 的映射
        type_list = dataset.list_data_dict["type"]
        self.type_to_indices = {}
        for idx, t in enumerate(type_list):
            self.type_to_indices.setdefault(t, []).append(idx)

        # 计算所有可能的完整批次
        self.total_batches = self._calculate_total_batches()

        # 计算每个副本的样本数（这里是批次数）
        if self.drop_last and self.total_batches % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            self.num_samples = math.ceil((self.total_batches - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(self.total_batches / self.num_replicas)

        self.total_size = self.num_samples * self.num_replicas

    def _calculate_total_batches(self) -> int:
        # 计算在 drop_last=True 的情况下，整个数据集可以产生多少个完整的批次
        return sum(len(indices) // self.batch_size for indices in self.type_to_indices.values())

    def __iter__(self) -> Iterator[List[int]]:
        # 1. 生成全局的批次列表 (在所有 rank 上都相同)
        g = torch.Generator()
        g.manual_seed(self.epoch)  # 使用 epoch 作为种子，确保每个 epoch 的 shuffle 不同但所有 rank 相同

        all_batches = []
        for t, indices in self.type_to_indices.items():
            # 为每个类型的索引列表创建副本进行操作
            idxs = list(indices)
            if self.shuffle:
                # 使用带种子的生成器进行打乱
                perm = torch.randperm(len(idxs), generator=g).tolist()
                idxs = [idxs[i] for i in perm]

            # 按 batch_size 切分，并只保留完整的批次
            for i in range(0, len(idxs) - self.batch_size + 1, self.batch_size):
                all_batches.append(idxs[i : i + self.batch_size])

        # 如果需要，再次打乱所有批次的顺序
        if self.shuffle:
            perm = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in perm]

        # 2. 填充或截断批次列表以适应分布式设置
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(all_batches)
            if padding_size > 0:
                all_batches += all_batches[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            all_batches = all_batches[:self.total_size]

        assert len(all_batches) == self.total_size

        # 3. 为当前 rank 切分出子集
        # subsample
        indices_on_this_rank = all_batches[self.rank : self.total_size : self.num_replicas]
        assert len(indices_on_this_rank) == self.num_samples

        return iter(indices_on_this_rank)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward

class NonMixTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize a dictionary to store the latest loss values
        # Since the model returns python floats (scalars), we just need to store the latest value
        self.latest_loss_info = {}
        self._step_timing = self._new_step_timing()

    def _new_step_timing(self):
        return {
            "step_start_time": None,
            "batch_load_time": 0.0,
            "prepare_inputs_time": 0.0,
            "forward_time": 0.0,
            "backward_time": 0.0,
        }

    def _timing_enabled(self):
        return getattr(self.args, "enable_step_timing", False)

    def _sync_timing_device(self):
        if not self._timing_enabled():
            return
        if not getattr(self.args, "step_timing_sync_cuda", True):
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _ensure_step_timer_started(self):
        if self._step_timing["step_start_time"] is None:
            self._step_timing["step_start_time"] = time.perf_counter()

    def consume_step_timing_metrics(self):
        if self._step_timing["step_start_time"] is None:
            return {}

        step_total_time = time.perf_counter() - self._step_timing["step_start_time"]
        metrics = {
            "step_total_time": step_total_time,
            "batch_load_time": self._step_timing["batch_load_time"],
            "prepare_inputs_time": self._step_timing["prepare_inputs_time"],
            "forward_time": self._step_timing["forward_time"],
            "backward_time": self._step_timing["backward_time"],
        }
        tracked_time = (
            metrics["batch_load_time"]
            + metrics["prepare_inputs_time"]
            + metrics["forward_time"]
            + metrics["backward_time"]
        )
        metrics["other_iteration_time"] = max(0.0, step_total_time - tracked_time)
        self._step_timing = self._new_step_timing()
        return metrics


    def get_train_dataloader(self):
        """
        重写此方法以使用我们自定义的分布式批次采样器。
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # 使用新的分布式采样器
        batch_sampler = DistributedTaskTypeBatchSampler(
            self.train_dataset,
            batch_size=self._train_batch_size, # 注意：使用 _train_batch_size (per_device)
            shuffle=True,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            drop_last=self.args.dataloader_drop_last,
        )

        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            persistent_workers=self.args.dataloader_persistent_workers,
            prefetch_factor=self.args.dataloader_prefetch_factor if self.args.dataloader_num_workers > 0 else None,
            pin_memory=self.args.dataloader_pin_memory,
            # Dataloader需要 set_epoch 方法，通过将其设置为 True 来自动调用
            # 但是，Hugging Face Trainer 会手动调用，所以这里可以不设置
        )

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        if not self._timing_enabled():
            return super().get_batch_samples(epoch_iterator, num_batches, device)

        self._ensure_step_timer_started()
        start_time = time.perf_counter()
        result = super().get_batch_samples(epoch_iterator, num_batches, device)
        self._step_timing["batch_load_time"] += time.perf_counter() - start_time
        return result

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """
        这个方法现在可以被 get_train_dataloader 覆盖，但为了保持完整性，
        我们可以让它返回 None，因为我们使用的是 batch_sampler。
        或者，如果你的逻辑在某些情况下回退到使用这个方法，
        你需要确保它不会与 get_train_dataloader 中的 batch_sampler 冲突。

        在当前实现中，由于我们重写了 get_train_dataloader，这个方法不会被调用来创建训练数据加载器。
        """
        # 返回 None，因为我们使用的是 batch_sampler
        return None

    def _attach_optimizer_managed_param_snapshot(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        named_params = list(model.named_parameters())
        param_id_to_name = {id(param): name for name, param in named_params}

        optimizer_param_names = []
        optimizer_param_missing = []
        seen = set()

        for group_idx, group in enumerate(optimizer.param_groups):
            for param_idx, param in enumerate(group["params"]):
                name = param_id_to_name.get(id(param))
                if name is None:
                    optimizer_param_missing.append(
                        f"<group={group_idx} param={param_idx} shape={tuple(param.shape)}>"
                    )
                    continue
                if name in seen:
                    continue
                seen.add(name)
                optimizer_param_names.append(name)

        candidate_targets = []
        for candidate in (
            model,
            getattr(model, "model", None),
            getattr(model, "base_model", None),
            getattr(getattr(model, "base_model", None), "model", None),
        ):
            if isinstance(candidate, nn.Module) and candidate not in candidate_targets:
                candidate_targets.append(candidate)

        for candidate in candidate_targets:
            setattr(candidate, "_optimizer_managed_param_names", optimizer_param_names)
            setattr(candidate, "_optimizer_managed_param_missing", optimizer_param_missing)

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            def get_action_head_parameter_names():
                parameter_names = {"action_dit_projector", "action_dit_norm"}
                if getattr(opt_model.config, "use_vit_cls_regression_head", False):
                    parameter_names.update({"regression_loc_head"})
                elif getattr(opt_model.config, "use_vit_regression_head", False):
                    parameter_names.update({"regression_loc_head", "cross_view_fusion"})
                elif getattr(opt_model.config, "use_codex_vit_regression_head", False):
                    parameter_names.update({"regression_loc_head", "vit_loc_fusion"})
                return [
                    name for name, _ in opt_model.named_parameters()
                    if any(key in name for key in parameter_names)
                ]

            if self.args.is_action_dit_projector and self.args.is_loc_learnable_query and self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                action_dit_projector_parameters = get_action_head_parameter_names()
                loc_learnable_query_parameters = [name for name, _ in opt_model.named_parameters() if "loc_learnable_query" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and n not in action_dit_projector_parameters and n not in loc_learnable_query_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and n not in action_dit_projector_parameters and n not in loc_learnable_query_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and n not in action_dit_projector_parameters and n not in loc_learnable_query_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and n not in action_dit_projector_parameters and n not in loc_learnable_query_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and n not in loc_learnable_query_parameters and n in action_dit_projector_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.action_dit_projector_lr,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and n not in loc_learnable_query_parameters and n in action_dit_projector_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                        "lr": self.args.action_dit_projector_lr,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and n in loc_learnable_query_parameters and n not in action_dit_projector_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.loc_learnable_query_lr,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and n in loc_learnable_query_parameters and n not in action_dit_projector_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                        "lr": self.args.loc_learnable_query_lr,
                    },
                ]
            elif self.args.is_action_dit_projector and self.args.is_loc_learnable_query:
                action_dit_projector_parameters = get_action_head_parameter_names()
                loc_learnable_query_parameters = [name for name, _ in opt_model.named_parameters() if "loc_learnable_query" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in loc_learnable_query_parameters and n not in action_dit_projector_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in loc_learnable_query_parameters and n not in action_dit_projector_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in loc_learnable_query_parameters and n not in action_dit_projector_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.loc_learnable_query_lr,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in loc_learnable_query_parameters and n not in action_dit_projector_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                        "lr": self.args.loc_learnable_query_lr,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in loc_learnable_query_parameters and n in action_dit_projector_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.action_dit_projector_lr,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in loc_learnable_query_parameters and n in action_dit_projector_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                        "lr": self.args.action_dit_projector_lr,
                    },
                ]
            elif self.args.is_loc_learnable_query and self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                loc_learnable_query_parameters = [name for name, _ in opt_model.named_parameters() if "loc_learnable_query" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in loc_learnable_query_parameters and n not in projector_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in loc_learnable_query_parameters and n not in projector_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in loc_learnable_query_parameters and n not in projector_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.loc_learnable_query_lr,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in loc_learnable_query_parameters and n not in projector_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                        "lr": self.args.loc_learnable_query_lr,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in loc_learnable_query_parameters and n in projector_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in loc_learnable_query_parameters and n in projector_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            elif self.args.is_action_dit_projector and self.args.mm_projector_lr is not None:
                action_dit_projector_parameters = get_action_head_parameter_names()
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and n not in action_dit_projector_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and n not in action_dit_projector_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and n not in action_dit_projector_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and n not in action_dit_projector_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and n in action_dit_projector_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.action_dit_projector_lr,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and n in action_dit_projector_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                        "lr": self.args.action_dit_projector_lr,
                    },
                ]
            elif self.args.is_loc_learnable_query:
                loc_learnable_query_parameters = [name for name, _ in opt_model.named_parameters() if "loc_learnable_query" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in loc_learnable_query_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in loc_learnable_query_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in loc_learnable_query_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.loc_learnable_query_lr,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in loc_learnable_query_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                        "lr": self.args.loc_learnable_query_lr,
                    },
                ]
            elif self.args.is_action_dit_projector:
                action_dit_projector_parameters = get_action_head_parameter_names()
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in action_dit_projector_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in action_dit_projector_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in action_dit_projector_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.action_dit_projector_lr,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in action_dit_projector_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                        "lr": self.args.action_dit_projector_lr,
                    },
                ]
            elif self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]

            # 会报错：重复加载parameters
            # if getattr(opt_model.config, 'use_vit_regression_head', False):
            #     reg_head_parameters = [name for name, _ in opt_model.named_parameters() if "regression_loc_head" in name]
            #     if self.args.action_dit_projector_lr:
            #         reg_head_lr = self.args.action_dit_projector_lr
            #     else:
            #         reg_head_lr = 0.001
            #     reg_optimizer_grouped_parameters = [
            #         {
            #             "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in reg_head_parameters and p.requires_grad)],
            #             "weight_decay": self.args.weight_decay,
            #             "lr": reg_head_lr,
            #         },
            #         {
            #             "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in reg_head_parameters and p.requires_grad)],
            #             "weight_decay": 0.0,
            #             "lr": reg_head_lr,
            #         },
            #     ]
            #     optimizer_grouped_parameters.extend(reg_optimizer_grouped_parameters)

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            optimizer_grouped_parameters = [
                group for group in optimizer_grouped_parameters
                if len(group["params"]) > 0
            ]
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            self._attach_optimizer_managed_param_snapshot(opt_model, self.optimizer)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self._timing_enabled():
            return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

        self._ensure_step_timer_started()

        cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)

        with cp_context():
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()

            prepare_start = time.perf_counter()
            inputs = self._prepare_inputs(inputs)
            self._step_timing["prepare_inputs_time"] += time.perf_counter() - prepare_start

            if is_sagemaker_mp_enabled():
                self._sync_timing_device()
                forward_start = time.perf_counter()
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                self._sync_timing_device()
                self._step_timing["forward_time"] += time.perf_counter() - forward_start
                return loss_mb.reduce_mean().detach().to(self.args.device)

            self._sync_timing_device()
            forward_start = time.perf_counter()
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
            self._sync_timing_device()
            self._step_timing["forward_time"] += time.perf_counter() - forward_start

            del inputs
            if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
            ):
                if is_torch_xpu_available():
                    torch.xpu.empty_cache()
                elif is_torch_mlu_available():
                    torch.mlu.empty_cache()
                elif is_torch_musa_available():
                    torch.musa.empty_cache()
                elif is_torch_npu_available():
                    torch.npu.empty_cache()
                elif is_torch_mps_available():
                    torch.mps.empty_cache()
                elif is_torch_hpu_available():
                    logger.warning(
                        "`torch_empty_cache_steps` is set but HPU device/backend does not support empty_cache()."
                    )
                else:
                    torch.cuda.empty_cache()

            kwargs = {}
            if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                kwargs["learning_rate"] = self._get_learning_rate()

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self._sync_timing_device()
            backward_start = time.perf_counter()
            if self.use_apex:
                from apex import amp

                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                if (
                    not self.model_accepts_loss_kwargs or num_items_in_batch is None
                ) and self.compute_loss_func is None:
                    loss = loss / self.current_gradient_accumulation_steps

                if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                    kwargs["scale_wrt_gas"] = False

                self.accelerator.backward(loss, **kwargs)
            self._sync_timing_device()
            self._step_timing["backward_time"] += time.perf_counter() - backward_start

            return loss.detach()

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        """
        Overridden to capture 'extras' from the model output.
        """
        # --- 1. Prepare inputs (Standard Trainer logic) ---
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        if self.model_accepts_loss_kwargs:
            kwargs = {}
            if num_items_in_batch is not None:
                kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **kwargs}

        # --- 2. Forward pass ---
        outputs = model(**inputs)

        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # --- 3. [CUSTOM] Capture extra loss info ---
        # We capture the 'extras' attribute attached in your model's forward
        if hasattr(outputs, "extras") and "other_info" in outputs.extras:
            # We store it in self.latest_loss_info to be used later in log()
            # Since you already detached and converted to item() in the model, we just copy it.
            self.latest_loss_info = outputs.extras["other_info"]

        # --- 4. Compute final scalar loss (Standard Trainer logic) ---
        if self.compute_loss_func is not None:
            if labels is None:
                logger.warning(
                    "Trainer: `compute_loss_func` is defined but `labels=None`. "
                    "Your custom loss function will still be called with labels=None. "
                )
            loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
        elif labels is not None:
            # Handle label smoothing / PEFT unwrapping
            unwrapped_model = self.accelerator.unwrap_model(model)
            model_name = (
                unwrapped_model.base_model.model._get_name()
                if _is_peft_model(unwrapped_model)
                else unwrapped_model._get_name()
            )
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            # Standard causal LM loss extraction
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # Handle average tokens across devices
        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time=None) -> None:
        """
        Overridden to inject 'other_info' into the logs dict.
        """
        # --- 1. [CUSTOM] Inject stored loss info ---
        if self.latest_loss_info:
            # Add the 'other_info' dictionary to the logs
            # Your Callback expects 'other_info' key
            logs["other_info"] = self.latest_loss_info

            # Optional: Clear it to avoid stale data (though typically overwritten next step)
            # self.latest_loss_info = {}

        # --- 2. Standard Trainer logging logic ---
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch

        if self.args.include_num_input_tokens_seen != "no":
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
            if start_time is not None:
                logs.update(speed_metrics("train", start_time, num_tokens=self.state.num_input_tokens_seen))

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)

        # Triggers all Callbacks.on_log
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)




