import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 1. Utils
# =========================================================

def set_requires_grad(module: nn.Module, flag: bool) -> None:
    """Enable or disable gradients for all parameters in a module."""
    for p in module.parameters():
        p.requires_grad_(flag)


def zero_all_grads(model: nn.Module) -> None:
    """Clear all parameter grads safely."""
    for p in model.parameters():
        p.grad = None


def grad_norm(t: Optional[torch.Tensor]) -> Optional[float]:
    """Return L2 norm of a gradient tensor, or None."""
    if t is None:
        return None
    return float(t.norm().item())


def tensor_abs_sum(t: Optional[torch.Tensor]) -> Optional[float]:
    """Return abs-sum of a tensor, or None."""
    if t is None:
        return None
    return float(t.abs().sum().item())


# =========================================================
# 2. Toy unified multimodal model
#    This is only for debugging template demonstration.
# =========================================================

class TextModule(nn.Module):
    """
    Simulates an image->text model:
    image_feature -> hidden -> logits
    """
    def __init__(self, image_dim: int, hidden_dim: int, vocab_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, vocab_dim),
        )

    def forward(self, image_feat: torch.Tensor) -> torch.Tensor:
        return self.net(image_feat)


class ImageModule(nn.Module):
    """
    Simulates a text->image model:
    text_feature -> hidden -> generated_image_feature
    """
    def __init__(self, text_dim: int, hidden_dim: int, image_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, image_dim),
        )

    def forward(self, text_feat: torch.Tensor) -> torch.Tensor:
        return self.net(text_feat)


class UnifiedModel(nn.Module):
    def __init__(self, image_dim: int, text_dim: int, hidden_dim: int, vocab_dim: int) -> None:
        super().__init__()
        self.text_module = TextModule(image_dim=image_dim, hidden_dim=hidden_dim, vocab_dim=vocab_dim)
        self.image_module = ImageModule(text_dim=text_dim, hidden_dim=hidden_dim, image_dim=image_dim)

    def forward_image_to_text(self, real_image_feat: torch.Tensor) -> torch.Tensor:
        return self.text_module(real_image_feat)

    def forward_text_to_image(self, real_text_feat: torch.Tensor) -> torch.Tensor:
        return self.image_module(real_text_feat)

    def forward_generated_image_to_text(self, generated_image_feat: torch.Tensor) -> torch.Tensor:
        return self.text_module(generated_image_feat)


# =========================================================
# 3. Loss computation matching your routing design
# =========================================================

@dataclass
class ForwardOutputs:
    loss_txt: torch.Tensor
    loss_img: torch.Tensor
    loss_cons: torch.Tensor
    generated_image: torch.Tensor
    text_logits_real: torch.Tensor
    text_logits_cons: torch.Tensor


def build_losses_for_debug(
    model: UnifiedModel,
    real_image_feat: torch.Tensor,
    real_text_feat: torch.Tensor,
    target_text_ids: torch.Tensor,
    target_image_feat: torch.Tensor,
) -> ForwardOutputs:
    """
    Build three losses in one forward pass:
      - loss_txt: image -> text
      - loss_img: text -> image
      - loss_cons: text -> image_hat -> text
        only intended to update image_module, not text_module
    """
    # -----------------------------
    # 1) Main text task: image -> text
    # -----------------------------
    set_requires_grad(model.text_module, True)
    model.text_module.train()

    text_logits_real = model.forward_image_to_text(real_image_feat)
    loss_txt = F.cross_entropy(text_logits_real, target_text_ids)

    # -----------------------------
    # 2) Main image task: text -> image
    # -----------------------------
    set_requires_grad(model.image_module, True)
    model.image_module.train()

    generated_image = model.forward_text_to_image(real_text_feat)
    loss_img = F.mse_loss(generated_image, target_image_feat)

    # -----------------------------
    # 3) Consistency task:
    #    text -> image_hat -> frozen text_module -> text loss
    # -----------------------------
    # Important:
    #   - freeze text_module params
    #   - use eval mode for stable behavior
    #   - DO NOT use torch.no_grad()
    # -----------------------------
    set_requires_grad(model.text_module, False)
    model.text_module.eval()

    # Keep gradient on intermediate tensor for inspection
    generated_image.retain_grad()

    text_logits_cons = model.forward_generated_image_to_text(generated_image)
    loss_cons = F.cross_entropy(text_logits_cons, target_text_ids)

    # Restore for future forwards if needed
    set_requires_grad(model.text_module, True)
    model.text_module.train()

    return ForwardOutputs(
        loss_txt=loss_txt,
        loss_img=loss_img,
        loss_cons=loss_cons,
        generated_image=generated_image,
        text_logits_real=text_logits_real,
        text_logits_cons=text_logits_cons,
    )


# =========================================================
# 4. Debug helpers
# =========================================================

def inspect_loss_routes_with_autograd_grad(model, loss_dict):
    # 只传当前 requires_grad=True 的参数给 autograd.grad
    named_params_req = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    req_names = [n for n, _ in named_params_req]
    req_params = [p for _, p in named_params_req]

    # 全量参数名用于最终展示
    all_named_params = [(n, p) for n, p in model.named_parameters()]
    all_names = [n for n, _ in all_named_params]

    results = {}

    for loss_name, loss in loss_dict.items():
        # loss 本身如果不需要 grad，也直接跳过
        if loss is None or (not loss.requires_grad):
            results[loss_name] = [(name, "LOSS_NO_GRAD", None) for name in all_names]
            continue

        grads = torch.autograd.grad(
            loss,
            req_params,
            retain_graph=True,
            allow_unused=True,
        )

        grad_map = {}
        for name, g in zip(req_names, grads):
            if g is None:
                grad_map[name] = ("NONE", None)
            else:
                s = float(g.abs().sum().item())
                if s == 0.0:
                    grad_map[name] = ("ZERO", 0.0)
                else:
                    grad_map[name] = ("NONZERO", s)

        per_loss = []
        for name, p in all_named_params:
            if not p.requires_grad:
                per_loss.append((name, "FROZEN", None))
            else:
                per_loss.append((name, *grad_map.get(name, ("NONE", None))))

        results[loss_name] = per_loss

    return results


def print_loss_route_summary(results: Dict[str, List[Tuple[str, str, Optional[float]]]]) -> None:
    """Pretty-print which parameters are affected by each loss."""
    print("\n" + "=" * 80)
    print("LOSS ROUTE SUMMARY")
    print("=" * 80)

    for loss_name, items in results.items():
        print(f"\n[{loss_name}]")
        for name, status, value in items:
            if status == "NONZERO":
                print(f"  {name:<50s} -> {status:<8s} abs_sum={value:.6f}")
            elif status == "ZERO":
                print(f"  {name:<50s} -> {status:<8s} abs_sum=0.0")
            else:
                print(f"  {name:<50s} -> {status}")


def run_single_backward_and_print_param_grads(
    model: nn.Module,
    total_loss: torch.Tensor,
) -> None:
    """
    Run a single backward on summed losses and print final param.grad stats.
    This shows the accumulated gradient after all losses are combined.
    """
    zero_all_grads(model)
    total_loss.backward()

    print("\n" + "=" * 80)
    print("FINAL PARAM.GRAD AFTER total_loss.backward()")
    print("=" * 80)

    for name, p in model.named_parameters():
        gnorm = grad_norm(p.grad)
        gsum = tensor_abs_sum(p.grad)
        print(f"{name:<50s} grad_norm={gnorm} abs_sum={gsum}")


def register_param_hooks_for_debug(model: nn.Module) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Register backward hooks on parameters to print gradient norm as soon as
    gradients pass through them. Useful for tracing actual backward flow.
    """
    handles: List[torch.utils.hooks.RemovableHandle] = []

    print("\n" + "=" * 80)
    print("REGISTERING PARAMETER HOOKS")
    print("=" * 80)

    for name, p in model.named_parameters():
        def _make_hook(param_name: str):
            def _hook(grad: torch.Tensor) -> None:
                print(f"[HOOK] {param_name:<50s} grad_norm={grad.norm().item():.6f}")
            return _hook

        handles.append(p.register_hook(_make_hook(name)))

    return handles


def inspect_intermediate_tensor_grad(
    model: nn.Module,
    outputs: ForwardOutputs,
    total_loss: torch.Tensor,
) -> None:
    """
    Show whether the intermediate generated_image receives gradient.
    This is critical for checking whether loss_cons can flow back to image_module.
    """
    zero_all_grads(model)
    if outputs.generated_image.grad is not None:
        outputs.generated_image.grad = None

    total_loss.backward()

    print("\n" + "=" * 80)
    print("INTERMEDIATE TENSOR GRAD CHECK")
    print("=" * 80)
    print("generated_image.grad norm:", grad_norm(outputs.generated_image.grad))
    print("generated_image.grad abs_sum:", tensor_abs_sum(outputs.generated_image.grad))


def print_module_level_grad_summary(model: nn.Module) -> None:
    """
    Aggregate parameter grad norms by top-level module prefix.
    Example:
      text_module.*
      image_module.*
    """
    summary: Dict[str, float] = {}

    for name, p in model.named_parameters():
        prefix = name.split(".")[0]
        val = 0.0 if p.grad is None else float(p.grad.norm().item())
        summary[prefix] = summary.get(prefix, 0.0) + val

    print("\n" + "=" * 80)
    print("MODULE-LEVEL GRAD SUMMARY")
    print("=" * 80)
    for module_name, total_norm in summary.items():
        print(f"{module_name:<20s} total_param_grad_norm_sum={total_norm:.6f}")


# =========================================================
# 5. Demo main
# =========================================================

def main() -> None:
    torch.manual_seed(42)

    # -----------------------------
    # Hyperparameters for toy setup
    # -----------------------------
    batch_size = 4
    image_dim = 16
    text_dim = 12
    hidden_dim = 32
    vocab_dim = 10

    lambda_img = 1.0
    lambda_cons = 0.7

    # -----------------------------
    # Model / optimizer
    # -----------------------------
    model = UnifiedModel(
        image_dim=image_dim,
        text_dim=text_dim,
        hidden_dim=hidden_dim,
        vocab_dim=vocab_dim,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # -----------------------------
    # Fake inputs / targets
    # -----------------------------
    real_image_feat = torch.randn(batch_size, image_dim)
    real_text_feat = torch.randn(batch_size, text_dim)
    target_text_ids = torch.randint(low=0, high=vocab_dim, size=(batch_size,))
    target_image_feat = torch.randn(batch_size, image_dim)

    # -----------------------------
    # Build losses
    # -----------------------------
    outputs = build_losses_for_debug(
        model=model,
        real_image_feat=real_image_feat,
        real_text_feat=real_text_feat,
        target_text_ids=target_text_ids,
        target_image_feat=target_image_feat,
    )

    total_loss = outputs.loss_txt + lambda_img * outputs.loss_img + lambda_cons * outputs.loss_cons

    print("\nLoss values:")
    print(f"  loss_txt  = {outputs.loss_txt.item():.6f}")
    print(f"  loss_img  = {outputs.loss_img.item():.6f}")
    print(f"  loss_cons = {outputs.loss_cons.item():.6f}")
    print(f"  total     = {total_loss.item():.6f}")

    # =====================================================
    # A. Inspect exact routes for each loss using autograd.grad
    # =====================================================
    route_results = inspect_loss_routes_with_autograd_grad(
        model=model,
        loss_dict={
            "loss_txt": outputs.loss_txt,
            "loss_img": outputs.loss_img,
            "loss_cons": outputs.loss_cons,
        },
    )
    print_loss_route_summary(route_results)

    # =====================================================
    # B. Show final accumulated param.grad after total backward
    # =====================================================
    run_single_backward_and_print_param_grads(model, total_loss)
    print_module_level_grad_summary(model)

    # =====================================================
    # C. Check intermediate tensor gradient
    # =====================================================
    # Need to rebuild forward, because previous backward freed the graph.
    outputs2 = build_losses_for_debug(
        model=model,
        real_image_feat=real_image_feat,
        real_text_feat=real_text_feat,
        target_text_ids=target_text_ids,
        target_image_feat=target_image_feat,
    )
    total_loss2 = outputs2.loss_txt + lambda_img * outputs2.loss_img + lambda_cons * outputs2.loss_cons
    inspect_intermediate_tensor_grad(model, outputs2, total_loss2)

    # =====================================================
    # D. Optional: parameter hooks during backward
    # =====================================================
    outputs3 = build_losses_for_debug(
        model=model,
        real_image_feat=real_image_feat,
        real_text_feat=real_text_feat,
        target_text_ids=target_text_ids,
        target_image_feat=target_image_feat,
    )
    total_loss3 = outputs3.loss_txt + lambda_img * outputs3.loss_img + lambda_cons * outputs3.loss_cons

    zero_all_grads(model)
    handles = register_param_hooks_for_debug(model)
    total_loss3.backward()
    for h in handles:
        h.remove()

    # =====================================================
    # E. Normal optimizer step
    # =====================================================
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    print("\nDone.")


if __name__ == "__main__":
    main()