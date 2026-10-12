from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .config import CLASS_NAMES, Config
from .data import (
    ce_class_weights,
    channel_mean_std,
    class_pixels,
    extract_rare_instances,
    load_all,
    sample_batch,
)
from .evaluate import bootstrap, evaluate_split, markdown_table, per_image_conf
from .loss import segmentation_loss
from .model import build_model
from .visualize import save_examples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a ResNet34 U-Net on UMOD")
    p.add_argument("--data-dir", type=Path, required=True, help="Unzipped UMOD root with train/validation/test")
    p.add_argument("--run-dir", type=Path, default=Path("runs/resnet34_unet"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--encoder", type=str, default="resnet34")
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--eval-only", action="store_true", help="Load best.pt and score val/test")
    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def save_ckpt(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def main() -> None:
    args = parse_args()
    cfg = Config(
        data_dir=args.data_dir,
        run_dir=args.run_dir,
        seed=args.seed,
        encoder=args.encoder,
        total_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_tta=not args.no_tta,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.amp and device.type == "cuda"
    if device.type != "cuda":
        print("WARNING: no GPU. Training on CPU will be very slow.")
    else:
        print("GPU:", torch.cuda.get_device_name(0))

    data = load_all(cfg.data_dir)
    tr_imgs, tr_masks = data["train"]
    mean, std = channel_mean_std(tr_imgs)
    print("channel mean:", mean, "| std:", std)
    class_pix = class_pixels(tr_masks, seed=cfg.seed)
    paste_bank = extract_rare_instances(tr_imgs, tr_masks)
    print("rare-instance paste bank:", len(paste_bank), "patches")
    vals, cnts = np.unique(tr_masks, return_counts=True)
    ce_w = ce_class_weights(dict(zip(vals.tolist(), cnts.tolist()))).to(device)

    set_seed(cfg.seed)
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print("parameters: %.2f M | encoder %s" % (n_params, cfg.encoder))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.total_epochs * cfg.steps_per_epoch, pct_start=0.1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    state = {"epoch": 0, "best": -1.0, "history": [], "mean": mean.tolist(), "std": std.tolist()}

    last, best, logf = cfg.run_dir / "last.pt", cfg.run_dir / "best.pt", cfg.run_dir / "log.json"
    cfg.run_dir.mkdir(parents=True, exist_ok=True)

    def payload() -> dict:
        return {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "state": state,
            "cfg": vars(cfg) | {"data_dir": str(cfg.data_dir), "run_dir": str(cfg.run_dir)},
        }

    resume_path = last if last.exists() else None
    if args.eval_only:
        resume_path = best if best.exists() else last if last.exists() else None
        if resume_path is None:
            raise SystemExit(f"No checkpoint in {cfg.run_dir}. Train first.")
    if resume_path is not None and resume_path.exists():
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if not args.eval_only:
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
        mean = np.array(state["mean"], np.float32)
        std = np.array(state["std"], np.float32)
        print("resumed from", resume_path, "at epoch", state["epoch"])

    if not args.eval_only:
        while state["epoch"] < cfg.total_epochs:
            ep = state["epoch"]
            rng_ep = np.random.default_rng(1000 + ep + 100000 * cfg.seed)
            model.train()
            t0, run_loss = time.time(), 0.0
            for _ in range(cfg.steps_per_epoch):
                xb, yb = sample_batch(tr_imgs, tr_masks, class_pix, cfg, rng_ep, mean, std, paste_bank)
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(xb)
                loss, _ce = segmentation_loss(logits, yb, ce_w, cfg.n_classes)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                run_loss += float(loss.item())
            state["epoch"] = ep + 1
            rec = {
                "epoch": ep + 1,
                "train_loss": round(run_loss / cfg.steps_per_epoch, 4),
                "lr": round(opt.param_groups[0]["lr"], 6),
                "sec": round(time.time() - t0),
            }
            if (ep + 1) % cfg.eval_every == 0 or ep + 1 == cfg.total_epochs:
                tta_flag = cfg.use_tta
                cfg.use_tta = False
                va_imgs, va_masks = data["validation"]
                r = evaluate_split(model, va_imgs, va_masks, mean, std, device, cfg)
                cfg.use_tta = tta_flag
                rec["val_macro_dice"] = round(r["macro_dice"], 4)
                rec["val_fg_dice"] = round(r["fg_dice"], 4)
                if r["macro_dice"] > state["best"]:
                    state["best"] = r["macro_dice"]
                    save_ckpt(best, payload())
                    rec["best"] = True
            state["history"].append(rec)
            save_ckpt(last, payload())
            logf.write_text(json.dumps(state["history"], indent=1))
            print(rec, flush=True)

    ck_path = best if best.exists() else last
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    mean = np.array(ck["state"]["mean"], np.float32)
    std = np.array(ck["state"]["std"], np.float32)
    print("loaded", ck_path, "| best val macro Dice", round(ck["state"]["best"], 4))

    va_imgs, va_masks = data["validation"]
    te_imgs, te_masks = data["test"]
    val_confs = per_image_conf(model, va_imgs, va_masks, mean, std, device, cfg)
    test_confs = per_image_conf(model, te_imgs, te_masks, mean, std, device, cfg)
    val_res, test_res = bootstrap(val_confs), bootstrap(test_confs)
    print("\n## Validation\n" + markdown_table(val_res))
    print("\n## Test\n" + markdown_table(test_res))

    np.save(cfg.run_dir / "val_confs.npy", val_confs)
    np.save(cfg.run_dir / "test_confs.npy", test_confs)

    try:
        import pandas as pd

        cm = test_confs.sum(0)
        cm_norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        print("\nrow-normalised confusion matrix, test:")
        print(
            pd.DataFrame(
                cm_norm,
                index=CLASS_NAMES,
                columns=["bg", "rod", "rbc", "yst", "msc", "epc", "sm", "lg"],
            )
            .round(2)
            .to_string()
        )
    except ImportError:
        pass

    results = cfg.run_dir / f"results_epoch{ck['state']['epoch']}.md"
    results.write_text(
        "\n".join(
            [
                f"# UMOD 8-class {cfg.encoder} U-Net, epoch-{ck['state']['epoch']} checkpoint",
                "",
                "Whole-image evaluation with /32 padding. Hard argmax. Optional H/V flip TTA.",
                "Checkpoint chosen on validation macro Dice (no TTA during the training loop).",
                "",
                "## Test",
                markdown_table(test_res),
                "",
                "## Validation",
                markdown_table(val_res),
                "",
            ]
        )
        + "\n"
    )
    print("saved", results)
    save_examples(
        model,
        cfg.data_dir,
        te_imgs,
        te_masks,
        mean,
        std,
        device,
        cfg.run_dir / "examples_test.png",
        use_amp,
        cfg.use_tta,
    )


if __name__ == "__main__":
    main()
