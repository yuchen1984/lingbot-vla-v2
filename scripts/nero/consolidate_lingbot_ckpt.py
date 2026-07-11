#!/usr/bin/env python3
"""Consolidate a LingBot LoRA DCP checkpoint into deploy-loadable safetensors.

Training saves a sharded DCP checkpoint whose model state contains peft LoRA layers
(`<mod>.base_layer.weight` + `<mod>.lora_A.default.weight` + `<mod>.lora_B.default.weight`)
plus the trainable action projections. The deploy loader (LingbotVLAv2Server) builds
the model WITHOUT LoRA and globs `*.safetensors`, so we:
  1. dcp_to_torch_state_dict(ckpt)  -> full training state_dict
  2. fold each LoRA:  W <- W + (B @ A) * (alpha/r);  rename base_layer.weight -> weight
  3. drop lora_A/lora_B; keep everything else as-is
  4. save one model.safetensors under <serve_root>/ckpt/<tag>/model/ and copy the
     training lingbotvla_cli.yaml to <serve_root>/ so the deploy loader
     (path.parent.parent.parent/lingbotvla_cli.yaml) finds it.

Usage:
  python scripts/nero/consolidate_lingbot_ckpt.py \
    --dcp output/nerocloth_lora/checkpoints/global_step_4000 \
    --cli-yaml output/nerocloth_lora/lingbotvla_cli.yaml \
    --serve-root output/nerocloth_serve --tag step_4000
Prints the model dir to pass as --model_path to the deploy server.
"""
import argparse
import os
import shutil

import torch
import yaml
from safetensors.torch import save_file

from lingbotvla.checkpoint.format_utils import dcp_to_torch_state_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dcp", required=True, help="DCP checkpoint dir (…/global_step_N)")
    ap.add_argument("--cli-yaml", required=True, help="training lingbotvla_cli.yaml")
    ap.add_argument("--serve-root", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lora-rank", type=int, default=None)
    ap.add_argument("--lora-alpha", type=int, default=None)
    args = ap.parse_args()

    with open(args.cli_yaml) as f:
        cli = yaml.safe_load(f)
    r = args.lora_rank or int(cli["train"].get("lora_rank", 16))
    alpha = args.lora_alpha or int(cli["train"].get("lora_alpha", 32))
    scaling = alpha / r
    print(f"LoRA merge scaling = alpha/r = {alpha}/{r} = {scaling}")

    print(f"loading DCP state_dict from {args.dcp} ...")
    sd = dcp_to_torch_state_dict(args.dcp)
    print(f"  {len(sd)} keys")

    out = {}
    merged = renamed = dropped = passthrough = 0
    for k in list(sd.keys()):
        if k.endswith(".base_layer.weight"):
            mod = k[: -len(".base_layer.weight")]
            W = sd[k].float()
            A = sd.get(f"{mod}.lora_A.default.weight")
            B = sd.get(f"{mod}.lora_B.default.weight")
            if A is not None and B is not None:
                W = W + (B.float() @ A.float()) * scaling
                merged += 1
            else:
                renamed += 1
            out[f"{mod}.weight"] = W.to(sd[k].dtype).contiguous()
        elif k.endswith(".base_layer.bias"):
            mod = k[: -len(".base_layer.bias")]
            out[f"{mod}.bias"] = sd[k].contiguous()
            renamed += 1
        elif ".lora_A." in k or ".lora_B." in k:
            dropped += 1
        else:
            out[k] = sd[k].contiguous()
            passthrough += 1
    print(f"  merged={merged} renamed={renamed} dropped={dropped} passthrough={passthrough} -> {len(out)} keys")

    model_dir = os.path.join(args.serve_root, "ckpt", args.tag, "model")
    os.makedirs(model_dir, exist_ok=True)
    # deploy loader reads <model_path>.parent.parent.parent/lingbotvla_cli.yaml
    shutil.copy(args.cli_yaml, os.path.join(args.serve_root, "lingbotvla_cli.yaml"))
    out_path = os.path.join(model_dir, "model.safetensors")
    print(f"saving {out_path} ...")
    save_file(out, out_path, metadata={"format": "pt"})
    print("DONE. model_path for deploy server:")
    print(model_dir)


if __name__ == "__main__":
    main()
