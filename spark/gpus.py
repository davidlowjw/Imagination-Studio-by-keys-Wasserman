"""Two (or more) discrete NVIDIA cards working as one Studio: the stages of a ComfyUI workflow placed per card.

A DGX Spark has one GPU with the machine's memory; an RTX Pro 6000 has one 96 GB card. A workstation with two smaller
cards (2 x 48 GB RTX PRO 5000 Blackwell, for instance) holds no video model and its text encoder together on either
card, but each stage of a render fits on one: the transformer (FastH3, 22 GB int8 or 44 GB bf16) on the first card, the
text encoder (H3's 32B, 8-bit) and the video and audio decoders on the second. ComfyUI (v0.42+) carries the nodes that
do this (Select CLIP Device, Select VAE Device: comfy_extras/nodes_multigpu.py); ``place`` puts them into a workflow
just before it is queued, so every workflow the Studio builds (pictures, clips, references, music, sound, the Library's
actions) is placed the same way without each one knowing about it.

Environment:
  STUDIO_GPU_SPLIT    auto (default: on with two or more discrete cards), on, off
  STUDIO_ENCODER_GPU  the card (ComfyUI's gpu:N index among the visible ones) for text encoders and decoders, default 1
  STUDIO_DECODER_GPU  the card for the decoders (VAEs) when it should differ from the encoders', default the same
"""

from __future__ import annotations

import os
import subprocess
import time

# loaders whose outputs are (output index -> what it is); the model outputs stay on ComfyUI's default device (gpu:0)
CLIP_OUTPUTS = {"CLIPLoader": (0,), "DualCLIPLoader": (0,), "TripleCLIPLoader": (0,), "QuadrupleCLIPLoader": (0,),
                "CheckpointLoaderSimple": (1,), "ImageOnlyCheckpointLoader": ()}
VAE_OUTPUTS = {"VAELoader": (0,), "CheckpointLoaderSimple": (2,), "ImageOnlyCheckpointLoader": (2,)}
_CARDS: dict = {"at": 0.0, "value": []}


def cards(fresh: bool = False) -> list[dict]:
    """Every NVIDIA GPU with memory of its own: [{"index", "name", "total_mb", "used_mb", "free_mb"}]. Empty on a
    DGX Spark (its GB10 has no memory of its own: nvidia-smi says [N/A]) or without nvidia-smi. At most every 2 s."""

    if not fresh and time.monotonic() - _CARDS["at"] < 2:
        return _CARDS["value"]
    found = []
    try:
        said = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        lines = said.stdout.splitlines() if said.returncode == 0 else []
    except (OSError, subprocess.SubprocessError):
        lines = []
    for line in lines:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 5 and all(part.isdigit() for part in (parts[0], *parts[2:5])):
            found.append({"index": int(parts[0]), "name": parts[1], "total_mb": int(parts[2]),
                          "used_mb": int(parts[3]), "free_mb": int(parts[4])})
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and all(part.strip().isdigit() for part in visible.split(",")):
        order = [int(part) for part in visible.split(",")]
        by_index = {card["index"]: card for card in found}
        found = [by_index[index] for index in order if index in by_index]
    _CARDS.update(at=time.monotonic(), value=found)
    return found


def split() -> bool:
    """Whether a workflow's stages are spread over the cards (STUDIO_GPU_SPLIT; auto: two or more discrete cards)."""

    wanted = os.environ.get("STUDIO_GPU_SPLIT", "auto").strip().lower()
    if wanted in ("0", "off", "no", "false"):
        return False
    if os.environ.get("STUDIO_VRAM_GB", "").strip():
        return False  # the one-card simulation on a Spark
    return len(cards()) >= 2  # "on" and "auto" alike: there is nothing to split on one card


def encoder_gpu() -> int:
    try:
        return int(os.environ.get("STUDIO_ENCODER_GPU", "1"))
    except ValueError:
        return 1


def decoder_gpu() -> int:
    try:
        return int(os.environ.get("STUDIO_DECODER_GPU", str(encoder_gpu())))
    except ValueError:
        return encoder_gpu()


def place(workflow: dict) -> dict:
    """The workflow with each text encoder and decoder routed to its card (a Select CLIP / VAE Device node after
    its loader, every use of the loader's output pointed at it); unchanged when the cards are not split. A workflow
    that already places a loader itself is left as it is for that loader."""

    if not split():
        return workflow
    out = {key: dict(node, inputs=dict(node.get("inputs", {}))) for key, node in workflow.items()}
    taken = {int(key) for key in out if str(key).isdigit()}
    next_id = max(taken, default=0) + 1000
    placed = {(ref[0], ref[1]) for node in out.values() if node.get("class_type") in ("SelectCLIPDevice", "SelectVAEDevice")
              for ref in node["inputs"].values() if isinstance(ref, list) and len(ref) == 2}
    routes = {}  # (loader id, output) -> new node id
    for key, node in list(out.items()):
        kind = str(node.get("class_type", ""))
        for slot in CLIP_OUTPUTS.get(kind, ()):
            if (key, slot) not in placed:
                routes[(key, slot)] = ("SelectCLIPDevice", "clip", f"gpu:{encoder_gpu()}")
        for slot in VAE_OUTPUTS.get(kind, ()):
            if (key, slot) not in placed:
                routes[(key, slot)] = ("SelectVAEDevice", "vae", f"gpu:{decoder_gpu()}")
    if not routes:
        return out
    new_ids = {}
    for (key, slot), (kind, field, device) in routes.items():
        new_id = str(next_id)
        next_id += 1
        new_ids[(key, slot)] = new_id
        out[new_id] = {"class_type": kind, "inputs": {field: [key, slot], "device": device}}
    for node_id, node in out.items():
        if node_id in new_ids.values():
            continue
        for name, value in node["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and (str(value[0]), value[1]) in new_ids:
                node["inputs"][name] = [new_ids[(str(value[0]), value[1])], 0]
    return out


def describe() -> str:
    """One line for the logs: how the cards are used."""

    found = cards()
    if not split():
        return "one card" if len(found) <= 1 else f"{len(found)} cards, split off (STUDIO_GPU_SPLIT)"
    return (f"{len(found)} cards: diffusion model on gpu:0, text encoders on gpu:{encoder_gpu()}, "
            f"decoders on gpu:{decoder_gpu()}")


if __name__ == "__main__":
    import json
    print(json.dumps({"cards": cards(fresh=True), "split": split(), "plan": describe()}, indent=1))
