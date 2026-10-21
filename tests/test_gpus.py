"""spark/gpus.py: a workflow's text encoders and decoders placed on the second card when there are two."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "spark"))
import gpus  # noqa: E402

TWO = [{"index": 0, "name": "card", "total_mb": 48000, "used_mb": 0, "free_mb": 48000},
       {"index": 1, "name": "card", "total_mb": 48000, "used_mb": 0, "free_mb": 48000}]


def video_workflow():
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "fasth3.safetensors", "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "te.safetensors", "type": "minimax", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "video.safetensors"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": "audio.safetensors"}},
        "6": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["3", 0], "vae": ["4", 0], "prompt": "x"}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["4", 0]}},
        "14": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["12", 0], "vae": ["5", 0]}},
        "12": {"class_type": "SamplerCustomAdvanced", "inputs": {"latent_image": ["6", 1]}},
    }


def with_cards(monkeypatch, found, split="auto"):
    monkeypatch.setattr(gpus, "cards", lambda fresh=False: found)
    monkeypatch.setenv("STUDIO_GPU_SPLIT", split)
    monkeypatch.delenv("STUDIO_VRAM_GB", raising=False)
    monkeypatch.delenv("STUDIO_ENCODER_GPU", raising=False)
    monkeypatch.delenv("STUDIO_DECODER_GPU", raising=False)


def test_one_card_leaves_the_workflow_alone(monkeypatch):
    with_cards(monkeypatch, TWO[:1])
    w = video_workflow()
    assert gpus.place(w) == w


def test_split_off_leaves_the_workflow_alone(monkeypatch):
    with_cards(monkeypatch, TWO, split="off")
    w = video_workflow()
    assert gpus.place(w) == w


def test_two_cards_route_encoder_and_decoders_to_the_second(monkeypatch):
    with_cards(monkeypatch, TWO)
    w = video_workflow()
    out = gpus.place(w)
    selects = {key: node for key, node in out.items() if node["class_type"].startswith("Select")}
    assert len(selects) == 3
    clip = [k for k, n in selects.items() if n["class_type"] == "SelectCLIPDevice"]
    vaes = {n["inputs"]["vae"][0]: k for k, n in selects.items() if n["class_type"] == "SelectVAEDevice"}
    assert len(clip) == 1 and out[clip[0]]["inputs"] == {"clip": ["3", 0], "device": "gpu:1"}
    assert set(vaes) == {"4", "5"} and all(selects[k]["inputs"]["device"] == "gpu:1" for k in vaes.values())
    # every consumer reads the placed output, not the loader's
    assert out["6"]["inputs"]["clip"] == [clip[0], 0]
    assert out["6"]["inputs"]["vae"] == [vaes["4"], 0]
    assert out["13"]["inputs"]["vae"] == [vaes["4"], 0]
    assert out["14"]["inputs"]["vae"] == [vaes["5"], 0]
    # the model and the other links are untouched, and the input is not changed in place
    assert out["12"]["inputs"]["latent_image"] == ["6", 1]
    assert not any(n["class_type"] == "SelectModelDevice" for n in out.values())
    assert w == video_workflow()


def test_devices_from_the_environment(monkeypatch):
    with_cards(monkeypatch, TWO)
    monkeypatch.setenv("STUDIO_ENCODER_GPU", "1")
    monkeypatch.setenv("STUDIO_DECODER_GPU", "0")
    out = gpus.place(video_workflow())
    devices = {n["class_type"]: n["inputs"]["device"] for n in out.values() if n["class_type"].startswith("Select")}
    assert devices == {"SelectCLIPDevice": "gpu:1", "SelectVAEDevice": "gpu:0"}


def test_checkpoint_loader_clip_and_vae_outputs(monkeypatch):
    with_cards(monkeypatch, TWO)
    w = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "x"}},
         "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "a"}},
         "3": {"class_type": "KSampler", "inputs": {"model": ["1", 0]}},
         "4": {"class_type": "VAEDecode", "inputs": {"vae": ["1", 2], "samples": ["3", 0]}}}
    out = gpus.place(w)
    assert out["3"]["inputs"]["model"] == ["1", 0]
    assert out[out["2"]["inputs"]["clip"][0]]["class_type"] == "SelectCLIPDevice"
    assert out[out["4"]["inputs"]["vae"][0]]["class_type"] == "SelectVAEDevice"


def test_a_workflow_that_places_itself_is_not_placed_twice(monkeypatch):
    with_cards(monkeypatch, TWO)
    w = {"1": {"class_type": "CLIPLoader", "inputs": {"clip_name": "te"}},
         "2": {"class_type": "SelectCLIPDevice", "inputs": {"clip": ["1", 0], "device": "gpu:0"}},
         "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "a"}}}
    assert gpus.place(w) == w


def test_the_simulation_of_one_card_never_splits(monkeypatch):
    with_cards(monkeypatch, TWO)
    monkeypatch.setenv("STUDIO_VRAM_GB", "96")
    assert not gpus.split()
