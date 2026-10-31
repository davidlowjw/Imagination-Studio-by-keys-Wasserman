"""Video decode on two cards for the MiniMax H3 video VAE (comfy/ldm/minimax/vae.py).

The decoder decodes independent temporal chunks of the latent and blends their overlaps in order afterwards. Here
the chunks alternate between the VAE's card and a copy of the VAE on the other card (spark/gpus.py place_decode
routes it); every chunk comes back to the VAE's card and the blending and writing run there exactly as before, in
order. Same card model, same kernels, the same tile batch: the output is the one-card output byte for byte.
STUDIO_DECODE_SPLIT=off keeps one card (the workflow then has a plain VAEDecode).
"""

import logging
import os
import threading
import time

import torch

import comfy.model_management as model_management
import nodes

try:
    from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
except ImportError:  # an older ComfyUI: nothing to split
    MiniMaxH3VideoVAE = None

AHEAD = 2  # chunks a card may decode ahead of the blending (each ~0.1-0.3 GB at 720p)


def _tile_row(self, z_row, x_idx, x_len):
    """_decode_tile_row with the tile batch fixed for the whole decode (both cards decode the same batches)."""
    batch = getattr(self, "_studio_batch", None)
    if batch is None:
        yield from _stock_tile_row(self, z_row, x_idx, x_len)
        return
    slices = [z_row[..., j_pos // self.vae_ratio:(j_pos + j_len) // self.vae_ratio] for j_pos, j_len in zip(x_idx, x_len)]
    for k in range(0, len(slices), batch):
        group = slices[k:k + batch]
        yield from self._decode_pixels(torch.cat(group)).chunk(len(group))


def _decode_temporal(self, z, output_buffer=None):
    twin = getattr(self, "_studio_twin", None)
    if twin is None:
        return _stock_decode_temporal(self, z, output_buffer)
    # the stock decode_temporal, with the chunks' decodes on two cards
    chunk_dec = self.tokens_chunk_size * self.vae_ratio_t
    split_count = int(self.token_drop > 0) + 1
    if output_buffer is None:
        output_buffer = torch.empty(self.decode_output_shape(z.shape), dtype=torch.float32,
                                    device=model_management.intermediate_device())
    pad_tokens, num_chunks = self._decode_temporal_chunks(z.shape[2])
    if pad_tokens > 0:
        pad_z = z[:, :, -1:, :, :].repeat(1, 1, pad_tokens, 1, 1)
        z = torch.cat([z, pad_z], dim=2)
    home, other = z.device, self._studio_twin_device
    free = model_management.get_free_memory(home)
    self._studio_batch = twin._studio_batch = int(max(1, min(4, free // (128 * 2**20 * z.shape[0]))))
    z_other = z.to(other)
    grad, inference = torch.is_grad_enabled(), torch.is_inference_mode_enabled()
    results, errors = {}, []
    ready = threading.Condition()
    room = {home: threading.Semaphore(AHEAD), other: threading.Semaphore(AHEAD)}
    stop = threading.Event()
    owner = {i: (home if i % 2 == 0 else other) for i in range(num_chunks)}

    def worker(model, device, zz):
        try:
            with torch.cuda.device(device), torch.inference_mode(inference), torch.set_grad_enabled(grad):
                for i in range(num_chunks):
                    if owner[i] != device:
                        continue
                    while not room[device].acquire(timeout=0.5):
                        if stop.is_set():
                            return
                    if stop.is_set():
                        return
                    t_start_idx = i * self.tokens_chunk_size
                    t_end_idx = t_start_idx + self.tokens_chunk_size + self.token_overlap
                    out = model._adaptive_decode(zz[:, :, t_start_idx:t_end_idx, :, :])
                    if device != home:
                        torch.cuda.synchronize(device)
                        out = out.to(home)
                        torch.cuda.synchronize(home)
                    with ready:
                        results[i] = out
                        ready.notify_all()
        except BaseException as error:  # noqa: BLE001
            with ready:
                errors.append(error)
                ready.notify_all()

    started = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(self, home, z), daemon=True, name="decode-home"),
               threading.Thread(target=worker, args=(twin, other, z_other), daemon=True, name="decode-other")]
    for thread in threads:
        thread.start()
    dec = output_buffer
    dec_overlap = None
    write_pos = 0

    def write_part(part):
        nonlocal write_pos
        part_frames = part.shape[2]
        if part_frames <= 0:
            return
        part = self._finalize_pixels(part)
        copy_frames = min(part_frames, max(0, dec.shape[2] - write_pos))
        if copy_frames > 0:
            dec[:, :, write_pos:write_pos + copy_frames, :, :].copy_(part[:, :, :copy_frames, :, :])
            write_pos += copy_frames

    try:
        for i in range(num_chunks):
            with ready:
                while i not in results and not errors:
                    ready.wait()
                if errors:
                    raise errors[0]
                clip_dec = results.pop(i)
            room[owner[i]].release()
            for j in range(split_count):
                f_start_idx = j * chunk_dec
                f_end_idx = min(f_start_idx + chunk_dec, clip_dec.shape[2])
                clip_dec_chunk = clip_dec[:, :, f_start_idx:f_end_idx, :, :]
                clip_dec_chunk = clip_dec_chunk[:, :, self.frame_pre_padding:, :, :]
                if j == 0:
                    if dec_overlap is not None:
                        clip_dec_chunk = self.blend(dec_overlap, clip_dec_chunk, self.frame_overlap, dim=-3)
                        dec_overlap = None
                    write_part(clip_dec_chunk)
                else:
                    dec_overlap = clip_dec_chunk.contiguous()
            if i == num_chunks - 1 and dec_overlap is not None:
                write_part(dec_overlap)
                dec_overlap = None
            del clip_dec, clip_dec_chunk
    finally:
        stop.set()
        for thread in threads:
            thread.join()
        self._studio_batch = twin._studio_batch = None
    logging.info(f"Studio two-card decode: {num_chunks} chunks on {home} and {other} in {time.perf_counter() - started:.2f} s")
    return dec


if MiniMaxH3VideoVAE is not None:
    _stock_decode_temporal = MiniMaxH3VideoVAE.decode_temporal
    _stock_tile_row = MiniMaxH3VideoVAE._decode_tile_row
    MiniMaxH3VideoVAE.decode_temporal = _decode_temporal
    MiniMaxH3VideoVAE._decode_tile_row = _tile_row


class StudioTwoCardVAEDecode:
    """VAEDecode with a second copy of the VAE on another card (vae_other): a MiniMax H3 video decodes its temporal
    chunks on both. Anything else decodes as VAEDecode does, on vae's card."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",), "vae": ("VAE",), "vae_other": ("VAE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "latent"

    def decode(self, samples, vae, vae_other):
        model, twin = vae.first_stage_model, vae_other.first_stage_model
        latent = samples["samples"]
        if (MiniMaxH3VideoVAE is None or not isinstance(model, MiniMaxH3VideoVAE) or not isinstance(twin, MiniMaxH3VideoVAE)
                or latent.ndim != 5 or latent.shape[2] < 2 or vae.device == vae_other.device
                or os.environ.get("STUDIO_DECODE_SPLIT", "on").strip().lower() in ("0", "off", "no", "false")):
            return nodes.VAEDecode().decode(vae, samples)
        try:
            with model_management.cuda_device_context(vae_other.device):
                model_management.load_models_gpu([vae_other.patcher], memory_required=vae_other.memory_used_decode(latent.shape, vae_other.vae_dtype),
                                                 force_full_load=vae_other.disable_offload)
        except Exception as error:  # noqa: BLE001
            logging.warning(f"Studio two-card decode: the copy on {vae_other.device} did not load ({error}); one card")
            return nodes.VAEDecode().decode(vae, samples)
        model._studio_twin, model._studio_twin_device = twin, vae_other.device
        try:
            return nodes.VAEDecode().decode(vae, samples)
        finally:
            model._studio_twin = model._studio_twin_device = None


NODE_CLASS_MAPPINGS = {"StudioTwoCardVAEDecode": StudioTwoCardVAEDecode}
NODE_DISPLAY_NAME_MAPPINGS = {"StudioTwoCardVAEDecode": "VAE Decode (two cards)"}
