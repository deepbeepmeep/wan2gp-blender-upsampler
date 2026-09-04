from __future__ import annotations

from typing import Any

import torch

from postprocessing.temporal_upsamplers import SimpleScaleSuffixMixin


def blend_midpoints(sample: torch.Tensor) -> torch.Tensor:
    """Insert one blended midpoint frame between every neighboring frame pair.

    WanGP passes decoded video post-processing media as [C, T, H, W]. The
    returned tensor keeps channels, height, width, dtype, and device unchanged
    while expanding the temporal dimension from T to 2T - 1.
    """
    frame_count = sample.shape[1]
    if frame_count < 2:
        return sample
    before = sample[:, :-1]
    after = sample[:, 1:]
    if sample.dtype == torch.uint8:
        midpoints = before.to(torch.float32).add_(after.to(torch.float32)).mul_(0.5).round_().clamp_(0, 255).to(torch.uint8)
    else:
        midpoints = before.add(after).mul_(0.5)
    output = sample.new_empty((sample.shape[0], frame_count * 2 - 1, *sample.shape[2:]))
    output[:, 0::2] = sample
    output[:, 1::2] = midpoints
    return output


class TemporalBlendUpsampler(SimpleScaleSuffixMixin):
    """Minimal temporal upsampler plugin handler.

    This class is intentionally small so plugin authors can copy it as a
    starting point. The API is the same one used by built-in temporal
    upsamplers: declare capabilities with query_temporal_upsampler_def(),
    validate a selected value, and return (sample, previous_last_frame, fps)
    from temporal_upsample().
    """

    METHOD = "blend"
    MULTIPLIERS = (2.0,)

    def __init__(self, server_config: dict[str, Any] | None = None, files_locator=None):
        self.server_config = server_config
        self.files_locator = files_locator

    @classmethod
    def query_temporal_upsampler_def(cls) -> dict[str, Any]:
        return {
            "name": "Temporal Blend",
            "config_key": "blender_upsampler",
            "pos": 900,
            "method_pos": {cls.METHOD: 900},
            "methods": [("Temporal Blend", cls.METHOD)],
            "multipliers": {cls.METHOD: cls.MULTIPLIERS},
            "default_temporal_upsampling": "blend*2",
            "description": "Fast model-free x2 interpolation that averages neighboring frames. It can reduce repeated-frame stutter, but moving subjects may show ghosting; it is primarily a reference and test method.",
        }

    def validate_upsampling(self, temporal_upsampling, *, source_is_image: bool = False) -> str:
        split = self.split_value(temporal_upsampling)
        if split is None or split[1] not in self.MULTIPLIERS:
            return "Temporal Blend only supports blend*2"
        return "Temporal Upsampling can not be used with an Image" if source_is_image else ""

    def download(self, process_files, send_cmd=None, status_text: str | None = None, temporal_upsampling=None) -> bool:
        """No-op download hook.

        Keep this method when your plugin has optional assets. This template has
        no model files, so returning False tells WanGP that nothing was
        downloaded.
        """
        return False

    def release_vram(self) -> None:
        """No-op release hook.

        If an upsampler creates an MMGP offload object, register it with
        shared.utils.offload_registry and release it here. This template only
        uses tensor operations, so there is nothing persistent to release.
        """
        return None

    def temporal_upsample(self, temporal_upsampling, sample, previous_last_frame, fps, *, abort_callback=None, progress_callback=None, **kwargs):
        split = self.split_value(temporal_upsampling)
        if split is None or split[1] not in self.MULTIPLIERS:
            raise ValueError(self.validate_upsampling(temporal_upsampling, source_is_image=False))
        if callable(abort_callback) and abort_callback():
            return sample, previous_last_frame, fps
        if callable(progress_callback):
            progress_callback("Temporal Blend", 0, 1)
        next_previous_last_frame = sample[:, -1:].clone()
        if previous_last_frame is not None:
            sample = torch.cat([previous_last_frame, sample], dim=1)
        output = blend_midpoints(sample)
        if previous_last_frame is not None:
            output = output[:, 1:]
        if callable(abort_callback) and abort_callback():
            return output, next_previous_last_frame, fps * 2
        if callable(progress_callback):
            progress_callback("Temporal Blend", 1, 1)
        return output, next_previous_last_frame, fps * 2
