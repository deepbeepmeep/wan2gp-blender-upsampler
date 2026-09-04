# Temporal Blend

This plugin is a tutorial implementation for WanGP temporal upsampler plugins.
It adds a simple frame interpolation method named **Temporal Blend**. The method supports
only **x2** temporal upsampling and creates each intermediate frame by blending
the frame before it with the frame after it.

The plugin is metadata-only: there is no `plugin.py` and no UI tab. WanGP
discovers the upsampler from `plugin_info.json`:

```json
{
  "type": "processor",
  "temporal_upsampler_handlers": [".upsampler.TemporalBlendUpsampler"]
}
```

`processor` is the Plugin Manager category for processing extensions such as
spatial and temporal upsamplers. Relative handler paths are resolved from the
plugin package root. The entry above loads
`plugins/wan2gp-blender-upsampler/upsampler.py` and instantiates
`TemporalBlendUpsampler(server_config, files_locator)`.

## Files

- `plugin_info.json`: plugin metadata and the `temporal_upsampler_handlers` list.
- `upsampler.py`: the actual handler class and frame blending function.
- `README.md`: tutorial notes for plugin authors.

## Handler Checklist

`TemporalBlendUpsampler` demonstrates the core temporal upsampler surface:

- Inherit `SimpleScaleSuffixMixin` when your serialized values follow the common
  `<method>*<scale>` convention, for example `blend*2`. The parser still accepts
  the former concatenated syntax, while newly built values use `*`.
- Implement `query_temporal_upsampler_def()` to declare display labels, method
  ids, supported multipliers, dropdown ordering, the config key, and a concise
  `description` shown by WanGP's Temporal Upsampling information button.
- Implement `validate_upsampling(...)` to return an empty string when the
  selection is valid, or a user-facing error message when it is not.
- Implement `temporal_upsample(...)` and return `(sample, previous_last_frame,
  fps)`.
- Implement `download(...)` if the plugin needs checkpoints or other assets.
  Return `False` when nothing was downloaded.
- Implement `release_vram()` if the plugin owns models, CUDA memory, or an MMGP
  offload object.

## Tensor Contract

Temporal post-processing receives decoded video tensors shaped:

```text
[channels, frames, height, width]
```

For x2 blending, this template inserts one midpoint between every neighboring
frame pair:

```python
midpoint = (before + after) * 0.5
```

So an input with `T` frames returns `2T - 1` frames. WanGP passes an optional
`previous_last_frame` when it processes sliding-window chunks. This plugin uses
that frame to create a smooth boundary midpoint, then drops the duplicated
previous original frame from the returned chunk.

## Values and Multipliers

The method id is `blend`, so the UI stores one supported value:

- `blend*2`

The method id must be multiplier-free in `methods`, `method_pos`, and
`multipliers`. Only serialized selections contain the multiplier suffix.

## Progress and Abort

`temporal_upsample(...)` receives optional callbacks:

- `abort_callback()`: return early when it becomes true.
- `progress_callback(status, current_step, total_steps)`: report integer
  progress.

This template reports a one-step operation. Longer temporal upsamplers should
report meaningful phases such as model loading, interpolation, and frame
stitching.

## Config Guidance

This template intentionally has no configuration section. Only add config when
there is real shared extension behavior to store.

If your temporal upsampler needs shared config, put it under:

```json
{
  "temporal_upsamplers": {
    "your_config_key": {
      "your_option": "value"
    }
  }
}
```

Do not add temporal upsampler options to `models/_settings.json`; settings are
per-model generation defaults, while temporal upsampler config is shared
extension configuration.

## Downloads and Offload Objects

If a plugin needs files, declare them in `download(...)` and use the provided
`process_files(...)` function. If a plugin creates an MMGP offload object,
register it with `shared.utils.offload_registry` so WanGP's unload tools can
release it.

This template has no downloads and no persistent VRAM state.

## Smoke Test

From the WanGP repo root:

```powershell
C:\Users\Marc\anaconda3\envs\py311\python.exe -m py_compile plugins\wan2gp-blender-upsampler\upsampler.py
```

You can also instantiate the handler directly and verify a tiny tensor:

```python
import importlib
import sys
import torch

sys.path.insert(0, "plugins")
module = importlib.import_module("wan2gp-blender-upsampler.upsampler")
handler = module.TemporalBlendUpsampler({}, None)
sample = torch.tensor([[[[0]], [[10]], [[20]]]], dtype=torch.uint8)
output, previous_last_frame, fps = handler.temporal_upsample("blend*2", sample, None, 12)
assert output[:, :, 0, 0].tolist() == [[0, 5, 10, 15, 20]]
assert previous_last_frame[:, :, 0, 0].tolist() == [[20]]
assert fps == 24
```

After enabling the plugin in WanGP and restarting, **Temporal Blend** appears as a
post-processing temporal upsampler with the single multiplier **x2**.
