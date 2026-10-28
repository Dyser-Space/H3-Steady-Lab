# H3 Steady Lab

A ComfyUI model patch for **MiniMax H3**, designed to reduce video generation failures and artifacts while the video is being generated. It is not a general quality enhancer or a repair tool for finished videos. The goal is to avoid faulty results as much as possible; it cannot prevent every error.

> [!WARNING]
> **Experimental**
> H3 Steady Lab has been tested, but not across all possible models, settings, and scenarios. Results may vary. Start with the recommended values and compare with the node disabled using the same seed.

## Installation

Place this repository in `ComfyUI/custom_nodes/h3_steady_lab/` and restart ComfyUI. Keep `__init__.py` and `profil_lab.safetensors` together in that folder.

Requires a ComfyUI installation with native MiniMax H3 support and the appropriate H3 model files.

## Where to connect it

Add **H3 Steady Lab** from `advanced/model_patches` to the **MODEL** connection before sampling:

```text
Model loader → H3 Steady Lab → KSampler (model)
```

For an advanced sampling workflow, connect its MODEL output to both **BasicGuider** and **BasicScheduler**. Leave the conditioning, latent, video, and audio connections as they are.

## Recommended starting values

| Setting | Non-Turbo | Turbo |
| --- | --- | --- |
| Strength | 1.0–2.0 | 3.0 |
| Step 0 | 0.0 | 0.10–0.20 |
| Step 1 | 1.0 | 1.0 |
| Step 2 | 1.0 | 1.0 |
| Start / End block | 38 / 48 | 42 / 48 |
| Threshold | 95 % | 100 % (strict) |

These are starting points, not guaranteed best settings. Step 0–2 are strength multipliers for the first three sampling steps, counted from zero. Higher strength is not automatically better.

For regular clips, leave **Start second = 0**, **End second = 600**, and **offset = 0**. Set Strength to `0` to disable the effect and compare using the same seed.

The included profile was calibrated for **er_sde / beta / 20 steps**. For Turbo, keep your working Turbo model and sampling setup; the table changes only the Steady settings. The patch acts on image tokens only.

## Example workflow

Drag [example_workflow.json](example_workflow.json) into ComfyUI. It contains a simple MiniMax H3 text-to-video setup with native audio and H3 Steady Lab already connected before sampling.

- Non-Turbo example: Strength `1.5`, Step 0 `0`, `er_sde`, `beta`, 20 steps.
- Select your installed H3 diffusion model, text encoder, video VAE, and audio VAE in the loader nodes.
- Edit the prompt and run the workflow. The example uses 1344 × 768, 124 frames, and 24 fps.

Only H3 Steady Lab and native ComfyUI nodes are used. Workflow connections and settings were checked against the local node schemas; a full video generation was not run for this example.
