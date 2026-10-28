"""H3 Steady Lab (Noah Oct 28): the stabilizer from H3 Steady with exposed controls for manual experimentation.

Same calculation as comfyui-h3-steady: for each second of new image tokens, mean m along direction w, m < b → + strength·(b−m)·w.
With the default settings (steps 1+2, blocks 42–48, threshold 90 %), results are bit-identical to H3 Steady.
Profile profil_lab.safetensors (film/forschung_zeit/steady_lab_profil.py): w [step 0–2, block 0–49, 5376],
b [quantile, step, block] for 50/75/90/95/100 % of the 28 clean segments. Affects image tokens only.
"""
import logging
import os

import torch
from safetensors.torch import load_file

from comfy_api.latest import ComfyExtension, io

_PROFIL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profil_lab.safetensors")
_FPS = 24
_SCHWELLEN = ["50 % (mild)", "75 %", "90 % (Standard)", "95 %", "100 % (strict)"]


def _frame(k):
    b, r = divmod(k, 5)
    return b * 17 + (0 if r == 0 else 1 + 4 * (r - 1))


def _sekunden(layout, offset, sek_von, sek_bis):
    """[(second, start row, end row)] of new image tokens, only seconds in the window [sek_von, sek_bis)."""
    _, latent_t, lat_h, lat_w, _ = layout.signature
    rows = (lat_h // 2) * (lat_w // 2)
    va, vb, _ = next(s for s in layout.segments if s[2] == "video")
    spans = {}
    for k in range(offset, latent_t):
        spans.setdefault(_frame(k) // _FPS, []).append(k)
    out = [(t, va + ks[0] * rows, va + (ks[-1] + 1) * rows) for t, ks in sorted(spans.items()) if sek_von <= t < sek_bis]
    if out and out[-1][2] > vb:
        raise ValueError("H3 Steady Lab: unexpected video layout")
    return out


def _schritt(opts):
    sig, sched = opts.get("sigmas"), opts.get("sample_sigmas")
    if sig is None or sched is None:
        return -1
    sched = sched.flatten().float().to(sig.device)
    return int(torch.argmin((sched - sig.flatten()[0].float()).abs()))


def _add(tok, delta, gen):
    if tok.dtype != torch.bfloat16:
        tok.add_(delta.to(tok.dtype))
        return
    bits = (tok.float() + delta).view(torch.int32)
    bits.add_(torch.randint(0, 1 << 16, bits.shape, device=bits.device, dtype=torch.int32, generator=gen))
    bits.bitwise_and_(-65536)
    tok.copy_(bits.view(torch.float32).to(torch.bfloat16))


def _patch(w, b, blk, staerken, sek, log):
    cache = {}

    @torch.no_grad()
    def block(args, extra):
        out = extra["original_block"](args)
        schritt = _schritt(args["transformer_options"])
        if staerken.get(schritt, 0.0) == 0.0:
            return out
        h, layout = out["img"], args["layout"]
        key = (layout.signature, h.device)
        if key not in cache:
            cache.clear()
            cache[key] = (_sekunden(layout, *sek), w[:, blk].to(h.device))
        spans, wb = cache[key]
        gen = torch.Generator(device=h.device)
        gen.manual_seed(1234 + 97 * schritt + (blk - 42) % 50)            # same as H3 Steady (indexed from block 42 there)
        st, schwelle = staerken[schritt], float(b[schritt, blk])
        korr = []
        for x in (h if h.dim() == 3 else [h]):
            for t, a, z in spans:
                tok = x[a:z]
                m = float((tok.float() @ wb[schritt]).mean())
                if m < schwelle:
                    _add(tok, st * (schwelle - m) * wb[schritt], gen)
                    korr.append(t)
        if log is not None:
            log(schritt, blk, korr, len(spans) * (h.shape[0] if h.dim() == 3 else 1), [t for t, _, _ in spans])
        return out

    return block


class H3SteadyLab(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="H3SteadyLab_Apply",
            display_name="H3 Steady Lab",
            category="advanced/model_patches",
            description="The stabilizer from H3 Steady with exposed controls (steps, blocks, threshold, time window). "
                        "Default settings = H3 Steady. Calibrated for er_sde, beta, 20 steps. Affects image tokens only.",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("staerke", display_name="Strength", default=2.0, min=-4.0, max=6.0, step=0.05,
                               tooltip="1.0 = up to the threshold, 2.0 = twice the distance. Negative = counter-test: "
                                       "pushes at-risk seconds further toward the failure side."),
                io.Float.Input("schritt_0", display_name="Step 0", default=0.0, min=0.0, max=3.0, step=0.05,
                               tooltip="Strength multiplier at step 0 (almost pure noise). 0 = off."),
                io.Float.Input("schritt_1", display_name="Step 1", default=1.0, min=0.0, max=3.0, step=0.05,
                               tooltip="Strength multiplier at step 1 (where H3 establishes the plan)."),
                io.Float.Input("schritt_2", display_name="Step 2", default=1.0, min=0.0, max=3.0, step=0.05,
                               tooltip="Strength multiplier at step 2."),
                io.Int.Input("block_von", display_name="Start block", default=42, min=0, max=49,
                             tooltip="First block. The direction mainly separated cases in blocks 32–48; blocks 0–25 and 49 do not separate them."),
                io.Int.Input("block_bis", display_name="End block", default=48, min=0, max=49,
                             tooltip="Last block (inclusive)."),
                io.Combo.Input("schwelle", display_name="Threshold", options=_SCHWELLEN, default=_SCHWELLEN[2],
                               tooltip="When a second is considered at risk: quantile of the 28 clean segments. "
                                       "Higher = more seconds are affected and pushed further."),
                io.Int.Input("sek_von", display_name="Start second", default=0, min=0, max=600,
                             tooltip="Only seconds from here onward (whole seconds of the generated clip, including context)."),
                io.Int.Input("sek_bis", display_name="End second", default=600, min=1, max=600,
                             tooltip="Only seconds before this point (exclusive). 3 to 5 = seconds 3.0 to 5.0."),
                io.Int.Input("offset", default=0, min=0, max=4096,
                             tooltip="Initial latent frames to leave untouched (chain from stage 2 onward: 27). 0 for regular clips."),
                io.Boolean.Input("protokoll", display_name="Logging", default=True,
                                 tooltip="Logs how many cells (second × block) were corrected at each step to the console."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, staerke, schritt_0, schritt_1, schritt_2, block_von, block_bis, schwelle, sek_von, sek_bis,
                offset, protokoll) -> io.NodeOutput:
        staerken = {s: staerke * f for s, f in enumerate((schritt_0, schritt_1, schritt_2)) if staerke * f != 0.0}
        bloecke = list(range(min(block_von, block_bis), max(block_von, block_bis) + 1))
        if not staerken or sek_bis <= sek_von:
            logging.info("H3 Steady Lab: nothing to do (strength, steps, or time window empty)")
            return io.NodeOutput(model)
        prof = load_file(_PROFIL)
        w, b = prof["w"], prof["b"][_SCHWELLEN.index(schwelle)]
        zaehler = {}

        def log(schritt, blk, korr, gesamt, sekunden):
            z = zaehler.setdefault(schritt, [0, 0, {}])
            z[0] += len(korr)
            z[1] += gesamt
            for t in korr:
                z[2][t] = z[2].get(t, 0) + 1
            if blk == bloecke[-1]:
                je = " ".join(f"{t}s:{z[2].get(t, 0)}" for t in sekunden)
                logging.info(f"H3 Steady Lab: step {schritt}: {z[0]} of {z[1]} cells corrected "
                             f"(blocks {bloecke[0]}–{bloecke[-1]}); per second, out of {len(bloecke)} blocks: {je}")
                zaehler.pop(schritt)

        m = model.clone()
        for blk in bloecke:
            m.set_model_patch_replace(_patch(w, b, blk, staerken, (offset, sek_von, sek_bis), log if protokoll else None),
                                      "dit", "double_block", blk)
        logging.info(f"H3 Steady Lab: strength {staerke}, steps {staerken}, blocks {bloecke[0]}–{bloecke[-1]}, "
                     f"threshold {schwelle}, seconds {sek_von}–{sek_bis}, offset {offset}")
        return io.NodeOutput(m)


class H3SteadyLabExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [H3SteadyLab]


async def comfy_entrypoint() -> H3SteadyLabExtension:
    return H3SteadyLabExtension()
