from __future__ import annotations

from typing import Dict, Optional


def _fmt(x: Optional[float], nd: int = 2) -> str:
    if x is None:
        return "—"
    return f"{x:.{nd}f}"


def generate_claims(summary: Dict) -> str:
    """Create paper-safe claims based on observed metrics.

    Required summary keys:
    - accuracy: dict method -> percent
    - mcnemar: dict comparison -> {p, ...}
    - faithfulness: dict method -> {deletion, insertion}
    """
    acc = summary.get("accuracy", {})
    faith = summary.get("faithfulness", {})
    tests = summary.get("mcnemar", {})
    primary = summary.get("primary_judge", "semantic")

    base = acc.get("baseline")
    vg = acc.get("vg_crop")
    ours = acc.get("mcot_vg")
    router = acc.get("router")
    lines = []

    if base is None or ours is None:
        return "Insufficient results to generate claims."

    delta_base = ours - base
    p_base = tests.get("baseline_vs_mcot_vg", {}).get("p")
    p_vg = tests.get("vg_crop_vs_mcot_vg", {}).get("p")

    lines.append("# Paper-safe claim language")
    lines.append("")
    lines.append("## Main result framing")

    if delta_base > 0 and p_base is not None and p_base < 0.05:
        lines.append(
            f"MCoT+VG-v2 significantly improves {primary} accuracy over the direct Baseline "
            f"({_fmt(base)}% → {_fmt(ours)}%, +{_fmt(delta_base)} points; McNemar p={p_base:.4f})."
        )
    elif delta_base > 0:
        p_txt = f"; McNemar p={p_base:.4f}" if p_base is not None else ""
        lines.append(
            f"MCoT+VG-v2 achieves the highest or near-highest {primary} accuracy among the core non-router configurations "
            f"({_fmt(base)}% → {_fmt(ours)}%, +{_fmt(delta_base)} points{p_txt}), but the gain should be described as modest unless significance is confirmed."
        )
    elif abs(delta_base) <= 0.25:
        lines.append(
            f"MCoT+VG-v2 preserves Baseline-level {primary} accuracy ({_fmt(base)}% vs. {_fmt(ours)}%) while adding step-wise grounding. "
            "The paper should emphasize interpretability/faithfulness rather than accuracy gains."
        )
    else:
        lines.append(
            f"MCoT+VG-v2 reduces {primary} accuracy relative to the Baseline ({_fmt(base)}% → {_fmt(ours)}%, {_fmt(delta_base)} points). "
            "The paper should frame the result as an interpretability–accuracy trade-off unless the pipeline is further improved."
        )

    if vg is not None:
        delta_vg = ours - vg
        if delta_vg > 0 and p_vg is not None and p_vg < 0.05:
            lines.append(f"Compared with VG-Crop, MCoT+VG-v2 gives a significant accuracy gain (+{_fmt(delta_vg)} points; McNemar p={p_vg:.4f}).")
        elif delta_vg > 0:
            lines.append(f"Compared with VG-Crop, the accuracy gain is small (+{_fmt(delta_vg)} points), so VG-Crop should remain a strong baseline rather than being dismissed.")
        elif abs(delta_vg) <= 0.25:
            lines.append("MCoT+VG-v2 is accuracy-matched with VG-Crop; its value should be argued through step-wise reasoning and faithfulness rather than raw accuracy.")
        else:
            lines.append(f"VG-Crop outperforms MCoT+VG-v2 by {_fmt(-delta_vg)} points; the extra reasoning stage is currently not accuracy-beneficial.")

    lines.append("")
    lines.append("## Faithfulness framing")
    f_vg = faith.get("vg_crop", {})
    f_ours = faith.get("mcot_vg", {})
    del_vg = f_vg.get("deletion")
    del_ours = f_ours.get("deletion")
    ins_vg = f_vg.get("insertion")
    ins_ours = f_ours.get("insertion")
    if del_vg is not None and del_ours is not None:
        if del_ours > del_vg:
            lines.append(f"Deletion faithfulness increases from {_fmt(del_vg, 5)} to {_fmt(del_ours, 5)}, suggesting that the highlighted regions are more causally relevant under MCoT+VG-v2.")
        elif abs(del_ours - del_vg) < 1e-4:
            lines.append("Deletion faithfulness is approximately unchanged; do not claim a faithfulness improvement from deletion alone.")
        else:
            lines.append(f"Deletion faithfulness decreases from {_fmt(del_vg, 5)} to {_fmt(del_ours, 5)}; do not claim improved deletion faithfulness.")
    if ins_vg is not None and ins_ours is not None:
        if ins_ours > ins_vg:
            lines.append(f"Insertion faithfulness also increases from {_fmt(ins_vg, 5)} to {_fmt(ins_ours, 5)}.")
        else:
            lines.append(f"Insertion faithfulness is comparable or slightly lower ({_fmt(ins_vg, 5)} vs. {_fmt(ins_ours, 5)}), so report deletion and insertion separately rather than merging them into one score.")

    if router is not None:
        lines.append("")
        lines.append("## Optional router framing")
        if router > max(base, ours, vg if vg is not None else -1):
            lines.append(f"The optional MCoT+VG-Router reaches {_fmt(router)}%, but it should be reported as an additional ensemble/selector variant, not as the core MCoT+VG result.")
        else:
            lines.append("The router does not improve over the best core method; omit it from the main paper or keep it in an appendix.")

    lines.append("")
    lines.append("## Claims to avoid")
    lines.append("- Do not claim HIS, SC, or Fleiss' κ until real five-annotator labels are collected.")
    lines.append("- Do not claim that VG improves reasoning accuracy unless MCoT+VG-v2 beats Baseline with a statistically defensible margin.")
    lines.append("- Do not describe CLIP faithfulness as independent human faithfulness; it is an automated proxy and should be acknowledged as a limitation.")
    return "\n".join(lines) + "\n"
