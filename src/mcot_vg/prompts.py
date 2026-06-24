from __future__ import annotations


def baseline_prompt(question: str) -> str:
    return (
        f"Question: {question}\n"
        "Answer in 1-3 words only.\n"
        "Output only the answer phrase, with no explanation.\n"
        "FINAL ANSWER:"
    )


def mcot_prompt_v2(question: str) -> str:
    # Do not mention dataset names here; some LLaVA checkpoints copy them as the answer.
    return (
        "Solve this visual question using visible evidence and necessary commonsense/world knowledge.\n\n"
        "Step 1: Identify the visual evidence relevant to the question.\n"
        "Step 2: Add only the necessary commonsense or world knowledge.\n"
        "Step 3: Infer the most likely short answer.\n\n"
        f"Question: {question}\n"
        "Keep the reasoning concise. End with the answer phrase only.\n"
        "Answer in 1-3 words only.\n"
        "FINAL ANSWER:"
    )


def step1_visual_evidence_prompt(question: str) -> str:
    return (
        f"Question: {question}\n\n"
        "Identify the single most relevant visible evidence needed to answer the question. "
        "Do not answer yet. One concise sentence only."
    )


def step2_grounded_evidence_prompt(question: str, step1: str) -> str:
    return (
        "The image shows the original full image on the left and a zoomed grounding crop on the right. "
        "Use the full image as the primary evidence and the crop only as an extra visual hint.\n\n"
        f"Question: {question}\n"
        f"Initial visual evidence: {step1[:220]}\n"
        "Describe the grounded evidence relevant to the answer in one concise sentence. "
        "Do not force the answer from the crop if the full image contradicts it."
    )


def final_answer_prompt(question: str, step1: str, step2: str) -> str:
    return (
        "The image shows the original full image on the left and a zoomed grounding crop on the right. "
        "Use the full image first; use the crop and evidence notes only as support.\n\n"
        f"Question: {question}\n"
        f"Visual evidence: {step1[:220]}\n"
        f"Grounded evidence: {step2[:240]}\n"
        "Use visible evidence plus commonsense/world knowledge if needed.\n"
        "Return only the object, action, attribute, number, place, or short phrase that answers the question.\n"
        "Do not output a dataset name, benchmark name, task name, or explanation.\n"
        "Answer in 1-3 words only.\n"
        "FINAL ANSWER:"
    )


def fallback_final_prompt(question: str) -> str:
    return (
        "Answer the visual question from the image.\n"
        f"Question: {question}\n"
        "Return only the short answer phrase. Do not output a dataset name or task name.\n"
        "Answer in 1-3 words only.\n"
        "FINAL ANSWER:"
    )


def vg_crop_prompt(question: str) -> str:
    return (
        "The image shows the original full image on the left and a zoomed grounding crop on the right. "
        "Use the full image as primary evidence and the crop as a helpful hint.\n\n"
        f"Question: {question}\n"
        "Answer in 1-3 words only.\n"
        "Output only the answer phrase, with no explanation.\n"
        "FINAL ANSWER:"
    )


def router_prompt(question: str, candidates: dict[str, str]) -> str:
    lines = [
        "Choose the answer that is most consistent with the image and the question.",
        "Do not prefer a longer answer. Prefer the most specific correct answer.",
        "Output only the final answer phrase.",
        f"Question: {question}",
    ]
    for name, ans in candidates.items():
        lines.append(f"{name}: {ans}")
    lines.extend(["Answer in 1-3 words only.", "FINAL ANSWER:"])
    return "\n".join(lines)
