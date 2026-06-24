from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import List, Sequence

from .metrics import strict_match
from .utils import read_jsonl, append_jsonl


class GeminiJudge:
    """Gemini semantic equivalence judge with JSONL cache.

    Use only for answer equivalence, not for explanations or claims.
    The prompt is intentionally strict to avoid inflated accuracy.
    """

    def __init__(self, cache_path: str | Path, model_name: str = "gemini-2.5-flash", project: str | None = None, location: str | None = None):
        from google import genai
        from google.oauth2 import service_account

        self.cache_path = Path(cache_path)
        self.model_name = model_name
        self.cache = {}
        for row in read_jsonl(self.cache_path):
            if "key" in row and "result" in row:
                self.cache[row["key"]] = bool(row["result"])

        project = project or os.environ.get("GCP_PROJECT")
        location = location or os.environ.get("GCP_LOCATION", "us-central1")
        key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not project:
            raise RuntimeError("Set GCP_PROJECT for Gemini judge.")
        if key_path and Path(key_path).exists():
            creds = service_account.Credentials.from_service_account_file(
                key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            self.client = genai.Client(vertexai=True, project=project, location=location, credentials=creds)
        else:
            self.client = genai.Client(vertexai=True, project=project, location=location)

    @staticmethod
    def _key(question: str, prediction: str, answers: Sequence[str]) -> str:
        s = question + "\n" + prediction + "\n" + " || ".join(map(str, answers))
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def judge(self, question: str, prediction: str, answers: Sequence[str], fallback_strict: bool = True) -> bool:
        key = self._key(question, prediction, answers)
        if key in self.cache:
            return self.cache[key]
        prompt = (
            "You are judging A-OKVQA direct-answer correctness.\n"
            "Question: " + question + "\n"
            "Ground-truth acceptable answers: " + "; ".join(map(str, answers)) + "\n"
            "Model prediction: " + prediction + "\n\n"
            "Return YES only if the prediction is semantically equivalent to at least one ground-truth answer. "
            "Accept synonyms, singular/plural, and paraphrases. Reject vague, overly broad, contradictory, or merely related answers.\n"
            "Reply with exactly YES or NO."
        )
        result = None
        for attempt in range(4):
            try:
                resp = self.client.models.generate_content(model=self.model_name, contents=prompt)
                txt = (resp.text or "").strip().upper()
                if txt.startswith("YES"):
                    result = True
                    break
                if txt.startswith("NO"):
                    result = False
                    break
            except Exception:
                time.sleep(2 ** attempt)
        if result is None:
            result = strict_match(answers, prediction) if fallback_strict else False
        self.cache[key] = bool(result)
        append_jsonl({"key": key, "question": question, "prediction": prediction, "answers": list(answers), "result": bool(result)}, self.cache_path)
        return bool(result)
