"""LLM judge for answer correctness and faithfulness (Gemini structured output)."""

from pydantic import BaseModel

from wb_rag.config import get_settings
from wb_rag.llm import complete
from wb_rag.models import Answer


class Verdict(BaseModel):
    correctness: str            # correct | partially_correct | incorrect
    score: int                  # 1-5
    faithful: bool
    unsupported_claims: list[str]
    reasoning: str
    model: str | None = None


SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "correctness": {"type": "string", "enum": ["correct", "partially_correct", "incorrect"]},
        "score": {"type": "integer"},
        "faithful": {"type": "boolean"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reasoning", "correctness", "score", "faithful", "unsupported_claims"],
}

JUDGE_SYSTEM = """You grade answers from a question-answering system over the World Bank's Global Economic Prospects (June 2025) report.

You receive a question, a reference answer written from the report, the system's answer, and the excerpts the system was given.

correctness: compare the answer with the reference.
- "correct": it states the key facts and figures in the reference. Extra correct detail is fine; different wording or rounding of the same figure is fine.
- "partially_correct": some key facts are right and others are missing or wrong. For multi-part or summary questions, about half of the reference points are covered.
- "incorrect": the main fact or figure is wrong or missing, or the system declined to answer even though the reference has an answer.
score: 1-5, consistent with correctness (5 = fully matches the reference).
faithful: true only if every factual claim in the answer is supported by the excerpts. Judge this against the excerpts, not against the reference.
Write the reasoning first, in two or three sentences."""


def judge(answer: Answer, expected: str) -> Verdict:
    s = get_settings()
    context = "\n\n".join(f"[{b.n}] {b.section} (p.{b.page_start})\n{b.text}" for b in answer.context)
    prompt = (f"<question>\n{answer.question}\n</question>\n\n<reference_answer>\n{expected}\n</reference_answer>\n\n"
              f"<system_answer>\n{answer.answer}\n</system_answer>\n\n<excerpts>\n{context}\n</excerpts>")
    res = complete(JUDGE_SYSTEM, prompt, model=s.judge_model, json_schema=SCHEMA, temperature=0.0,
                   max_tokens=2048, fallbacks=["gemini-3.5-flash-lite"])
    return Verdict(**res.json(), model=res.model)  # type: ignore[arg-type]
