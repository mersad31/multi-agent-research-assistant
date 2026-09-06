from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from typing import Any

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

logger = logging.getLogger(__name__)

def _lookup_ground_truth(
        question: str,
        golden_dataset: Iterable[Mapping[str, Any]] | None,
) -> str | None:

    if not golden_dataset or not question:
        return None

    normalized_question = question.strip()

    for row in golden_dataset:
        row_question = str(row.get("query") or "").strip()
        if row_question == normalized_question:
            value = row.get("ground_truth")
            if value is not None:
                text = str(value).strip()
                if text:
                    return text

    return None


def _extract_context_from_sources(sources: Iterable[Mapping[str, Any]]) -> list[str]:
    contexts: list[str] = []

    for source in sources or []:
        if not isinstance(source, dict):
            continue

        content = str(source.get("content") or "").strip()
        if content:
            contexts.append(content)

    return contexts


def _send_scores_to_langfuse(
        langfuse_client: Any,
        trace_id: str,
        scores: Mapping[str, Any],
) -> None:
    if langfuse_client is None or not trace_id:
        return

    for name, value in scores.items():
        if value is None:
            continue

        try:
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                continue

            if hasattr(langfuse_client, "score"):
                langfuse_client.score(
                    trace_id=trace_id,
                    name=name,
                    value=numeric_value,
                )
            elif hasattr(langfuse_client, "create_score"):
                langfuse_client.create_score(
                    trace_id=trace_id,
                    name=name,
                    value=numeric_value,
                )
            else:
                logger.warning(
                    "Langfuse client does not support score/create_score; "
                    "skipping score '%s'", name
                )

        except Exception:
            logger.exception(
                "Failed to send RAGAS score '%s' to Langfuse", name
            )


def evaluate_state(
        state: dict[str, Any],
        trace_id: str,
        *,
        ground_truth: str | None = None,
        golden_dataset: Iterable[Mapping[str, Any]] | None = None,
        langfuse_client: Any,
        judge_llm: Any = None,
) -> dict[str, Any] | None:

    question = str(state.get("query") or "").strip()
    answer = str(state.get("report") or "").strip()

    if not answer:
        logger.warning(
            "Skipping RAGAS evaluation: empty report. trace_id=%s",
            trace_id,
        )
        return None

    sources = state.get("sources") or []
    contexts = _extract_context_from_sources(sources)

    if not contexts:
        logger.warning(
            "No contexts extracted. Skipping. trace_id=%s",
            trace_id
        )
        return None

    if ground_truth is None:
        ground_truth = _lookup_ground_truth(
            question=question,
            golden_dataset=golden_dataset,
        )

    data: dict[str, list[Any]] = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    }

    metrics = [faithfulness, answer_relevancy]

    if ground_truth and ground_truth.strip():
        gt = ground_truth.strip()
        data["ground_truth"] = [gt]
        metrics.extend([context_precision, context_recall])

    dataset = Dataset.from_dict(data)

    try:
        evaluation_result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=judge_llm,
        )
    except Exception:
        logger.exception(
            "RAGAS evaluation failed. trace_id=%s",
            trace_id,
        )
        return None

    try:
        result_df = evaluation_result.to_pandas()
        if result_df.empty:
            logger.warning(
                "RAGAS returned empty results. trace_id=%s",
                trace_id,
            )
            return None

        scores = result_df.iloc[0].to_dict()
    except Exception:
        logger.exception(
            "Could not convert RAGAS result to scores dict. trace_id=%s",
            trace_id,
        )
        return None

    _send_scores_to_langfuse(
        langfuse_client=langfuse_client,
        trace_id=trace_id,
        scores=scores,
    )

    if langfuse_client is not None and hasattr(langfuse_client, "flush"):
        try:
            langfuse_client.flush()
        except Exception:
            logger.exception(
                "Langfuse flush failed. trace_id=%s",
                trace_id,
            )
    return scores


def evaluate_batch(
        rows: list[dict[str, Any]],
        trace_ids: list[str | None],
        *,
        langfuse_client: Any,
        judge_llm: Any,
) -> list[dict[str, Any]] | None:

    if not rows:
        logger.warning("No rows provided for batch evaluation.")
        return None

    metrics = [faithfulness, answer_relevancy]

    data: dict[str, list[Any]] = {
        "question": [r["question"] for r in rows],
        "answer": [r["answer"] for r in rows],
        "contexts": [r["contexts"] for r in rows],
    }

    has_ground_truth_for_all = all(
        str(r.get("ground_truth") or "").strip()
        for r in rows
    )

    if has_ground_truth_for_all:
        data["ground_truth"] = [
            str(r["ground_truth"]).strip()
            for r in rows
        ]
        metrics.extend([context_precision, context_recall])

    dataset = Dataset.from_dict(data)

    try:
        evaluate_result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=judge_llm,
        )
    except Exception:
        logger.exception("RAGAS evaluation failed.")
        return None

    try:
        result_df = evaluate_result.to_pandas()
        if result_df.empty:
            logger.warning("RAGAS returned empty results.")
            return None

        all_scores: list[dict[str, Any]] = []

        for i in range(len(result_df)):
            row_scores = result_df.iloc[i].to_dict()
            _send_scores_to_langfuse(
                langfuse_client=langfuse_client,
                trace_id=trace_ids[i],
                scores=row_scores,
            )
            all_scores.append(row_scores)

    except Exception:
        logger.exception("Could not process RAGAS batch results.")
        return None

    if langfuse_client is not None and hasattr(langfuse_client, "flush"):
        try:
            langfuse_client.flush()
        except Exception:
            logger.exception("Langfuse flush failed.")

    return all_scores

