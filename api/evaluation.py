from __future__ import annotations

import logging

from fastapi import APIRouter
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel
from starlette.background import BackgroundTasks

from src.evaluation.evaluate_pipeline import init_eval_component, load_golden_dataset
from src.evaluation.evaluators.ragas_evaluator import evaluate_batch, evaluate_state


class EvaluateRequest(BaseModel):
    query: str

class BatchEvaluateRequest(BaseModel):
    use_golden_dataset: bool = False
    queries: list[str] = []

router = APIRouter(prefix="/evaluate", tags=["evaluation"])

logger = logging.getLogger(__name__)


@router.post("")
def evaluate_single(request: EvaluateRequest) -> dict:
    components = init_eval_component()

    judge_llm = components["judge_llm"]
    graph = components["graph"]
    langfuse_client = components["langfuse_client"]
    handler = CallbackHandler()

    config = {
        "callbacks": [handler],
        "configurable": {
            "thread_id": "eval_test"
        }
    }

    result_state = graph.invoke(
        {
            "query": request.query,
            "count_retries": 0,
            "max_retries": 3,
        },
        config=config,
    )

    trace_id = getattr(handler, "last_trace_id", None)

    try:
        evaluate_result = evaluate_state(
            state=result_state,
            trace_id=trace_id,
            golden_dataset=None,
            langfuse_client=langfuse_client,
            judge_llm=judge_llm,
        )
        return evaluate_result

    except Exception:  # noqa: BLE001 — evaluation must never crash the API; log/report failure instead
        return {"success": False, "message": "No valid row for evaluation"}


@router.post("/batch")
async def batch_evaluate(request: BatchEvaluateRequest, background_tasks: BackgroundTasks):

    def _run_batch(queries: list[str]) -> None:
        components = init_eval_component()

        judge_llm = components["judge_llm"]
        graph = components["graph"]
        langfuse_client = components["langfuse_client"]

        ground_truth_by_query: dict[str, str] = {}

        if request.use_golden_dataset:
            golden_data = load_golden_dataset()
            queries = [item["query"] for item in golden_data]
            ground_truth_by_query = {
                item["query"]: item.get("ground_truth", "")
                for item in golden_data
            }

        all_trace_ids = []
        all_rows = []

        for i, query in enumerate(queries):

            handler = CallbackHandler()

            config = {
                "callbacks": [handler],
                "configurable":
                    {
                        "thread_id": f"eval_{i}"
                    },
            }

            result_state = graph.invoke(
                {
                    "query": query,
                    "count_retries": 0,
                    "max_retries": 3,
                },
                config=config,
            )

            trace_id = getattr(handler, "last_trace_id", None)
            all_trace_ids.append(trace_id)

            contexts = [s["content"] for s in result_state.get("sources", []) if s.get("content")]
            reports = result_state["report"]

            if not contexts or not reports:
                continue

            rows = {
                "question": query,
                "answer": reports,
                "contexts": contexts,
                "ground_truth": ground_truth_by_query.get(query, ""),
            }
            all_rows.append(rows)

        if not all_rows:
            logger.warning("No valid rows for batch evaluation")
            return

        all_scores = evaluate_batch(
            all_rows,
            all_trace_ids,
            langfuse_client=langfuse_client,
            judge_llm=judge_llm
        )
        if all_scores is None:
            logger.warning("Batch evaluation failed or returned no results.")
        else:
            logger.info("Batch Completed. %d scores computed.", len(all_scores))

    background_tasks.add_task(_run_batch, request.queries)
    return {
        "success": True,
        "message": "Batch evaluation started",
        "query_count": len(request.queries)
    }
