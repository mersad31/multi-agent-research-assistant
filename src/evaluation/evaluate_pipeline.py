from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_openai import ChatOpenAI
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.memory import MemorySaver
from ragas.llms import LangchainLLMWrapper

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.builder import build_graph
from src.config.settings import settings
from src.evaluation.evaluators.ragas_evaluator import evaluate_state, evaluate_batch




logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOLDEN_DATASET_PATH = PROJECT_ROOT / "src" / "evaluation" / "evaluations" / "golden_dataset.json"


def load_golden_dataset() -> list[dict]:
    with GOLDEN_DATASET_PATH.open("r", encoding="utf-8") as file:
        golden_data = json.load(file)

    logger.info(
        "Loaded %d golden examples",
        len(golden_data)
    )

    return golden_data


def init_eval_component() -> dict:

    raw_llm = ChatOpenAI(
        model=settings.EVAL_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0
    )
    judge_llm = LangchainLLMWrapper(llm=raw_llm)

    langfuse_client = get_client()
    if not langfuse_client.auth_check():
        logger.warning("Langfuse authentication failed. Scores will not be sent to Langfuse.")


    graph = build_graph(checkpointer=MemorySaver())


    return  {
        "judge_llm": judge_llm,
        "langfuse_client": langfuse_client,
        "graph": graph,
    }


def run_evaluation(query: str, golden_data: list[dict]) -> None:

    components = init_eval_component()

    langfuse_client = components["langfuse_client"]

    judge_llm = components["judge_llm"]

    handler = CallbackHandler()

    graph = components["graph"]

    config = {
        "callbacks": [handler],
        "configurable": {
            "thread_id": "eval_test"
        }
    }

    logger.info(
        "Running graph for: %s",
        query
    )
    result_state = graph.invoke(
        {
            "query": query,
            "count_retries": 0,
            "max_retries": 3,
        },
        config=config,
    )

    trace_id = getattr(handler, "last_trace_id", None)
    logger.info(
        "Captured Trace ID: %s",
        trace_id
    )

    logger.info("Starting RAGAS evaluation...")
    metric_results = evaluate_state(
        state=result_state,
        trace_id=trace_id,
        golden_dataset=golden_data,
        langfuse_client=langfuse_client,
        judge_llm=judge_llm
    )

    if metric_results:
        logger.info("--- Evaluation Results ---")
        for metric, value in metric_results.items():
            logger.info("%s: %s", metric, value)
    else:
        logger.info("Evaluation failed or returned no results.")



def run_batch_evaluation(golden_data: list[dict]) -> list[dict] | None:
    all_rows = []
    all_trace_ids = []

    components = init_eval_component()

    langfuse_client = components["langfuse_client"]

    judge_llm = components["judge_llm"]

    graph = components["graph"]

    for i, item in enumerate(golden_data):
        query = item["query"]

        handler = CallbackHandler()

        config = {
            "callbacks": [handler],
            "configurable":
                {
                    "thread_id": f"eval_{i}"
                },
        }

        logger.info(
            "Running graph for: %s",
            query
        )
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
            "ground_truth": item.get("ground_truth", ""),
        }
        all_rows.append(rows)

    if not all_rows:
        logger.warning("No valid rows for batch evaluation")
        return None

    logger.info(
        "Running RAGAS batch evaluation for %d queries",
        len(all_rows)
    )

    all_scores = evaluate_batch(
        all_rows,
        all_trace_ids,
        langfuse_client=langfuse_client,
        judge_llm=judge_llm
    )
    if all_scores is None:
        logger.warning("Batch evaluation failed or returned no results.")

    return all_scores


