from dotenv import load_dotenv

from src.graph.builder import build_graph


def main():
    load_dotenv()

    graph = build_graph(interrupt_before=["publisher"])

    initial_state = {
        "messages": [],
        "query": "which materials used for product a gasoline engine?",          # write a research question
        "sub_tasks": [],
        "current_sub_tasks": [],
        "sources": [],
        "findings": [],
        "report": "",
        "current_node": None,
        "max_retries": 2,
        "count_retries": 0,
        "errors": [],
        "is_sufficient": None,
        "review_feedback": None,
    }

    config = {
        "configurable": {
            "thread_id": "test-session-1"
        }
    }

    # -------------------------
    # Phase 1: Execute until interrupt
    # -------------------------
    result = graph.invoke(initial_state, config=config)

    print("sub_tasks:", result.get("sub_tasks"))
    print("sources:", result.get("sources"))
    print("errors:", result.get("errors"))
    print("count_retries:", result.get("count_retries"))
    print("is_sufficient:", result.get("is_sufficient"))

    print("\n⏸️ Graph paused before publisher.")
    print("=" * 60)

    print("\nSources:")
    for source in result.get("sources", []):
        print("-", source)

    print("\nReview feedback:")
    print(result.get("review_feedback"))

    input("\nPress Enter to continue...")

    # -------------------------
    # Phase 2: Resume execution
    # -------------------------
    result = graph.invoke(None, config=config)
    print("\n📄 Final Report")
    print("=" * 60)
    print(result.get("report", ""))


if __name__ == "__main__":
    main()