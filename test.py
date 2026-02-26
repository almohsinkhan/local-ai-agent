from app.graph import build_graph

def main():
    graph = build_graph()

    # Print Mermaid diagram syntax
    mermaid_code = graph.get_graph().draw_mermaid()
    print("\n=== MERMAID WORKFLOW ===\n")
    print(mermaid_code)


if __name__ == "__main__":
    main()