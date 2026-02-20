from rag_pipeline import build_query_engine

def main():
    engine = build_query_engine()

    query = "What is fat-tailedness?"
    response = engine.query(query)

    print(response)

if __name__ == "__main__":
    main()
