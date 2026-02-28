from llama_index_ingestion_pipeline import build_query_engine

def main():
    engine = build_query_engine()

    query = "List top rated movies."
    response = engine.query(query)

    print(response)

if __name__ == "__main__":
    main()
