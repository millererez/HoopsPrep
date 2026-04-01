import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load environment variables
load_dotenv()

# 2. Read narratives and clean them
if not os.path.exists("data/narratives.txt"):
    print("Error: data/narratives.txt not found!")
    texts = []
else:
    with open("data/narratives.txt", "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f.readlines() if line.strip()]

# 3. Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(texts, embedding=embeddings)

# 4. Query function
def get_top_insights(query, k=3):
    results = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in results]

# 5. Test the RAG
if __name__ == "__main__":
    test_query = "How does Denver's offense work?"
    print(f"\n--- Testing RAG with query: '{test_query}' ---\n")

    insights = get_top_insights(test_query)

    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
