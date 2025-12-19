from dotenv import load_dotenv

load_dotenv()

from pipelines.rag_pipeline import run

if __name__ == "__main__":
    result = run("산학협력단 관련 사업 내용을 요약해 줘")

    print("=== ANSWER ===")
    print(result["answer"])
