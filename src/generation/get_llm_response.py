from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

@retry(
    wait=wait_random_exponential(min=1, max=20), # 1초에서 20초 사이 지수적 대기
    stop=stop_after_attempt(3),                  # 최대 3번까지 재시도
    retry=retry_if_exception_type(Exception),     # 모든 예외에 대해 재시도 (실제로는 특정 API 에러만 지정 권장)
    reraise=True                                 # 마지막 시도 실패 시 에러를 그대로 발생시킴
)
def get_llm_response_safe(messages, client, model="gpt-5"):
    """
    OpenAI API 호출을 안전하게 수행하는 함수입니다.
    - timeout: 30초 내에 응답이 없으면 끊습니다.
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        timeout=30.0     # 30초 타임아웃 설정
    )
    return response.choices[0].message.content.strip()