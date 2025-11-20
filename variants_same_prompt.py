# variants_same_prompt.py
from customer_service_langgraph import (
    ChatState, ml_churn_predict, aggregate_risk,
    recommend_products, escalate_if_needed, final_response
)

PROMPT = "Please review my case: I contacted support last week and still haven't heard back."

def generate_variant(sentiment: str) -> str:
    frus = {"negative": "high", "neutral": "medium", "positive": "low"}[sentiment]
    # Pre-seed state with the sentiment you want to test
    st = ChatState(
        messages=[PROMPT],
        topic="support",
        sentiment=sentiment,
        frustration_level=frus,
        churn_risk_llm="likely" if sentiment == "negative" else "unlikely",
        negative_streak=1 if sentiment == "negative" else 0,
    )
    # walk the same path the graph would (minus the classifier)
    ml_churn_predict(st)
    aggregate_risk(st)
    recommend_products(st)
    escalate_if_needed(st)
    final_response(st)
    return st.final_reply or ""

for s in ["negative", "neutral", "positive"]:
    print(f"\n== {s.upper()} ==")
    print(generate_variant(s))
