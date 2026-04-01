import os
from dotenv import load_dotenv
from openai import OpenAI
from nba_tools import get_player_stats
from simple_rag import get_top_insights

load_dotenv()
client = OpenAI()


def generate_report(user_request):
    context_parts = []

    # Player stats
    if "jokic" in user_request.lower():
        context_parts.append(f"Player Stats: {get_player_stats('Nikola Jokic')}")
    if "edwards" in user_request.lower():
        context_parts.append(f"Player Stats: {get_player_stats('Anthony Edwards')}")

    # Tactical insights
    if "denver" in user_request.lower() or "jokic" in user_request.lower():
        insights = get_top_insights("How does Denver's offense work?")
        context_parts.append("Denver Tactical Insights:\n" + "\n".join(f"- {i}" for i in insights))
    if "minnesota" in user_request.lower() or "edwards" in user_request.lower():
        insights = get_top_insights("How does Minnesota defend?")
        context_parts.append("Minnesota Tactical Insights:\n" + "\n".join(f"- {i}" for i in insights))

    context = "\n\n".join(context_parts)

    prompt = f"""You are a professional, high-energy Israeli sports commentator in the style of Sport 5 broadcasters.
Write an exciting, passionate pre-game report in HEBREW.
Use dramatic language, build anticipation, and bring the energy of a live broadcast.

User Request: {user_request}

Retrieved Data:
{context}

Write the full report in Hebrew now:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    user_request = "Prepare a pre-game report for Nikola Jokic vs Anthony Edwards, include stats and tactical analysis for both teams."
    report = generate_report(user_request)
    print(report)
