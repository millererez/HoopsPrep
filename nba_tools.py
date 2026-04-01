import pandas as pd
from simple_rag import get_top_insights


def get_player_stats(player_name):
    df = pd.read_csv("data/nba_players.csv")
    match = df[df["player"].str.lower() == player_name.lower()]
    if match.empty:
        return "Player not found"
    row = match.iloc[0]
    return (
        f"{row['player']} - PPG: {row['ppg']}, RPG: {row['rpg']}, "
        f"APG: {row['apg']}, Status: {row['status']}"
    )


if __name__ == "__main__":
    print(get_player_stats("Nikola Jokic"))
    print(get_player_stats("Anthony Edwards"))

    print("\n--- RAG: How does Minnesota defend? ---\n")
    insights = get_top_insights("How does Minnesota defend?")
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
