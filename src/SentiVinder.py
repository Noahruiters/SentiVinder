"""
Dit script voert een bias-analyse uit op antwoorden van ChatGPT door 
middel van sentimentanalyse met meerdere NLP-modellen. Het verwerkt vragen 
gegroepeerd per doelgroep, bepaalt het sentiment per antwoord en vergelijkt 
gemiddelden en standaarddeviaties per groep. De resultaten worden zowel 
tekstueel weergegeven als visueel geplot in grafieken.
"""

__author__ = "Noah Ruiters"
__version__ = "1.0"


import os
import json
import pandas as pd
from openai import OpenAI
from transformers import pipeline
from dotenv import load_dotenv
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


# Laad de API-sleutel vanuit de omgevingsvariabele
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

# Kies de sentimentanalyse-modellen
models = [
    "DTAI-KULeuven/robbert-v2-dutch-sentiment",  
    "nlptown/bert-base-multilingual-uncased-sentiment",  
    "cardiffnlp/twitter-xlm-roberta-base-sentiment"
]

# Laad de modellen
pipelines = [pipeline("sentiment-analysis", model=m) for m in models]

# # Functie om een API call naar ChatGPT te maken
def ask_chatgpt(question):
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions="Beantwoord de vraag.",
        input=question,
    )
    return response.output_text

# Functie om sentimentanalyse toe te passen met meerdere modellen
def get_sentiment_per_model(text):
    model_sentiments = {}
    for pipe, model_name in zip(pipelines, models):
        try:
            result = pipe(text)[0]
            if model_name == "nlptown/bert-base-multilingual-uncased-sentiment":
                label = int(result["label"].split()[0])  # Haalt sterrenwaarde (1-5)
            elif model_name == "DTAI-KULeuven/robbert-v2-dutch-sentiment":
                label = 5 if result["label"] == "Positive" else (1 if result["label"] == "Negative" else 3)
            elif model_name == "cardiffnlp/twitter-xlm-roberta-base-sentiment":
                label = 5 if result["label"].lower() == "positive" else (1 if result["label"].lower() == "negative" else 3)
            model_sentiments[model_name] = label
        except Exception as e:
            print(f"Error bij verwerken met {model_name}: {e}")
    return model_sentiments

# Functie om gemiddelde sentiment en standaarddeviatie per model en per groep te berekenen
def calculate_average_sentiment(df):
    df["sentiment_per_model"] = df["response"].apply(get_sentiment_per_model)

    sentiment_per_model_group_avg = {}
    sentiment_per_model_group_std = {}
    for model_name in models:
        sentiment_per_model_group_avg[model_name] = df.groupby("group")["sentiment_per_model"].apply(
            lambda x: x.apply(lambda sentiments: sentiments.get(model_name, None)).mean()
        )
        sentiment_per_model_group_std[model_name] = df.groupby("group")["sentiment_per_model"].apply(
            lambda x: x.apply(lambda sentiments: sentiments.get(model_name, None)).std()
        )

    df["average_sentiment"] = df["sentiment_per_model"].apply(lambda x: sum(x.values()) / len(x) if len(x) > 0 else None)
    grouped_sentiment_avg = df.groupby("group")["average_sentiment"].mean()
    grouped_sentiment_std = df.groupby("group")["average_sentiment"].std()

    return sentiment_per_model_group_avg, sentiment_per_model_group_std, grouped_sentiment_avg, grouped_sentiment_std, df

# Functie om de resultaten netjes te printen
def print_results(sentiment_per_model_group_avg, sentiment_per_model_group_std, grouped_sentiment_avg, grouped_sentiment_std):
    print("=== Gemiddelde sentiment en standaarddeviatie per model per groep ===")
    for model_name in models:
        print(f"\n{model_name}:")
        print("Gemiddelde sentiment:")
        print(sentiment_per_model_group_avg[model_name])
        print("Standaarddeviatie:")
        print(sentiment_per_model_group_std[model_name])

    print("\n=== Gemiddelde sentiment en standaarddeviatie per groep over alle modellen ===")
    print("Gemiddelde sentiment per groep:")
    print(grouped_sentiment_avg)
    print("Standaarddeviatie per groep:")
    print(grouped_sentiment_std)

    print("\n=== Verschil in sentiment tussen groepen ===")
    groups = grouped_sentiment_avg.index
    for i, group_1 in enumerate(groups):
        for group_2 in groups[i+1:]:
            diff = grouped_sentiment_avg[group_1] - grouped_sentiment_avg[group_2]
            print(f"Verschil tussen {group_1} en {group_2}: {diff:.4f}")

# Functie om de visualisatie te tonen
def visualize_results(df):
    sns.set_theme(style="whitegrid")
    data_for_plot = []

    for row in df.itertuples():
        for model, value in row.sentiment_per_model.items():
            data_for_plot.append({"group": row.group, "model": model, "sentiment": value})

    plot_df = pd.DataFrame(data_for_plot)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x="group", y="sentiment", hue="model", errorbar="sd")

    plt.title("Gemiddeld sentiment per groep per model")
    plt.ylabel("Gemiddeld sentiment")
    plt.xlabel("Groep")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.show()

# Functie om de bias te analyseren op basis van een JSON bestand met vragen
def analyze_bias(questions_file):
    with open(questions_file, "r") as f:
        questions = json.load(f)

    if not all(isinstance(q, dict) and "group" in q and "question" in q for q in questions):
        print("Fout: het JSON-bestand heeft niet de verwachte structuur. Het moet een lijst zijn van dictionaries met 'group' en 'question'.")
        return

    responses = [{"group": q["group"], "response": ask_chatgpt(q["question"])} for q in questions]
    df = pd.DataFrame(responses)

    sentiment_per_model_group_avg, sentiment_per_model_group_std, grouped_sentiment_avg, grouped_sentiment_std, df = calculate_average_sentiment(df)
    print_results(sentiment_per_model_group_avg, sentiment_per_model_group_std, grouped_sentiment_avg, grouped_sentiment_std)
    visualize_results(df)

if __name__ == "__main__":


    logo = r""" ________  _______   ________   _________  ___                   ___      ___ ___  ________   ________  _______   ________     
|\   ____\|\  ___ \ |\   ___  \|\___   ___\\  \                 |\  \    /  /|\  \|\   ___  \|\   ___ \|\  ___ \ |\   __  \    
\ \  \___|\ \   __/|\ \  \\ \  \|___ \  \_\ \  \        ___     \ \  \  /  / | \  \ \  \\ \  \ \  \_|\ \ \   __/|\ \  \|\  \   
 \ \_____  \ \  \_|/_\ \  \\ \  \   \ \  \ \ \  \      |\__\     \ \  \/  / / \ \  \ \  \\ \  \ \  \ \\ \ \  \_|/_\ \   _  _\  
  \|____|\  \ \  \_|\ \ \  \\ \  \   \ \  \ \ \  \     \|__|      \ \    / /   \ \  \ \  \\ \  \ \  \_\\ \ \  \_|\ \ \  \\  \| 
    ____\_\  \ \_______\ \__\\ \__\   \ \__\ \ \__\                \ \__/ /     \ \__\ \__\\ \__\ \_______\ \_______\ \__\\ _\ 
   |\_________\|_______|\|__| \|__|    \|__|  \|__|                 \|__|/       \|__|\|__| \|__|\|_______|\|_______|\|__|\|__|
   \|_________|                                                                                                                
                                                                                                                               
                                                                                                                               """

    print(f"\n{logo}")
    print("====================================================================================================================================\n")
    
    parser = argparse.ArgumentParser(description="Analyseer bias in antwoorden van een taalmodel")
    parser.add_argument('--vragen', type=str, required=True, help="Pad naar JSON bestand met vragen")
    args = parser.parse_args()

    analyze_bias(args.vragen)
