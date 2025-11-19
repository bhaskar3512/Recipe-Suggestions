from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re

app = Flask(__name__)

vectorizer = TfidfVectorizer()


# Load CSV and convert ingredients string to list
recipes = pd.read_csv("data.csv", converters={
    "ingredients": lambda x: [i.strip() for i in x.split(",")]
})

print(recipes)


def canonicalize(ing):
    return ing.lower().strip()

recipes["ing_str"] = recipes["ingredients"].apply(lambda lst: " ".join(sorted(set(canonicalize(x) for x in lst))))
vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
tfidf_matrix = vectorizer.fit_transform(recipes["ing_str"])

def strip_quant(s):
    return re.sub(r"[\d/]+|ml|g|kg|cup|cups|tbsp|tsp", "", s, flags=re.I).strip()

def suggest_recipes(user_ingredients, top_k=5, min_score=0.1):
    user_tokens = [canonicalize(strip_quant(x)) for x in user_ingredients if x.strip()]
    user_doc = " ".join(sorted(set(user_tokens)))
    user_vec = vectorizer.transform([user_doc])
    sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
    idx_sorted = sims.argsort()[::-1]
    suggestions = []
    user_set = set(user_tokens)
    for idx in idx_sorted[:top_k]:
        score = float(sims[idx])
        if score < min_score:
            continue
        rec = recipes.iloc[idx]
        rec_set = set(canonicalize(x) for x in rec["ingredients"])
        missing = list(rec_set - user_set)
        have = list(rec_set & user_set)
        suggestions.append({
            "title": rec["title"],
            "score": round(score, 3),
            "have": sorted(have),
            "missing": sorted(missing),
            "ingredients": sorted(rec_set)
        })
    return suggestions

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form["ingredients"]
        user_ingredients = [x.strip() for x in user_input.split(",")]
        results = suggest_recipes(user_ingredients, top_k=5)
        return render_template("results.html", results=results, user_ingredients=user_ingredients)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
