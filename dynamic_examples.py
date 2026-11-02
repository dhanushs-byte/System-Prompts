import json
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).parent
EXAMPLES = [
    ("A duplicate subscription payment needs a refund.", "Billing", "High"),
    ("I cannot log in after my password reset link expired.", "Account", "High"),
    ("My parcel is delayed and tracking has not changed.", "Shipping", "Medium"),
    ("The app crashes whenever I upload a document.", "Technical Issue", "High"),
    ("What payment methods are supported?", "General Inquiry", "Low"),
    ("There is a pending card authorization; is it a completed charge?", "Billing", "Medium"),
    ("How do I update the email on my profile?", "Account", "Low"),
    ("The website is slow but still works.", "Technical Issue", "Medium"),
]

def main():
    print("Loading local embedding model: all-MiniLM-L6-v2")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [x[0] for x in EXAMPLES]
    embeddings = model.encode(texts, normalize_embeddings=True)
    tickets = pd.read_csv(BASE / "data/tickets.csv")
    rows = []
    for _, ticket in tickets.iterrows():
        query = model.encode(ticket["text"], normalize_embeddings=True)
        scores = np.dot(embeddings, query)
        top = np.argsort(scores)[-3:][::-1]
        chosen = [{"example": EXAMPLES[i][0], "category": EXAMPLES[i][1],
                   "priority": EXAMPLES[i][2], "similarity": round(float(scores[i]), 4)}
                  for i in top]
        rows.append({"ticket_id": ticket["ticket_id"], "ticket": ticket["text"],
                     "selected_examples": json.dumps(chosen)})
    out = BASE / "results"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "dynamic_example_selection.csv", index=False)
    print("Completed. Saved results/dynamic_example_selection.csv")

if __name__ == "__main__":
    main()
