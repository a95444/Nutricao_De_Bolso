import pandas as pd
from ollama import embed
import faiss
import numpy as np

# 1. Ficheiros de entrada
food_csv           = "food.csv"
nutrient_csv       = "nutrient.csv"
food_nutrient_csv  = "food_nutrient.csv"

# 2. Carregar dados
food = pd.read_csv(food_csv, usecols=["fdc_id", "description"])
nutr = pd.read_csv(nutrient_csv, usecols=["id", "name", "unit_name"])
nutr.rename(columns={"id":"nutrient_id","name":"nutrient_name"}, inplace=True)
fn   = pd.read_csv(food_nutrient_csv, usecols=["fdc_id","nutrient_id","amount"])

# 3. Merge para juntar tudo
df = fn.merge(food, on="fdc_id").merge(nutr, on="nutrient_id")

# 4. Criar lista de documentos manualmente
docs = []
for fdc_id, group in df.groupby("fdc_id"):
    text = "\n".join(
        f"{row['nutrient_name']}: {row['amount']} {row['unit_name']}"
        for _, row in group.iterrows()
    )
    docs.append({
        "id": int(fdc_id),
        "description": group["description"].iloc[0],
        "text": text
    })

docs_df = pd.DataFrame(docs)

# 5. Gerar embeddings com modelo dedicado
texts = docs_df["text"].tolist()
embeddings = embed("nomic-embed-text", texts)

# 6. Construir índice FAISS
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(np.array(embeddings))

# 7. Guardar índice e metadados
faiss.write_index(index, "fdc_index.faiss")
docs_df.to_pickle("fdc_docs.pkl")

print("✅ FAISS index and docs prontos!")
