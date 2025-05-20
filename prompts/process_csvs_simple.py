import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import numpy as np
import pickle
import re
# Simple RAG pipeline for nutrition dataset

def build_docs(food_csv="food.csv",
               nutrient_csv="nutrient.csv",
               food_nutrient_csv="food_nutrient.csv",
               docs_path="docs.pkl"):
    """
    1. Load CSVs
    2. Merge on fdc_id/nutrient_id
    3. Group by food item, concatenate nutrient lines
    4. Save DataFrame to pickle
    """
    # 1. Load minimal columns
    food = pd.read_csv(food_csv, usecols=["fdc_id", "description"])
    nutr = pd.read_csv(nutrient_csv, usecols=["id", "name", "unit_name"])
    nutr.rename(columns={"id":"nutrient_id", "name":"nutrient_name"}, inplace=True)
    fn = pd.read_csv(food_nutrient_csv, usecols=["fdc_id", "nutrient_id", "amount"])

    # 2. Merge
    df = fn.merge(food, on="fdc_id").merge(nutr, on="nutrient_id")

    # 3. Group and build text per food
    docs = []
    for fdc_id, grp in df.groupby("fdc_id"):
        desc = grp["description"].iloc[0]
        lines = [
            f"{row['nutrient_name']}: {row['amount']} {row['unit_name']}"
            for _, row in grp.iterrows()
        ]
        docs.append({"fdc_id": fdc_id, "description": desc, "text": "\n".join(lines)})

    docs_df = pd.DataFrame(docs)
    docs_df.to_pickle(docs_path)
    print(f"Saved {len(docs_df)} documents to {docs_path}")
    return docs_df

def build_index(docs_df,
                vectorizer_path="vectorizer.pkl",
                index_path="index.faiss",
                tfidf_path="tfidf_docs.pkl",
                max_features=5000):
    """
    1. Fit TF-IDF on docs_df['text']
    2. Build FAISS index
    3. Save vectorizer, index, and docs_df
    """
    texts = docs_df["text"].tolist()
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf = vectorizer.fit_transform(texts).astype(np.float32)

    # FAISS
    d = tfidf.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(tfidf.toarray())

    # Save artifacts
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    faiss.write_index(index, index_path)
    docs_df.to_pickle(tfidf_path)

    print(f"Saved vectorizer -> {vectorizer_path}, index -> {index_path}, docs -> {tfidf_path}")
    return index, vectorizer

def retrieve_rag(
    query, k=5,
    vectorizer_path="vectorizer.pkl",
    index_path="index.faiss",
    tfidf_path="tfidf_docs.pkl",
    min_protein=10.0
):
    import pickle, faiss
    import pandas as pd
    # 1) Carregar artefatos
    with open(vectorizer_path, "rb") as f: vectorizer = pickle.load(f)
    index = faiss.read_index(index_path)
    docs = pd.read_pickle(tfidf_path)

    # 2) Vetorizar e recuperar 10*k candidatos
    q = vectorizer.transform([query]).toarray().astype("float32")
    D,I = index.search(q, k*10)
    candidates = docs.iloc[I[0]].copy()

    # 3) Extrair valor numérico de proteína do texto
    def extract_protein(text):
        m = re.search(r"Protein:\s*([\d.]+)", text)
        return float(m.group(1)) if m else 0.0

    candidates["ProteinValue"] = candidates["text"].apply(extract_protein)

    # 4) Filtrar bebidas e proteína insuficiente
    candidates = candidates[
        (candidates["ProteinValue"] >= min_protein) &
        (~candidates["description"].str.lower().str.contains("wine|beverage|drink"))
    ]

    # 5) Ordenar por proteína descendente e pegar top k
    top = candidates.sort_values("ProteinValue", ascending=False).head(k)

    # 6) Formatar
    return top[["description", "text", "ProteinValue"]]

if __name__ == "__main__":
    # Build pipeline
    #docs = build_docs()
    #idx, vec = build_index(docs)

    # Test retrieval
    sample = retrieve_rag("protein sources low fat", k=6)
    print("Top 3 RAG results:")
    for _, row in sample.iterrows():
        print(f"{row.description}\n{row.text}\n---")
