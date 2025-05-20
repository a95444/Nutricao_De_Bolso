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
    #categorias a serem consideradas:
    categories_filter = [3,5,6,7,8,9,10,12,13,15,16,17,19,20,22]
    # 1. Load minimal columns
    food = pd.read_csv(food_csv, usecols=["fdc_id", "description", "food_category_id"])
    food = food[food["food_category_id"].isin(categories_filter)]
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

#versao boa para prota
def retrieve_rag(
    query, k=5,
    vectorizer_path="vectorizer.pkl",
    index_path="index.faiss",
    tfidf_path="tfidf_docs.pkl",
    min_protein=10.0
):
    import pickle, faiss, pandas as pd, re

    # 1) Carregar artefatos
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    index = faiss.read_index(index_path)
    docs_df = pd.read_pickle(tfidf_path)

    # 2) Expandir query para português
    query_pt = f"{query} proteína magra gordura baixa"
    q_vec = vectorizer.transform([query_pt]).toarray().astype("float32")

    # 3) Recuperar 20×k candidatos
    D, I = index.search(q_vec, k * 20)
    candidates = docs_df.iloc[I[0]].copy()

    # 4) Extrair proteína
    def extract_prot(text):
        m = re.search(r"Protein[:\s]+([\d.]+)", text)
        return float(m.group(1)) if m else 0.0
    candidates.loc[:, "ProteinValue"] = candidates["text"].apply(extract_prot)

    # 5) Blacklist de termos não desejados
    blacklist = r"bear|owl|game meat|herring|dried|canned|squirrel"
    mask_blacklist = ~candidates["description"].str.lower().str.contains(blacklist)

    # 6) Whitelist de termos úteis
    whitelist = r"chicken|salmon|tuna|lentil|tofu|egg|beef|pork|yogurt|turkey|meat"
    mask_whitelist = candidates["description"].str.lower().str.contains(whitelist)

    # 7) Aplica todos os filtros
    filtered = candidates[
        (candidates["ProteinValue"] >= min_protein) &
        mask_whitelist &
        mask_blacklist
    ].copy()

    # Define uma lista de padrões principais


    def make_key(desc):
        patterns = ["tofu", "chicken", "salmon", "tuna", "lentil", "egg", "beef", "pork", "yogurt", "turkey","meat"]
        low = desc.lower()
        for pat in patterns:
            if pat in low:
                return pat
        # fallback ao nome antes da vírgula
        return low.split(",", 1)[0].strip()

    filtered.loc[:, "group_key"] = filtered["description"].apply(make_key)
    filtered = filtered.drop_duplicates(subset=["group_key"])

    # 9) Ordena por proteína e seleciona top k
    top = filtered.sort_values("ProteinValue", ascending=False).head(k)

    # 10) Retorna description, text e ProteinValue
    return top[["description", "text", "ProteinValue"]]



if __name__ == "__main__":
    # Build pipeline
    #docs = build_docs()
    #idx, vec = build_index(docs)

    # Test retrieval
    sample = retrieve_rag("Vitamin C sources low fat", k=20)
    print("Top 3 RAG results:")
    for _, row in sample.iterrows():
        print(f"{row.description}\n{row.text}\n---")
