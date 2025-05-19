import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import numpy as np
import pickle
import os

def build_docs(
    food_csv="food.csv",
    nutrient_csv="nutrient.csv",
    food_nutrient_csv="food_nutrient.csv",
    basic_categories=[4,5,6,7,8],
    docs_path="fdc_docs.pkl"
):
    # Carregar e filtrar dados
    food = pd.read_csv(food_csv, usecols=["fdc_id","description","food_category_id"])
    nutrient = pd.read_csv(nutrient_csv, usecols=["id","name","unit_name"])
    nutrient.rename(columns={"id":"nutrient_id","name":"nutrient_name"}, inplace=True)
    fn = pd.read_csv(food_nutrient_csv, usecols=["fdc_id","nutrient_id","amount"])
    food = food[food["food_category_id"].isin(basic_categories)]
    fn = fn[fn["fdc_id"].isin(food["fdc_id"])]

    # Merge
    df = fn.merge(food[["fdc_id","description"]], on="fdc_id") \
           .merge(nutrient, on="nutrient_id")

    # Criar docs
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

    # Guardar
    docs_df.to_pickle(docs_path)
    print(f"✅ Documentos salvos em '{docs_path}' ({len(docs_df)} itens)")
    return docs_df

def build_tfidf_index(
    docs_df,
    max_features=10000,
    index_path="tfidf_index.faiss",
    vectorizer_path="tfidf_vectorizer.pkl",
    docs_path="tfidf_docs.pkl"
):
    texts = docs_df["text"].tolist()
    print("🔢 Gerando TF-IDF matrix...")
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(texts).astype(np.float32)

    d = tfidf_matrix.shape[1]
    index = faiss.IndexFlatL2(d)
    print("🗄️ Construindo índice FAISS...")
    index.add(tfidf_matrix.toarray())

    # Guardar artefatos
    faiss.write_index(index, index_path)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    docs_df.to_pickle(docs_path)

    print(f"✅ TF-IDF index salvo em '{index_path}'")
    print(f"✅ Vectorizer salvo em '{vectorizer_path}'")
    print(f"✅ Documentos (TF-IDF) salvo em '{docs_path}'")
    return index, vectorizer

def load_rag(index_path="tfidf_index.faiss",
             vectorizer_path="tfidf_vectorizer.pkl",
             docs_path="tfidf_docs.pkl"):
    index = faiss.read_index(index_path)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    docs_df = pd.read_pickle(docs_path)
    return index, vectorizer, docs_df

def retrieve_rag(query, k=5, index=None, vectorizer=None, docs_df=None):
    if index is None or vectorizer is None or docs_df is None:
        index, vectorizer, docs_df = load_rag()
    q_vec = vectorizer.transform([query]).toarray().astype(np.float32)
    D, I = index.search(q_vec, k)
    return docs_df.iloc[I[0]][["description","text"]]

if __name__ == "__main__":
    # 1. Build docs
    docs_df = build_docs()

    # 2. Build TF-IDF + FAISS
    build_tfidf_index(docs_df)

    # 3. Teste rápido
    idx, vec, df = load_rag()
    sample = retrieve_rag("alimentos ricos em proteínas e baixo teor de gordura", k=3, index=idx, vectorizer=vec, docs_df=df)
    print("\n🔍 Resultados RAG:\n", sample.to_string(index=False))