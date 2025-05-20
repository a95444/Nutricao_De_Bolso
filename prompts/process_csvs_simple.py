import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import numpy as np
import pickle
import re
# Simple RAG pipeline for nutrition dataset

def build_docs_wide(
        food_csv="food.csv",
        nutrient_csv="nutrient.csv",
        food_nutrient_csv="food_nutrient.csv",
        docs_path="docs_wide.pkl"
):
    # 1) Definir categorias a considerar
    categories_filter = [3, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 19, 20, 22]

    # 2) Carregar food com filtro de categorias
    food = pd.read_csv(food_csv, usecols=["fdc_id", "description", "food_category_id"])
    food = food[food["food_category_id"].isin(categories_filter)]

    # 3) Carregar nutriente
    nutr = (
        pd.read_csv(nutrient_csv, usecols=["id", "name", "unit_name"])
        .rename(columns={"id": "nutrient_id", "name": "nutrient_name"})
    )

    # 4) Carregar food_nutrient
    fn = pd.read_csv(food_nutrient_csv, usecols=["fdc_id", "nutrient_id", "amount"])

    # 5) Merge
    df = fn.merge(food, on="fdc_id").merge(nutr, on="nutrient_id")

    # 6) Pivot para wide (cada nutriente numa coluna)
    pivot = (
        df.pivot_table(
            index=["fdc_id", "description"],
            columns="nutrient_name",
            values="amount",
            aggfunc="first"
        )
        .reset_index()
    )

    # 7) Reconstruir o campo 'text' para exibição
    #    – Listamos cada nutriente com valor + unidade
    def make_text(row):
        lines = []
        for nut in pivot.columns[2:]:
            val = row.get(nut)
            if pd.notna(val):
                unit = nutr.loc[nutr.nutrient_name == nut, "unit_name"].iat[0]
                lines.append(f"{nut}: {val} {unit}")
        return "\n".join(lines)

    pivot["text"] = pivot.apply(make_text, axis=1)

    # 8) Guardar
    pivot.to_pickle(docs_path)
    print(f"✅ Documentos salvos em '{docs_path}' com {len(pivot)} itens.")
    return pivot
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
    min_protein=10.0,
    min_nutrient=1.0
):
    import pickle, faiss, pandas as pd, re

    # 1) Carregar artefatos
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    index = faiss.read_index(index_path)
    docs_df = pd.read_pickle(tfidf_path)

    # 2) Detectar nutriente na query (simplesmente procura “vitamina c”, “proteína”, “ferro”...)
    nutrient_map = {
        # Macronutrientes
        r"prote[íi]na": "Protein",
        r"lip[íi]dio(s)?": "Total lipid (fat)",
        r"gordur(a|as)": "Total lipid (fat)",
        r"hidrato(s)? de carbono": "Carbohydrate, by difference",
        r"carboidrato(s)?": "Carbohydrate, by difference",
        r"fibra": "Fiber, total dietary",
        r"energia": "Energy",
        r"caloria(s)?": "Energy",
        r"álcool": "Alcohol, ethyl",

        # Minerais principais
        r"ferro": "Iron, Fe",
        r"c[áa]lcio": "Calcium, Ca",
        r"sódio": "Sodium, Na",
        r"pot[áa]ssio": "Potassium, K",
        r"magnesi(o|u)m": "Magnesium, Mg",
        r"zinco": "Zinc, Zn",
        r"fosforo": "Phosphorus, P",
        r"cobre": "Copper, Cu",
        r"s[ée]lenio": "Selenium, Se",
        r"magn[eê]sio": "Magnesium, Mg",
        r"manganês": "Manganese, Mn",

        # Vitaminas
        r"vitamina\s*a\b": "Vitamin A, IU",
        r"vitamina\s*d\b": "Vitamin D (D2 + D3)",
        r"vitamina\s*e\b": "Vitamin E (alpha-tocopherol)",
        r"vitamina\s*k\b": "Vitamin K (phylloquinone)",
        r"vitamina\s*c\b": "Vitamin C, total ascorbic acid",
        r"vitamina\s*b1\b": "Thiamin",
        r"vitamina\s*b2\b": "Riboflavin",
        r"vitamina\s*b3\b": "Niacin",
        r"vitamina\s*b5\b": "Pantothenic acid",
        r"vitamina\s*b6\b": "Vitamin B-6",
        r"vitamina\s*b7\b": "Biotin",
        r"vitamina\s*b9\b": "Folate, total",
        r"vitamina\s*b12\b": "Vitamin B-12",

        # Ácidos gordos
        r"trans": "Fatty acids, total trans",
        r"saturad[oa]s": "Fatty acids, total saturated",
        r"monoinsaturad[oa]s": "Fatty acids, total monounsaturated",
        r"poliinsaturad[oa]s": "Fatty acids, total polyunsaturated",

        # Outros
        r"a[çc]úcar": "Sugars, Total",
        r"colesterol": "Cholesterol",
        r"agua|água": "Water",
        r"cinco principal": "Ash"
    }

    def detect_nutrient(query: str):
        for pattern, column in nutrient_map.items():
            if re.search(pattern, query, flags=re.IGNORECASE):
                return column
        return None


    target_nutrient = detect_nutrient(query)

    # 3) Expandir query para português e vetorizar
    #query_pt = f"{query} proteína magra gordura baixa"
    q_vec = vectorizer.transform([query]).toarray().astype("float32")

    # 4) Recuperar 20×k candidatos
    D, I = index.search(q_vec, k * 20)
    candidates = docs_df.iloc[I[0]].copy()

    # 5) Extrair valores numéricos de nutrientes (passes a ter todas as colunas no docs_df!)
    #    Aqui assumimos que tens uma coluna por nutriente, ex. docs_df["Vitamin C, total ascorbic acid"]
    #    Se não tiveres, precisarás de parsear o campo "text" para extrair
    if target_nutrient and target_nutrient in docs_df.columns:
        # Filtrar quem tem valor >= min_nutrient nesse nutriente
        print(f"ESTÁ NA COLUNA {target_nutrient}")
        candidates = candidates[candidates[target_nutrient] >= min_nutrient]
    else:
        # Caso contrário, mantém o filtro de proteína original
        def extract_prot(text):
            m = re.search(r"Protein[:\s]+([\d.]+)", text)
            return float(m.group(1)) if m else 0.0
        candidates.loc[:, "ProteinValue"] = candidates["text"].apply(extract_prot)
        candidates = candidates[candidates["ProteinValue"] >= min_protein]

    # 6) Aplicar blacklist/whitelist e deduplicação se quiseres
    #    (igual ao que já tinhas: make_key + drop_duplicates)

    # 7) Por fim, ordenar e devolver os top-k
    #    Podes ordenar pelo valor do nutriente alvo, ou pela proteína se for fallback

    # Exemplo de ordenação pelo nutriente alvo:
    if target_nutrient and target_nutrient in docs_df.columns:
        top = candidates.sort_values(by=target_nutrient, ascending=False).head(k)
    else:
        top = candidates.sort_values(by="ProteinValue", ascending=False).head(k)

    return top[["description", "text", target_nutrient or "ProteinValue"]]




if __name__ == "__main__":
    # Build pipeline
    #docs = build_docs_wide()
    #idx, vec = build_index(docs)

    # Test retrieval
    sample = retrieve_rag("Boa fonte de vitamina e", k=4)
    print("Top 3 RAG results:")
    for _, row in sample.iterrows():
        print(f"{row.description}\n{row.text}\n---")
