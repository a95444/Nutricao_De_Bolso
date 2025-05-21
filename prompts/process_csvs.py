import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import numpy as np
import pickle
import re
from tabulate import tabulate  # Adicionado para formatação da tabela


# Simple RAG pipeline for nutrition dataset

def build_docs_wide(
        food_csv="food.csv",
        nutrient_csv="nutrient.csv",
        food_nutrient_csv="food_nutrient.csv",
        docs_path="docs_wide.pkl"
):
    # [Manter o mesmo código anterior...]
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
    # [Manter o mesmo código anterior...]
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
        min_protein=10.0,
        min_nutrient=1.0
):
    # [Manter o mesmo código anterior até a detecção do nutriente...]
    import pickle, faiss, pandas as pd, re

    # 1) Carregar artefatos
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    index = faiss.read_index(index_path)
    docs_df = pd.read_pickle(tfidf_path)

    # 2) Detectar nutriente na query
    nutrient_map = {
        # ... [Manter o mesmo mapeamento anterior] ...
        r"prote[íi]na": "Protein",
        r"lip[íi]dio(s)?": "Total lipid (fat)",
        r"gordur(a|as)": "Total lipid (fat)",
        r"hidrato(s)? de carbono": "Carbohydrate, by difference",
        r"carboidrato(s)?": "Carbohydrate, by difference",
        r"fibra": "Fiber, total dietary",
        r"energia": "Energy",
        r"caloria(s)?": "Energy",
        r"álcool": "Alcohol, ethyl",
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
        r"trans": "Fatty acids, total trans",
        r"saturad[oa]s": "Fatty acids, total saturated",
        r"monoinsaturad[oa]s": "Fatty acids, total monounsaturated",
        r"poliinsaturad[oa]s": "Fatty acids, total polyunsaturated",
        r"a[çc]úcar": "Sugars, Total",
        r"colesterol": "Cholesterol",
        r"agua|água": "Water",
        r"cinco principal": "Ash"
    }

    def detect_nutrients(query: str, max_hits=2):
        hits = []
        for pattern, col in nutrient_map.items():
            if re.search(pattern, query, flags=re.IGNORECASE):
                hits.append(col)
                if len(hits) >= max_hits:
                    break
        return hits

    target_nutrients = detect_nutrients(query)

    # 3) Vetorizar e recuperar candidatos
    q_vec = vectorizer.transform([query]).toarray().astype("float32")
    D, I = index.search(q_vec, k * 20)
    candidates = docs_df.iloc[I[0]].copy()

    # 4) Filtrar pelos nutrientes detectados (ou proteína como fallback)
    if target_nutrients:
        for nut in target_nutrients:
            if nut in candidates.columns:
                candidates = candidates[candidates[nut].fillna(0) >= min_nutrient]
    else:
        def extract_prot(text):
            m = re.search(r"Protein[:\s]+([\d.]+)", text)
            return float(m.group(1)) if m else 0.0
        candidates.loc[:, "ProteinValue"] = candidates["text"].apply(extract_prot)
        candidates = candidates[candidates["ProteinValue"] >= min_protein]
        target_nutrients = ["ProteinValue"]

    # 5) Ordenar pelos nutrientes (dando prioridade ao primeiro, depois ao segundo se houver)
    sort_by = [nut for nut in target_nutrients if nut in candidates.columns]
    candidates = candidates.sort_values(by=sort_by, ascending=False)

    blacklist = r"bear|owl|game meat|herring|dried|canned|squirrel|ostrich"
    mask_blacklist = ~candidates["description"].str.lower().str.contains(blacklist)
    candidates = candidates[mask_blacklist]


    # 6) **Deduplicar pela “primeira palavra” da descrição**
    candidates.loc[:, "first_word"] = (
        candidates["description"]
        .str.split()
        .str[0]
        .str.lower()
    )

    candidates = candidates.drop_duplicates(subset="first_word")
    candidates = candidates.drop(columns="first_word")
    top = candidates.head(k)

    # 8) Preparar colunas de saída
    cols = ['description', 'Energy', 'Protein']
    display_cols = {
        'description': 'Alimento',
        'Energy':      'Calorias (kcal)',
        'Protein':     'Proteína (g)'
    }

    unit = None
    for nut in sort_by:
        cols.append(nut)
        # extrair unidade do primeiro texto que tiver esse nutriente
        for line in top['text'].iloc[0].split('\n'):
            if line.startswith(nut + ':'):
                unit = line.split()[-1]
                break
        display_cols[nut] = f"{nut} ({unit})" if unit else nut

    result_df = top[cols].rename(columns=display_cols)
    return result_df


if __name__ == "__main__":
    # Build pipeline
    #docs = build_docs_wide()
    #idx, vec = build_index(docs)

    # Test retrieval
    sample = retrieve_rag(" Ótimo trabalho! Podes fazer agora uma tabela que compare diferentes alimentos ricos em Proteína e Ferro.", k=10)
    print("Top 4 RAG results:\n")
    # imprime toda a tabela de uma vez:
    print(tabulate(sample,
                   headers=sample.columns,
                   tablefmt="psql",
                   showindex=False))
