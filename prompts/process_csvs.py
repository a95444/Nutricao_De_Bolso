import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import numpy as np
import pickle
import re

# Configurações otimizadas
NUTRIENT_PRIORITY = {
    'Protein': 5, 'Total lipid (fat)': 4, 'Carbohydrate': 2,
    'Fiber': 2, 'Energy': 3, 'Sugars, Total': 1
}

BLACKLISTED_KEYWORDS = {
    'powder', 'instant', 'beverage', 'artificial', 'processed',
    'concentrate', 'fortified', 'supplement', 'drink', 'shake'
}

VALID_CATEGORIES = {
    'Fresh Meat': [1, 2],  # Carnes frescas
    'Fresh Fish': [3],  # Peixes frescos
    'Natural Dairy': [4],  # Laticínios naturais
    'Legumes': [5]  # Leguminosas
}


def convert_energy_units(df):
    """Padroniza todas as unidades de energia para kcal/100g"""
    energy_mask = df['nutrient_name'].str.contains('Energy', case=False)

    # Converter kJ para kcal onde necessário
    df.loc[energy_mask & (df['unit_name'] == 'kJ'), 'amount'] = df['amount'] / 4.184
    df.loc[energy_mask, 'unit_name'] = 'kcal'

    return df


def filter_processed_foods(description):
    """Filtra alimentos ultraprocessados pela descrição"""
    desc_lower = description.lower()
    return not any(keyword in desc_lower for keyword in BLACKLISTED_KEYWORDS)


def normalize_serving(group):
    """Normaliza todos os nutrientes registrados no grupo para base 100 g."""
    description = group['description'].iloc[0].lower()

    # Extrair tamanho da porção da descrição (ex: "porção de 50g")
    serving_match = re.search(r'(\d+)\s*g', description)
    if serving_match:
        serving_size = float(serving_match.group(1))
        scaling_factor = 100 / serving_size

        # Aplicar escala a toda a coluna 'amount'
        group.loc[:, 'amount'] = group['amount'] * scaling_factor

    return group

def validate_nutrients(row):
    """Validação rigorosa do perfil nutricional"""
    try:
        protein = row.get('Protein', 0)
        fat = row.get('Total lipid (fat)', 0)
        energy = row.get('Energy', 0)

        # Critérios de qualidade
        valid = (
                protein >= 10 and  # Mínimo 10g de proteína
                fat <= protein / 2 and  # Máximo metade da proteína em gordura
                energy <= 300 and  # Máximo 300kcal/100g
                protein / energy >= 0.3  # Pelo menos 30% das calorias de proteína
        )
        return valid
    except Exception as e:
        print(f"Erro na validação: {e}")
        return False


def build_docs(
        food_csv="food.csv",
        nutrient_csv="nutrient.csv",
        food_nutrient_csv="food_nutrient.csv",
        docs_path="fdc_docs.pkl"
):
    # Carregar e processar dados
    food = pd.read_csv(food_csv, usecols=["fdc_id", "description", "food_category_id"])
    nutrient = pd.read_csv(nutrient_csv, usecols=["id", "name", "unit_name"])
    nutrient.rename(columns={"id": "nutrient_id", "name": "nutrient_name"}, inplace=True)

    fn = pd.read_csv(food_nutrient_csv)

    # Pipeline modificado
    df = (
        fn.merge(food, on="fdc_id")
        .merge(nutrient, on="nutrient_id")
        .pipe(convert_energy_units)
    )

    # Aplicar normalização por grupo
    df = df.groupby("fdc_id").apply(normalize_serving)

    # Filtrar categorias e processados
    valid_cats = [cat for cats in VALID_CATEGORIES.values() for cat in cats]
    df = df[
        df["food_category_id"].isin(valid_cats) &
        df['description'].apply(filter_processed_foods)
        ]

    docs = []
    for fdc_id, group in df.groupby("fdc_id"):
        nutrient_data = {}
        nutrients = []

        for _, row in group.iterrows():
            nutrient_name = row['nutrient_name']
            amount = round(row['amount'], 2)
            unit = row['unit_name']

            # Repetir nutrientes prioritários
            repeats = NUTRIENT_PRIORITY.get(nutrient_name, 1)
            nutrients.extend([f"{nutrient_name}: {amount} {unit}"] * repeats)

            nutrient_data[nutrient_name] = amount

        # Normalizar por 100g
        #nutrient_data = normalize_serving(nutrient_data)

        # Validação final
        if validate_nutrients(nutrient_data):
            docs.append({
                "id": int(fdc_id),
                "description": re.sub(r'\s+', ' ', group["description"].iloc[0]).strip(),
                "text": " ".join(sorted(
                    nutrients,
                    key=lambda x: NUTRIENT_PRIORITY.get(x.split(':')[0].strip(), 0),
                    reverse=True
                )),
                **nutrient_data
            })

    docs_df = pd.DataFrame(docs)
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

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 3),
        token_pattern=r'(?u)\b[\w.:/]+\b',
        stop_words=None
    )

    tfidf_matrix = vectorizer.fit_transform(texts).astype(np.float32)
    index = faiss.IndexFlatL2(tfidf_matrix.shape[1])
    index.add(tfidf_matrix.toarray())

    faiss.write_index(index, index_path)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    docs_df.to_pickle(docs_path)

    print(f"✅ Índice TF-IDF/FAISS salvo em '{index_path}'")
    return index, vectorizer


def retrieve_rag(
        query,
        k=5,
        index=None,
        vectorizer=None,
        docs_df=None,
        nutrient_filters=None
):
    # Filtros nutricionais rigorosos
    default_filters = {
        'Protein': lambda x: x >= 15,
        'Total lipid (fat)': lambda x: x <= 8,
        'Energy': lambda x: x <= 250,
        'Carbohydrate': lambda x: x <= 10
    }

    # Expandir query com sinônimos
    query_expanded = f"{query} proteína magra baixa gordura natural fonte"
    q_vec = vectorizer.transform([query_expanded]).toarray().astype(np.float32)

    # Busca com janela ampliada
    _, I = index.search(q_vec, k * 10)
    results = docs_df.iloc[I[0]].copy()

    # Aplicar filtros dinamicamente
    filters = nutrient_filters or default_filters
    for nutrient, condition in filters.items():
        if nutrient in results.columns:
            results = results[results[nutrient].apply(condition)]

    # Ordenação por qualidade nutricional
    results = results.assign(
        protein_ratio=results['Protein'] / results['Total lipid (fat)']
    ).sort_values(
        by=['protein_ratio', 'Protein'],
        ascending=[False, False]
    ).head(k)

    # Formatar resultados
    results['formatted'] = results.apply(
        lambda r: (
            f"{r.description}\n"
            f"▶ Proteína: {r.Protein}g | "
            f"▶ Gordura: {r['Total lipid (fat)']}g | "
            f"▶ Calorias: {r.Energy}kcal\n"
            f"▶ Densidade Proteica: {r.Protein / r.Energy:.2f}g/kcal"
        ), axis=1
    )

    return results[['formatted', 'text']]


if __name__ == "__main__":
    # Pipeline completo
    docs_df = build_docs()
    index, vectorizer = build_tfidf_index(docs_df)

    # Verificação de dados
    print("\n🔍 Amostra de alimentos válidos:")
    print(docs_df[['description', 'Protein', 'Total lipid (fat)', 'Energy']]
          .sample(3, random_state=42)
          .to_string(index=False))

    # Busca otimizada
    sample = retrieve_rag(
        "fontes naturais de proteína magra",
        k=3,
        index=index,
        vectorizer=vectorizer,
        docs_df=docs_df
    )

    print("\n🔍 Top Recomendações:")
    print(sample['formatted'].to_string(index=False))