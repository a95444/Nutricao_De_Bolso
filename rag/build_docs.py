# build_docs.py

import pandas as pd

# 1. Carregar os CSVs mínimos
food = pd.read_csv("food.csv", usecols=["fdc_id","description","food_category_id"])
nutr = pd.read_csv("nutrient.csv", usecols=["id","name","unit_name"])
fn   = pd.read_csv("food_nutrient.csv", usecols=["fdc_id","nutrient_id","amount"])

# 2. Renomear colunas para merge
nutr.rename(columns={"id":"nutrient_id","name":"nutrient_name"}, inplace=True)

# 3. Filtrar categorias básicas
basic_cats = [4,5,6,7,8]  # Ajusta conforme o teu CSV
food = food[food["food_category_id"].isin(basic_cats)]
fn   = fn[fn["fdc_id"].isin(food["fdc_id"])]

# 4. Merge para criar cada documento
df = fn.merge(food[["fdc_id","description"]], on="fdc_id") \
       .merge(nutr, on="nutrient_id")

# 5. Construir lista de docs
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

# 6. Guardar
docs_df.to_pickle("fdc_docs.pkl")
print(f"✅ Gerados {len(docs_df)} documentos em 'fdc_docs.pkl'")
