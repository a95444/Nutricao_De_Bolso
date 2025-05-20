# inspect_categories.py

import pandas as pd

def main():
    # 1) Carrega apenas as colunas necessárias
    food = pd.read_csv("food.csv", usecols=["fdc_id", "description", "food_category_id"])

    # 2) Agrupa por categoria, conta itens e recolhe um exemplo
    overview = (
        food.groupby("food_category_id")
            .agg(
                count=("fdc_id", "count"),
                example_description=("description", lambda x: x.iloc[0])
            )
            .reset_index()
            .sort_values("food_category_id")
    )

    # 3) Imprime o resultado
    print(f"{'Categoria':<12} {'#Itens':<8} Descrição de Exemplo")
    print("-" * 50)
    for _, row in overview.iterrows():
        cat = int(row["food_category_id"])
        cnt = int(row["count"])
        exempl = row["example_description"]
        print(f"{cat:<12} {cnt:<8} {exempl}")

if __name__ == "__main__":
    main()
