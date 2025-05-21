dic = {
    "dog": ["cão", "cachorro"],
    "table": ["mesa", "tabela"]
}


dic_invertido={}
for elemento in dic.keys():
    for valor in dic[elemento]:
        dic_invertido[valor]=elemento

print(dic_invertido)