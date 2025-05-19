from ollama import embed
import time

texts = ["Teste de embedding rápido"]
start = time.time()
embs = embed("nomic-embed-text", texts)
end = time.time()
print("Embedding:", embs, "— tempo:", end - start)
