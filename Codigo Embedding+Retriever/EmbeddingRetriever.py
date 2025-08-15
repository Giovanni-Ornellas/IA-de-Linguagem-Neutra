from sentence_transformers import SentenceTransformer 
import faiss
import numpy as np
import pandas as pd

frase_entrada = "Todos os funcionários saíram tarde."

dataset = pd.read_csv('dataset.csv') #Carrega o dataset
frases_originais = dataset['original'].tolist()
frases_neutras = dataset['neutro'].tolist()

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2') #Modelo de Embeddings

lista_embeddings = model.encode(frases_originais) # Faz o embedding da frase da lista

embedding = model.encode(frase_entrada) #Faz o embedding da frase do usuário

index = faiss.IndexFlatL2(embedding.shape[0]) #Modelo Retriever
index.add(np.array(lista_embeddings)) #Adciona a lista de embeddings da lista

D, I = index.search(np.array([embedding]), k=3)
#D = distâncias, I = índices

print("Frase de entrada: ", frase_entrada)
print("\nResultados encontrados:")
for dist, i in zip(D[0], I[0]):
    print(f"- Distância L2²: {dist:.4f}")
    print(f"  Original: {frases_originais[i]}")
    print(f"  Neutra:   {frases_neutras[i]}")
    print()

