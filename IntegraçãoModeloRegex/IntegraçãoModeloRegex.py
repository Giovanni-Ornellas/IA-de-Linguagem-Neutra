import re
from transformers import (
    MarianTokenizer, MarianMTModel,
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline
)

# ========= 1. Função de neutralização com regex ========= #
tradutor_neutro = {
    r"\bele(s)?\b": "elu",
    r"\bela(s)?\b": "elu",
    r"\bdele(s)?\b": "delu",
    r"\bdela(s)?\b": "delu",
    r"\bseu(s)?\b": "seu",
    r"\bsua(s)?\b": "seu",
    r"\bo(s)?\b": "le",
    r"\ba(s)?\b": "le",
    r"\bum(s)?\b": "une",
    r"\buma(s)?\b": "une",
    r"\baluno(s)?\b": "estudante",
    r"\baluna(s)?\b": "estudante",
    r"\bprofessor(es)?\b": "docente",
    r"\bprofessora(s)?\b": "docente",
    r"\bmenino(s)?\b": "criança",
    r"\bmenina(s)?\b": "criança",
    r"\bgaroto(s)?\b": "jovem",
    r"\bgarota(s)?\b": "jovem",
    r"\btodos\b": "todes",
    r"\btodas\b": "todes",
    r"\bbem-vindo(s)?\b": "bem-vinde",
    r"\bbem-vinda(s)?\b": "bem-vinde",
    r"\bquerido(s)?\b": "queride",
    r"\bquerida(s)?\b": "queride",
    r"\bobrigado\b": "obrigade",
    r"\bobrigada\b": "obrigade",
    r"\bsenhor(es)?\b": "pessoa",
    r"\bsenhora(s)?\b": "pessoa",
    r"\bmoço(s)?\b": "pessoa",
    r"\bmoça(s)?\b": "pessoa",
}

def neutralizar_texto(texto: str, regras=tradutor_neutro) -> str:
    for padrao, neutro in regras.items():
        texto = re.sub(padrao, neutro, texto, flags=re.IGNORECASE)
    return texto

# ========= 2. Carrega tradutor pt → en ========= #
modelo_traducao = "Helsinki-NLP/opus-mt-ROMANCE-en"
pt_en_tokenizer = MarianTokenizer.from_pretrained(modelo_traducao)
pt_en_model = MarianMTModel.from_pretrained(modelo_traducao)

def traduzir_pt_en(texto_pt: str) -> str:
    tokens = pt_en_tokenizer([texto_pt], return_tensors="pt", padding=True, truncation=True)
    translated = pt_en_model.generate(**tokens)
    texto_en = pt_en_tokenizer.decode(translated[0], skip_special_tokens=True)
    return texto_en

# ========= 3. Classificador NSFW ========= #
modelo_nsfw = "eliasalbouzidi/distilbert-nsfw-text-classifier"
nsfw_pipeline = pipeline("text-classification", model=modelo_nsfw, tokenizer=modelo_nsfw)

def classificar_nsfw(texto_en: str) -> str:
    resultado = nsfw_pipeline(texto_en)[0]
    return resultado["label"]  # Retorna: 'SFW' ou 'NSFW'

# ========= 4. Função principal integrada ========= #
def processar_texto(texto_original: str):
    print("🔁 Traduzindo texto para inglês...")
    texto_en = traduzir_pt_en(texto_original)

    print("🧠 Classificando texto...")
    categoria = classificar_nsfw(texto_en)

    print(f"📛 Classificação: {categoria}")

    print("⚙️ Aplicando neutralização...")
    texto_neutro = neutralizar_texto(texto_original)

    return {
        "original": texto_original,
        "traduzido_en": texto_en,
        "classificacao": categoria,
        "neutralizado": texto_neutro
    }

# ========= 5. Teste ========= #
if __name__ == "__main__":
    arquivo_txt = open("Teste.txt", "r")
    texto = arquivo_txt.read()
    arquivo_txt.close()
    texto_pt = texto
    resultado = processar_texto(texto_pt)

    print("\n--- RESULTADO FINAL ---")
    for chave, valor in resultado.items():
        print(f"{chave.capitalize()}:\n{valor}\n")
