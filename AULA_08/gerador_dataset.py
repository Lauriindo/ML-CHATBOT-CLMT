# =============================================================================
# BLOCO 1 — GERADOR DE DATASET SINTÉTICO DE PACIENTES
# =============================================================================
# Objetivo: Gerar 2.000 registros fictícios de pacientes com variáveis
# biomédicas realistas e uma regra clínica coerente para definir o risco.
#
# Variáveis geradas:
#   - Nome       : string fictícia (sem sobrenome)
#   - Idade      : inteiro entre 18 e 99 anos
#   - Glicose    : mg/dL (70–300)
#   - Pressao_Arterial : mmHg sistólica (80–200)
#   - IMC        : kg/m² (17–45)
#   - Colesterol : mg/dL (130–320)
#
# Variável alvo:
#   - Risco      : 0 = Baixo, 1 = Médio, 2 = Alto
# =============================================================================

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 1. SEMENTE ALEATÓRIA — garante reprodutibilidade
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(42)

N = 2000  # número de registros

# ─────────────────────────────────────────────────────────────────────────────
# 2. POOL DE NOMES FICTÍCIOS (simples, sem sobrenome)
# ─────────────────────────────────────────────────────────────────────────────
nomes_pool = [
    "Ana", "Bruno", "Carla", "Diego", "Elena", "Fábio", "Gabi", "Hugo",
    "Iris", "João", "Karen", "Lucas", "Marta", "Nina", "Otávio", "Paula",
    "Rafael", "Sara", "Tiago", "Uíra", "Vera", "Wagner", "Xênia", "Yago",
    "Zara", "Alícia", "Bento", "Clara", "Davi", "Érica", "Felipe", "Gláucia",
    "Henrique", "Ingrid", "Jorge", "Lara", "Marcos", "Natália", "Olga",
    "Pedro", "Raquel", "Sílvio", "Tatiana", "Ulisses", "Viviane", "Wanda",
    "Xavier", "Yasmin", "Zélio", "Adriana", "Bernardo", "Cíntia", "Danilo",
    "Estela", "Fernando", "Giovana", "Heitor", "Isabel", "Júlio", "Keila",
    "Leonardo", "Mônica", "Nando", "Olívia", "Priscila", "Quirino", "Renata",
    "Samuel", "Taís", "Ursula", "Valentina", "Wendell", "Ximena", "Yuri", "Zilda"
]

# Sorteia 2000 nomes (com reposição — é normal repetir num dataset grande)
nomes = np.random.choice(nomes_pool, size=N, replace=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3. GERAÇÃO DAS VARIÁVEIS BIOMÉDICAS COM DISTRIBUIÇÕES REALISTAS
# ─────────────────────────────────────────────────────────────────────────────

# Idade: distribuição uniforme entre 18 e 99 anos
idade = np.random.randint(18, 100, size=N)

# Glicose (mg/dL):
#   Normal em jejum: 70–99
#   Pré-diabético:   100–125
#   Diabético:       126–300
# Usamos uma mistura de distribuições normais para simular a realidade
glicose = np.clip(
    np.random.normal(loc=110, scale=35, size=N),
    70, 300
).round(1)

# Pressão Arterial Sistólica (mmHg):
#   Ótima:       < 120
#   Normal:      120–129
#   Elevada:     130–139
#   Hipertensão: ≥ 140
pressao_arterial = np.clip(
    np.random.normal(loc=130, scale=25, size=N),
    80, 200
).round(1)

# IMC (kg/m²):
#   Abaixo do peso: < 18.5
#   Normal:         18.5–24.9
#   Sobrepeso:      25–29.9
#   Obeso:          ≥ 30
imc = np.clip(
    np.random.normal(loc=27, scale=6, size=N),
    17, 45
).round(1)

# Colesterol Total (mg/dL):
#   Desejável:  < 200
#   Limítrofe:  200–239
#   Alto:       ≥ 240
colesterol = np.clip(
    np.random.normal(loc=210, scale=45, size=N),
    130, 320
).round(1)

# ─────────────────────────────────────────────────────────────────────────────
# 4. REGRA CLÍNICA PARA DEFINIR O RISCO
# ─────────────────────────────────────────────────────────────────────────────
# Cada variável contribui com pontos de risco conforme faixas clínicas reais.
# Ao final, somamos os pontos e classificamos em 3 categorias.
#
# Escala de pontuação por variável (0 = bom, 1 = atenção, 2 = crítico):
#
#   GLICOSE:
#       < 100           → 0 pontos (normal)
#       100–125         → 1 ponto  (pré-diabético)
#       ≥ 126           → 2 pontos (diabético)
#
#   PRESSÃO ARTERIAL:
#       < 130           → 0 pontos (normal/ótima)
#       130–139         → 1 ponto  (elevada)
#       ≥ 140           → 2 pontos (hipertensão)
#
#   IMC:
#       < 25            → 0 pontos (normal/abaixo)
#       25–29.9         → 1 ponto  (sobrepeso)
#       ≥ 30            → 2 pontos (obeso)
#
#   COLESTEROL:
#       < 200           → 0 pontos (desejável)
#       200–239         → 1 ponto  (limítrofe)
#       ≥ 240           → 2 pontos (alto)
#
#   IDADE:
#       < 40            → 0 pontos (jovem)
#       40–59           → 1 ponto  (meia-idade)
#       ≥ 60            → 2 pontos (idoso — maior vulnerabilidade)
#
# Pontuação total possível: 0 a 10
#   0–3  → Risco 0 (Baixo)
#   4–6  → Risco 1 (Médio)
#   7–10 → Risco 2 (Alto)
# ─────────────────────────────────────────────────────────────────────────────

def calcular_pontos(glicose, pressao, imc, colesterol, idade):
    """
    Recebe arrays NumPy e retorna um array de pontuação total de risco.
    Cada variável contribui com 0, 1 ou 2 pontos conforme faixas clínicas.
    """
    pontos = np.zeros(len(glicose), dtype=int)

    # Pontos de Glicose
    pontos += np.where(glicose >= 126, 2, np.where(glicose >= 100, 1, 0))

    # Pontos de Pressão Arterial
    pontos += np.where(pressao >= 140, 2, np.where(pressao >= 130, 1, 0))

    # Pontos de IMC
    pontos += np.where(imc >= 30, 2, np.where(imc >= 25, 1, 0))

    # Pontos de Colesterol
    pontos += np.where(colesterol >= 240, 2, np.where(colesterol >= 200, 1, 0))

    # Pontos de Idade
    pontos += np.where(idade >= 60, 2, np.where(idade >= 40, 1, 0))

    return pontos


# Calcula pontuação total para todos os pacientes
pontuacao = calcular_pontos(glicose, pressao_arterial, imc, colesterol, idade)

# Converte pontuação em classe de risco (0, 1 ou 2)
risco = np.where(pontuacao >= 7, 2, np.where(pontuacao >= 4, 1, 0))

# ─────────────────────────────────────────────────────────────────────────────
# 5. ADICIONANDO RUÍDO CONTROLADO (REALISMO)
# ─────────────────────────────────────────────────────────────────────────────
# Em dados reais, existem exceções e variações não capturadas pelas regras.
# Adicionamos um pequeno ruído: ~5% dos registros terão o risco deslocado em
# ±1 nível, simulando casos atípicos ou outliers clínicos.

n_ruido = int(N * 0.05)  # 5% = 100 registros
idx_ruido = np.random.choice(N, size=n_ruido, replace=False)
deslocamento = np.random.choice([-1, 1], size=n_ruido)
risco[idx_ruido] = np.clip(risco[idx_ruido] + deslocamento, 0, 2)

# ─────────────────────────────────────────────────────────────────────────────
# 6. MONTAGEM DO DATAFRAME
# ─────────────────────────────────────────────────────────────────────────────
df = pd.DataFrame({
    "Nome"             : nomes,
    "Idade"            : idade,
    "Glicose"          : glicose,
    "Pressao_Arterial" : pressao_arterial,
    "IMC"              : imc,
    "Colesterol"       : colesterol,
    "Risco"            : risco       # 0=Baixo | 1=Médio | 2=Alto
})

# ─────────────────────────────────────────────────────────────────────────────
# 7. SALVANDO O DATASET EM CSV
# ─────────────────────────────────────────────────────────────────────────────
df.to_csv("pacientes.csv", index=False, encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# 8. RELATÓRIO DE GERAÇÃO
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("       GERADOR DE DATASET — RELATÓRIO FINAL")
print("=" * 60)
print(f"\n✅ Dataset gerado com {N} registros.")
print(f"📁 Arquivo salvo como: pacientes.csv\n")

print("─── Distribuição da Variável Alvo (Risco) ───")
contagem = pd.Series(risco).value_counts().sort_index()
labels = {0: "Baixo (0)", 1: "Médio (1)", 2: "Alto  (2)"}
for k, v in contagem.items():
    pct = v / N * 100
    print(f"  {labels[k]}: {v:>5} registros ({pct:.1f}%)")

print("\n─── Estatísticas Descritivas das Features ───")
print(df[["Idade", "Glicose", "Pressao_Arterial", "IMC", "Colesterol"]].describe().round(2))

print("\n─── Primeiras 5 linhas do Dataset ───")
print(df.head())
print("=" * 60)
