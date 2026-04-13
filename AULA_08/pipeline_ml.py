# =============================================================================
# BLOCO 2 — PIPELINE COMPLETO DE MACHINE LEARNING
# =============================================================================
# Objetivo: Treinar, avaliar e comparar múltiplos modelos de classificação
# multiclasse para predição de risco clínico de pacientes.
#
# Etapas do pipeline:
#   1. Leitura e inspeção do dataset
#   2. Separação de features (X) e target (y)
#   3. Divisão treino/teste (80/20)
#   4. Normalização com StandardScaler
#   5. Treinamento: Regressão Logística, Random Forest, KNN
#   6. Avaliação: Acurácia, Precision, Recall, F1-score
#   7. Validação Cruzada (k-fold, k=5)
#   8. Visualizações: gráfico de acurácia, matrizes de confusão, curva ROC
#   9. Predição de novo paciente com exibição clara do resultado
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")  # Suprime avisos de convergência para limpeza didática

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.preprocessing   import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO GLOBAL DE ESTILO
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#f9f9f9",
    "axes.facecolor"  : "#ffffff",
    "axes.grid"       : True,
    "grid.alpha"      : 0.4,
    "font.family"     : "DejaVu Sans",
})

CLASSES        = ["Baixo (0)", "Médio (1)", "Alto (2)"]
CORES_MODELOS  = {"Regressão Logística": "#4C72B0",
                  "Random Forest"      : "#55A868",
                  "KNN"                : "#C44E52"}

# =============================================================================
# ETAPA 1 — LEITURA E INSPEÇÃO DO DATASET
# =============================================================================
print("=" * 65)
print("  ETAPA 1 — LEITURA E INSPEÇÃO DO DATASET")
print("=" * 65)

# Lê o CSV gerado pelo script anterior
df = pd.read_csv("pacientes.csv")

print(f"\n📋 Shape do dataset: {df.shape[0]} linhas × {df.shape[1]} colunas")
print("\n─── Tipos de dados ───")
print(df.dtypes)

print("\n─── Valores ausentes por coluna ───")
print(df.isnull().sum())

print("\n─── Estatísticas descritivas ───")
print(df.describe().round(2))

print("\n─── Distribuição da variável alvo ───")
print(df["Risco"].value_counts().sort_index()
      .rename(index={0: "Baixo (0)", 1: "Médio (1)", 2: "Alto (2)"}))

# =============================================================================
# ETAPA 2 — SEPARAÇÃO DE FEATURES (X) E TARGET (y)
# =============================================================================
print("\n" + "=" * 65)
print("  ETAPA 2 — SEPARAÇÃO DE FEATURES E TARGET")
print("=" * 65)

# Features: todas as colunas numéricas, exceto 'Nome' (texto) e 'Risco' (target)
FEATURES = ["Idade", "Glicose", "Pressao_Arterial", "IMC", "Colesterol"]
TARGET   = "Risco"

X = df[FEATURES].values   # matriz de entrada (NumPy array)
y = df[TARGET].values     # vetor de saída

print(f"\n✅ Features utilizadas: {FEATURES}")
print(f"   Shape de X: {X.shape}")
print(f"   Shape de y: {y.shape}")

# =============================================================================
# ETAPA 3 — DIVISÃO TREINO / TESTE
# =============================================================================
print("\n" + "=" * 65)
print("  ETAPA 3 — DIVISÃO TREINO / TESTE (80% / 20%)")
print("=" * 65)

# stratify=y garante proporção igual das classes em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size   = 0.20,
    random_state= 42,
    stratify    = y
)

print(f"\n✅ Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")

# =============================================================================
# ETAPA 4 — NORMALIZAÇÃO COM STANDARDSCALER
# =============================================================================
print("\n" + "=" * 65)
print("  ETAPA 4 — NORMALIZAÇÃO (StandardScaler)")
print("=" * 65)

# O scaler é AJUSTADO apenas com os dados de treino (evita data leakage)
# e depois aplicado (transform) tanto no treino quanto no teste.
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)   # fit + transform no treino
X_test  = scaler.transform(X_test)        # apenas transform no teste

print("\n✅ Dados normalizados (média≈0, desvio≈1 para cada feature).")
print(f"   Médias aprendidas: {scaler.mean_.round(2)}")
print(f"   Desvios aprendidos: {scaler.scale_.round(2)}")

# =============================================================================
# ETAPA 5 — DEFINIÇÃO E TREINAMENTO DOS MODELOS
# =============================================================================
print("\n" + "=" * 65)
print("  ETAPA 5 — TREINAMENTO DOS MODELOS")
print("=" * 65)

# Dicionário com os três modelos e seus hiperparâmetros básicos
modelos = {
    "Regressão Logística": LogisticRegression(
        solver       = "lbfgs",   # otimizador eficiente para classificação multiclasse
        max_iter     = 1000,      # aumentado para garantir convergência
        random_state = 42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators = 100,   # 100 árvores de decisão na floresta
        max_depth    = None,  # árvores crescem até pureza total
        random_state = 42,
        n_jobs       = -1     # usa todos os núcleos disponíveis
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors = 7,        # k=7 — equilibra underfitting e overfitting
        metric      = "euclidean",
        weights     = "distance" # vizinhos mais próximos têm mais peso
    )
}

# Treina todos os modelos e armazena os objetos treinados
modelos_treinados = {}
for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    modelos_treinados[nome] = modelo
    print(f"  ✅ {nome} — treinado com sucesso.")

# =============================================================================
# ETAPA 6 — AVALIAÇÃO COM MÉTRICAS
# =============================================================================
print("\n" + "=" * 65)
print("  ETAPA 6 — AVALIAÇÃO COM MÉTRICAS (conjunto de teste)")
print("=" * 65)

resultados = {}  # armazena métricas de cada modelo

print(f"\n{'Modelo':<25} {'Acurácia':>10} {'Precision':>10} {'Recall':>8} {'F1-Score':>10}")
print("-" * 67)

for nome, modelo in modelos_treinados.items():
    y_pred = modelo.predict(X_test)

    acc  = accuracy_score (y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score   (y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score       (y_test, y_pred, average="weighted", zero_division=0)

    resultados[nome] = {"Acurácia": acc, "Precision": prec,
                        "Recall": rec, "F1-Score": f1,
                        "y_pred": y_pred}

    print(f"  {nome:<23} {acc:>10.4f} {prec:>10.4f} {rec:>8.4f} {f1:>10.4f}")

# =============================================================================
# ETAPA 7 — VALIDAÇÃO CRUZADA (k-Fold, k=5)
# =============================================================================
print("\n" + "=" * 65)
print("  ETAPA 7 — VALIDAÇÃO CRUZADA (StratifiedKFold, k=5)")
print("=" * 65)
print("\n  A validação cruzada avalia a robustez do modelo em 5 partições")
print("  diferentes dos dados, evitando que a divisão treino/teste seja")
print("  favorável por acaso.\n")

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = {}
print(f"{'Modelo':<25} {'Média CV':>10} {'Desvio Padrão':>15} {'Mín':>8} {'Máx':>8}")
print("-" * 70)

for nome, modelo in modelos.items():
    # Atenção: cross_val_score usa os dados NÃO normalizados de propósito aqui?
    # Não — para manter consistência, passamos X já normalizado (X_train+X_test)
    # Usamos X e y originais E deixamos o sklearn re-normalizar internamente
    # via Pipeline em projetos avançados. Aqui, usamos X normalizado completo:
    X_norm_full = scaler.transform(df[FEATURES].values)  # normaliza o dataset completo
    scores = cross_val_score(modelo, X_norm_full, y,
                             cv=kfold, scoring="accuracy", n_jobs=-1)
    cv_scores[nome] = scores
    print(f"  {nome:<23} {scores.mean():>10.4f} {scores.std():>15.4f} "
          f"{scores.min():>8.4f} {scores.max():>8.4f}")

# Determina o melhor modelo com base na acurácia no conjunto de teste
melhor_modelo_nome = max(resultados, key=lambda k: resultados[k]["Acurácia"])
melhor_modelo      = modelos_treinados[melhor_modelo_nome]
print(f"\n🏆 Melhor modelo: {melhor_modelo_nome} "
      f"(Acurácia = {resultados[melhor_modelo_nome]['Acurácia']:.4f})")

# =============================================================================
# ETAPA 8 — VISUALIZAÇÕES
# =============================================================================
print("\n" + "=" * 65)
print("  ETAPA 8 — GERANDO VISUALIZAÇÕES")
print("=" * 65)

# ─── 8A. GRÁFICO COMPARATIVO DE MÉTRICAS ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
fig.suptitle("Comparação de Desempenho dos Modelos", fontsize=14, fontweight="bold", y=1.01)

metricas_nomes  = ["Acurácia", "Precision", "Recall", "F1-Score"]
n_modelos       = len(resultados)
n_metricas      = len(metricas_nomes)
x               = np.arange(n_metricas)
largura         = 0.22
offsets         = [-largura, 0, largura]

for i, (nome, vals) in enumerate(resultados.items()):
    valores = [vals[m] for m in metricas_nomes]
    barras  = ax.bar(x + offsets[i], valores, width=largura,
                     label=nome, color=CORES_MODELOS[nome],
                     edgecolor="white", linewidth=0.8)
    # Adiciona valor em cima de cada barra
    for barra in barras:
        h = barra.get_height()
        ax.text(barra.get_x() + barra.get_width() / 2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(metricas_nomes, fontsize=11)
ax.set_ylabel("Score", fontsize=11)
ax.set_ylim(0, 1.08)
ax.legend(loc="lower right", fontsize=9)
ax.set_title("Métricas por Modelo (conjunto de teste)", fontsize=11)
plt.tight_layout()
plt.savefig("grafico_comparativo_modelos.png", dpi=150, bbox_inches="tight")
plt.show()
print("  📊 Gráfico salvo: grafico_comparativo_modelos.png")

# ─── 8B. MATRIZES DE CONFUSÃO (todos os modelos) ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Matrizes de Confusão — Todos os Modelos", fontsize=13, fontweight="bold")

for ax, (nome, vals) in zip(axes, resultados.items()):
    cm   = confusion_matrix(y_test, vals["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Baixo", "Médio", "Alto"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{nome}\nAcurácia: {vals['Acurácia']:.4f}", fontsize=10)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")

plt.tight_layout()
plt.savefig("matrizes_confusao.png", dpi=150, bbox_inches="tight")
plt.show()
print("  📊 Matrizes salvas: matrizes_confusao.png")

# ─── 8C. CURVA ROC (multiclasse — One vs Rest) ───────────────────────────────
# Para curva ROC multiclasse, binarizamos y (One-vs-Rest)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])  # shape: (n, 3)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle("Curvas ROC (One-vs-Rest) — Todos os Modelos",
             fontsize=13, fontweight="bold")

cores_classes = ["#2196F3", "#FF9800", "#E91E63"]

for ax, (nome, modelo) in zip(axes, modelos_treinados.items()):
    # predict_proba retorna probabilidades para cada classe
    if hasattr(modelo, "predict_proba"):
        y_score = modelo.predict_proba(X_test)
    else:
        # KNN sempre tem predict_proba, mas garantimos o fallback
        y_score = modelo.predict_proba(X_test)

    for i, (classe, cor) in enumerate(zip(CLASSES, cores_classes)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=cor, lw=2,
                label=f"{classe} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatório (AUC=0.5)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Taxa de Falso Positivo (FPR)", fontsize=9)
    ax.set_ylabel("Taxa de Verdadeiro Positivo (TPR)", fontsize=9)
    ax.set_title(f"ROC — {nome}", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)

plt.tight_layout()
plt.savefig("curvas_roc.png", dpi=150, bbox_inches="tight")
plt.show()
print("  📊 Curvas ROC salvas: curvas_roc.png")

# ─── 8D. VALIDAÇÃO CRUZADA — boxplot por modelo ──────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
dados_cv  = list(cv_scores.values())
nomes_cv  = list(cv_scores.keys())
bp = ax.boxplot(dados_cv, patch_artist=True, notch=False,
                medianprops=dict(color="black", linewidth=2))
for patch, (nome, _) in zip(bp["boxes"], cv_scores.items()):
    patch.set_facecolor(CORES_MODELOS[nome])
    patch.set_alpha(0.75)

ax.set_xticklabels(nomes_cv, fontsize=10)
ax.set_ylabel("Acurácia (k-fold)", fontsize=11)
ax.set_title("Distribuição da Acurácia na Validação Cruzada (k=5)", fontsize=11)
plt.tight_layout()
plt.savefig("validacao_cruzada_boxplot.png", dpi=150, bbox_inches="tight")
plt.show()
print("  📊 Boxplot de validação cruzada salvo: validacao_cruzada_boxplot.png")

# =============================================================================
# ETAPA 9 — PREDIÇÃO DE NOVO PACIENTE
# =============================================================================
print("\n" + "=" * 65)
print("  ETAPA 9 — PREDIÇÃO DE NOVO PACIENTE")
print("=" * 65)

# ─── Dados do paciente fictício ───────────────────────────────────────────────
novo_paciente = {
    "Nome"             : "Marcos",
    "Idade"            : 62,
    "Glicose"          : 148.5,   # acima de 126 → faixa diabética
    "Pressao_Arterial" : 155.0,   # ≥ 140 → hipertensão
    "IMC"              : 31.2,    # ≥ 30  → obeso
    "Colesterol"       : 258.0    # ≥ 240 → colesterol alto
}

print(f"\n  👤 Novo paciente: {novo_paciente['Nome']}")
print(f"     Idade:             {novo_paciente['Idade']} anos")
print(f"     Glicose:           {novo_paciente['Glicose']} mg/dL")
print(f"     Pressão Arterial:  {novo_paciente['Pressao_Arterial']} mmHg")
print(f"     IMC:               {novo_paciente['IMC']} kg/m²")
print(f"     Colesterol:        {novo_paciente['Colesterol']} mg/dL")

# Monta o array de entrada na mesma ordem das features de treino
entrada = np.array([[
    novo_paciente["Idade"],
    novo_paciente["Glicose"],
    novo_paciente["Pressao_Arterial"],
    novo_paciente["IMC"],
    novo_paciente["Colesterol"]
]])

# Normaliza com o mesmo scaler usado no treinamento (CRÍTICO — não criar novo scaler!)
entrada_norm = scaler.transform(entrada)

# ─── Predição com todos os modelos ───────────────────────────────────────────
print("\n  ─── Resultado por Modelo ───")
mapa_risco   = {0: "🟢 BAIXO RISCO", 1: "🟡 MÉDIO RISCO", 2: "🔴 ALTO RISCO"}
mapa_detalhe = {0: "Perfil clínico dentro dos limites normais.",
                1: "Atenção: alguns indicadores merecem monitoramento.",
                2: "⚠️  Intervenção recomendada — múltiplos fatores de risco elevados."}

for nome, modelo in modelos_treinados.items():
    classe_pred  = modelo.predict(entrada_norm)[0]
    probabilidades = modelo.predict_proba(entrada_norm)[0]

    print(f"\n  📌 {nome}")
    print(f"     Classificação : {mapa_risco[classe_pred]}")
    print(f"     Probabilidades:")
    for idx_c, prob in enumerate(probabilidades):
        barra = "█" * int(prob * 30)
        print(f"       Classe {idx_c} ({['Baixo','Médio','Alto'][idx_c]:5s}): "
              f"{prob:5.1%}  {barra}")
    print(f"     Interpretação : {mapa_detalhe[classe_pred]}")

# ─── Resultado final com o melhor modelo ─────────────────────────────────────
classe_final   = melhor_modelo.predict(entrada_norm)[0]
probs_finais   = melhor_modelo.predict_proba(entrada_norm)[0]
confianca      = probs_finais[classe_final]

print("\n" + "=" * 65)
print(f"  🏆 RESULTADO FINAL — {melhor_modelo_nome} (Melhor Modelo)")
print("=" * 65)
print(f"\n  Paciente     : {novo_paciente['Nome']}")
print(f"  Diagnóstico  : {mapa_risco[classe_final]}")
print(f"  Confiança    : {confianca:.1%}")
print(f"  Observação   : {mapa_detalhe[classe_final]}")
print("\n" + "=" * 65)
print("  ✅ Pipeline concluído com sucesso!")
print("=" * 65)
