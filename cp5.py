import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Configurações visuais
pd.set_option("display.max_columns", None)
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


# 1. CARREGAMENTO E EXPLORAÇÃO

df = pd.read_csv("global_tech_salary.txt")
print(f"Dataset carregado: {df.shape[0]} linhas e {df.shape[1]} colunas\n")
print("Primeiras linhas:")
print(df.head())


# 2. PRÉ-PROCESSAMENTO


print("\n1. Tratamento de dados duplicados e ausentes")
# Remover duplicados e garantir cópia independente para evitar warnings
df_clean = df.drop_duplicates().copy()
print(f"Registros após remoção de duplicados: {df_clean.shape[0]}")

# Verificação de valores ausentes
if df_clean.isnull().sum().sum() == 0:
    print("Não há valores ausentes no dataset")


# 2.1 Criação da variável alvo

# 1 = salário acima da mediana | 0 = salário abaixo da mediana
salario_mediano = df_clean['salary_in_usd'].median()
df_clean.loc[:, 'salario_alto'] = (df_clean['salary_in_usd'] > salario_mediano).astype(int)
print(f"\nSalário mediano em USD: ${salario_mediano:,.0f}")
print("Distribuição da variável alvo:")
print(df_clean['salario_alto'].value_counts())


# 2.2 Codificação das variáveis categóricas

# Usamos one-hot encoding para variáveis categóricas não ordinais
df_encoded = pd.get_dummies(
    df_clean,
    columns=['experience_level','employment_type','company_size'],
    drop_first=True
)
print(f"\nDataset após codificação: {df_encoded.shape[1]} colunas")

# 2.3 Seleção das features e normalização

features = [col for col in df_encoded.columns if col not in ['salary','salary_currency',
                                                            'salary_in_usd','job_title',
                                                            'employee_residence','company_location',
                                                            'salario_alto']]

X = df_encoded[features]
y = df_encoded['salario_alto']

# Normalização (necessário para KNN)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)


# 2.4 Divisão treino/teste

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)
print(f"\nTreino: {X_train.shape[0]} registros | Teste: {X_test.shape[0]} registros")
print("Distribuição da classe no treino:", y_train.value_counts(normalize=True).to_dict())


# 3. FUNÇÃO PARA AVALIAR MODELOS

def avaliar_modelo(modelo, X_test, y_test, nome_modelo):
    """
    Avalia o modelo, exibe métricas e matriz de confusão
    """
    pred = modelo.predict(X_test)
    
    ac = accuracy_score(y_test, pred)
    pr = precision_score(y_test, pred, average='weighted')
    rc = recall_score(y_test, pred, average='weighted')
    f1 = f1_score(y_test, pred, average='weighted')
    
    print(f"\n{nome_modelo}")
    print(f"Acurácia: {ac:.4f} | Precisão: {pr:.4f} | Recall: {rc:.4f} | F1-score: {f1:.4f}")
    
    # Matriz de confusão
    matriz = confusion_matrix(y_test, pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Salário Baixo','Salário Alto'],
                yticklabels=['Salário Baixo','Salário Alto'])
    plt.title(f'Matriz de Confusão - {nome_modelo}')
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.show()
    
    # Relatório detalhado por classe
    print("\nRelatório de classificação por classe:")
    print(classification_report(y_test, pred, target_names=['Salário Baixo','Salário Alto']))
    
    return ac, pr, rc, f1, matriz

# 4. MODELAGEM

print("\nTreinando e avaliando os modelos...")


# 4.1 KNN

# Teste de diferentes k para garantir melhor escolha
melhor_k = 3
melhor_acc = 0
for k in [3,5,7,9,11,15]:
    knn_test = KNeighborsClassifier(n_neighbors=k)
    knn_test.fit(X_train, y_train)
    acc = accuracy_score(y_test, knn_test.predict(X_test))
    if acc > melhor_acc:
        melhor_acc = acc
        melhor_k = k

knn = KNeighborsClassifier(n_neighbors=melhor_k)
knn.fit(X_train, y_train)
result_knn = avaliar_modelo(knn, X_test, y_test, f"KNN (k={melhor_k})")


# 4.2 Regressão Logística

logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)
result_logreg = avaliar_modelo(logreg, X_test, y_test, "Regressão Logística")

# 4.3 Árvore de Decisão

arvore = DecisionTreeClassifier(max_depth=4, min_samples_split=20, random_state=42)
arvore.fit(X_train, y_train)
result_arvore = avaliar_modelo(arvore, X_test, y_test, "Árvore de Decisão")

# Visualização da árvore
plt.figure(figsize=(16,10))
plot_tree(arvore, feature_names=features, class_names=['Salário Baixo','Salário Alto'],
          filled=True, rounded=True, fontsize=10)
plt.show()


# 4.4 Random Forest

rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf.fit(X_train, y_train)
result_rf = avaliar_modelo(rf, X_test, y_test, "Random Forest")


# 5. COMPARAÇÃO DOS MODELOS

comparativo = pd.DataFrame({
    'Modelo': ['KNN','Regressão Logística','Árvore de Decisão','Random Forest'],
    'Acurácia':[result_knn[0], result_logreg[0], result_arvore[0], result_rf[0]],
    'Precisão':[result_knn[1], result_logreg[1], result_arvore[1], result_rf[1]],
    'Recall':[result_knn[2], result_logreg[2], result_arvore[2], result_rf[2]],
    'F1-Score':[result_knn[3], result_logreg[3], result_arvore[3], result_rf[3]]
})

print("\nComparativo de desempenho:")
print(comparativo.round(4))

# Gráfico comparativo
plt.figure(figsize=(12,6))
metricas = ['Acurácia','Precisão','Recall','F1-Score']
x = np.arange(len(metricas))
largura = 0.2

for i, modelo in enumerate(comparativo['Modelo']):
    plt.bar(x + i*largura, comparativo.iloc[i,1:], width=largura, label=modelo)

plt.xticks(x + largura*1.5, metricas)
plt.ylabel('Valor')
plt.title('Desempenho dos Modelos')
plt.ylim(0,1)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()


# 6. IMPORTÂNCIA DAS FEATURES

importancia = pd.DataFrame({
    'Variável': features,
    'Importância': rf.feature_importances_
}).sort_values('Importância', ascending=False)

print("\nImportância das variáveis (Random Forest):")
print(importancia.round(4))

plt.figure(figsize=(10,6))
sns.barplot(data=importancia, x='Importância', y='Variável')
plt.title('Importância das Variáveis no Random Forest')
plt.show()
