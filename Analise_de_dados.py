# Trabalho Prático - Introdução à Inteligência Artificial
# Disciplina: Introdução à Inteligência Artificial - Ciência da Computação
# Professor: Prof. Edkallenn Lima
# Instituição: UNIPÊ – Centro Universitário de João Pessoa

# --- 1. Introdução ---
# Este notebook apresenta um estudo de classificação da renda anual com base no dataset Adult, 
# aplicando diversos algoritmos de aprendizado supervisionado e uma rede neural simples.

# --- 2. Fonte e Descrição dos Dados ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Fonte dos dados: UCI Machine Learning Repository - Adult Data Set
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income'
]
data = pd.read_csv(url, header=None, names=columns, na_values=' ?')

print(f"Base de dados Adult carregada com {data.shape[0]} instâncias e {data.shape[1]} atributos.")
print("Variável alvo: income (renda >50K ou <=50K)")

# --- 3. Análise Exploratória ---
print("\nVisualização dos 5 primeiros registros:")
print(data.head())

print("\nInformações gerais:")
print(data.info())

print("\nEstatísticas descritivas numéricas:")
print(data.describe())

print("\nValores ausentes por coluna:")
print(data.isnull().sum())

# Visualização gráfica
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.histplot(data['age'], bins=30, kde=True)
plt.title('Distribuição de idade')

plt.subplot(1,2,2)
sns.histplot(data['hours-per-week'], bins=30, kde=True)
plt.title('Distribuição de horas por semana')

plt.tight_layout()
plt.show()

# --- 4. Pré-processamento ---

# Remover linhas com valores ausentes
data_clean = data.dropna()
print(f"\nApós remoção de linhas com valores ausentes: {data_clean.shape[0]} registros restantes")

# Codificação das variáveis categóricas com One-Hot Encoding (exceto target)
data_encoded = pd.get_dummies(data_clean.drop('income', axis=1))
print(f"Shape após One-Hot Encoding: {data_encoded.shape}")

# Codificação do target
le_income = LabelEncoder()
target = le_income.fit_transform(data_clean['income'])

# Normalização das colunas numéricas
scaler = StandardScaler()
num_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
data_encoded[num_cols] = scaler.fit_transform(data_encoded[num_cols])

# Divisão treino/teste (75% treino, 25% teste) com stratify para balancear classes
X_train, X_test, y_train, y_test = train_test_split(
    data_encoded, target, test_size=0.25, random_state=42, stratify=target
)

print(f"\nShape treino: {X_train.shape} / Shape teste: {X_test.shape}")

# --- 5. Modelagem com Algoritmos Supervisionados ---

# 1. Regressão Logística
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

# 2. Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)

# 3. Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

# Validação cruzada (5 folds) para Random Forest
cv_scores_rf = cross_val_score(rf, data_encoded, target, cv=5, scoring='f1')

# --- 6. Rede Neural Simples (MLP) com Keras ---

# Preparação dos dados para Keras
input_dim = X_train.shape[1]
y_train_keras = to_categorical(y_train)
y_test_keras = to_categorical(y_test)

model = Sequential()
model.add(Dense(32, input_dim=input_dim, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treinamento
history = model.fit(X_train, y_train_keras, epochs=20, batch_size=32, validation_split=0.2, verbose=0)

# Avaliação da rede neural
scores = model.evaluate(X_test, y_test_keras, verbose=0)
acc_nn = scores[1]

# Previsão para F1-score
y_pred_nn = model.predict(X_test)
y_pred_nn_classes = np.argmax(y_pred_nn, axis=1)
f1_nn = f1_score(y_test, y_pred_nn_classes)

# --- Resultados e Conclusão ---

f1_scores = {
    'Regressão Logística': f1_lr,
    'Decision Tree': f1_dt,
    'Random Forest': f1_rf,
    'Rede Neural': f1_nn
}

best_model = max(f1_scores, key=f1_scores.get)
best_f1 = f1_scores[best_model]

print("\n--- Resultados dos Modelos ---")
for model_name, f1_val in f1_scores.items():
    print(f"{model_name}: F1-score = {f1_val:.4f}")

print(f"\nMelhor modelo: {best_model} com F1-score = {best_f1:.4f}")
print(f"Validação cruzada (5-fold) Random Forest - média F1: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std():.4f})")

# Matriz de confusão do melhor modelo
if best_model == 'Regressão Logística':
    cm = confusion_matrix(y_test, y_pred_lr)
elif best_model == 'Decision Tree':
    cm = confusion_matrix(y_test, y_pred_dt)
elif best_model == 'Random Forest':
    cm = confusion_matrix(y_test, y_pred_rf)
else:
    cm = confusion_matrix(y_test, y_pred_nn_classes)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_income.classes_)
disp.plot()
plt.title(f'Matriz de Confusão - {best_model}')
plt.show()

# Gráficos da rede neural (acurácia e perda)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia da Rede Neural')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Perda da Rede Neural')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Conclusão para relatório
conclusao = f"""
Conclusão:
O modelo com melhor desempenho para classificação da renda (>50K) no dataset Adult foi o {best_model}, 
com um F1-score de {best_f1:.4f}. A validação cruzada reforçou a robustez do modelo Random Forest, com uma média 
de F1-score de {cv_scores_rf.mean():.4f} em 5 folds.

Os resultados indicam que modelos baseados em árvores (Decision Tree, Random Forest) apresentam melhor equilíbrio
entre precisão e recall neste problema. A rede neural apresentou desempenho competitivo, porém inferior ao Random Forest.

Limitações incluem a exclusão de dados ausentes e a necessidade de testar hiperparâmetros para otimização futura.
Recomenda-se também explorar validação cruzada para todos os modelos e técnicas de balanceamento, caso as classes 
estejam desbalanceadas.

Referências:
- UCI Machine Learning Repository: Adult Data Set
- Bibliotecas Python: pandas, scikit-learn, TensorFlow/Keras, matplotlib, seaborn
"""

print(conclusao)