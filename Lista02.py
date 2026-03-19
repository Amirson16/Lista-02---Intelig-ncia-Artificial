import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay

# --- ETAPA 1: CARREGAMENTO DOS DADOS ---
df = pd.read_csv('dataset_emprestimo_aprovacao.csv')
print("Primeiras linhas:")
print(df.head())
print("\nInformacoes gerais:")
df.info()

# --- ETAPA 2: ANALISE EXPLORATORIA ---
sns.pairplot(df, hue='emprestimo_aprovado')
plt.show()

# --- ETAPA 3: PREPARACAO DOS DADOS ---
X = df.drop('emprestimo_aprovado', axis=1)
y = df['emprestimo_aprovado']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- ETAPA 4: TREINAMENTO DO MODELO ---
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

# --- ETAPA 5: AVALIACAO DO MODELO ---
y_pred = modelo.predict(X_test)

print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.title("Matriz de Confusao")
plt.show()

# --- ETAPA 6: INTERPRETACAO ---
print("\nCoeficientes do Modelo:")
for col, coef in zip(X.columns, modelo.coef_[0]):
    print(f"{col}: {coef:.4f}")