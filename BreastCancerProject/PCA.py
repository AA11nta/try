import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

# df este DataFrame-ul care contine datele
# Variabila dependenta trebuie sa fie exclusa din DataFrame

# Exemplul csv:
df = pd.read_csv("breast-cancer.csv")

# Excluderea variabilei dependenta
# X este DataFrame-ul cu variabilele independente
# Y este seria care contine variabila dependenta
# Daca variabila dependenta este in DataFrame, o putem exclude astfel:
X = df.drop(columns=["diagnosis"])
Y = df["diagnosis"]

# Initializam si aplicatia PCA, specificand numarul de componente principale pe care dorim sa le pastram
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# transformam rezultatele intr-un DataFrame pentru a le putea analiza mai usor
# X_pca_df poate fi utilizata intr-o analiza ulterioara sau in vizualizari pentru a intelege structura datelor
X_pca_df = pd.DataFrame(data=X_pca, columns=['Componenta Principala 1', 'Componenta Principala 2'])

# examinam cum variabilele independente sunt proiectate pe noile componente principale
# investigam cata variatie explicata este capturata de aceste componente
st.write(print("Variatia explicata de fiecare componenta principala:"))
print(pca.explained_variance_ratio_)


# X si Y au fost definite si pregatite anterior, apo impartim datele in setul de antrenare si setul de testare
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Aplicam PCA pe setul de antrenare
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

# Antrenam un model de logistic regression folosind componentele principale
model = LogisticRegression()
model.fit(X_train_pca, y_train)

# Aplicam PCA si pe setul de testare (trebuie sa facem asta pentru a proiecta datele in acelasi spatiu al componentelor principale)
X_test_pca = pca.transform(X_test)

# Calculam probabilitatile de clasificare pentru setul de testare
probs = model.predict_proba(X_test_pca)
# Obtinem probabilitatea asociata clasei 1 (admiterea)
preds = probs[:,1]

# Calculam curba ROC si aria sub curba ROC
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

# Plotam curba ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Calculam predictiile pe datele de testare
y_pred = model.predict(X_test_pca)

# Calculam matricea de confuzie
conf_matrix = confusion_matrix(y_test, y_pred)

# Vizualizam matricea de confuzie
print("Matricea de confuzie:")
print(conf_matrix)

# Calculam matricea de confuzie
conf_matrix = confusion_matrix(y_test, y_pred)

# Vizualizam matricea de confuzie sub forma unui plot
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
