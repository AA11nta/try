import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Incarcam datele din fisierul CSV
data = pd.read_csv("breast-cancer.csv")

# Separarea variabilelor independente si a variabilei dependente
X = data.drop(columns=["diagnosis"])  # Variabile independente
y = data["diagnosis"]  # Variabila dependenta

# Impartirea datelor in setul de antrenare si setul de testare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializam si antrenam clasificatorul KNN
knn = KNeighborsClassifier(n_neighbors=5)  # Specificam numarul de vecini (k)
knn.fit(X_train, y_train)

# Facem predictii pe setul de testare
y_pred = knn.predict(X_test)

# Evaluam acuratetea modelului
accuracy = accuracy_score(y_test, y_pred)
print("Acuratetea modelului KNN:", accuracy)

# Calculam si afisam matricea de confuzie
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
plt.title('Confusion Matrix')
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

# Calculam probabilitatile de apartenenta la clasa pozitiva (clasa 1)
y_probs = knn.predict_proba(X_test)[:, 1]

# Calculam ratele de false positive si true positive pentru curba ROC
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Calculam aria sub curba ROC (AUC)
roc_auc = roc_auc_score(y_test, y_probs)

# Afisam curba ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
