import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Încarcă setul de date
df = pd.read_csv("breast-cancer.csv")

# Exclude variabila dependentă
X = df.drop(columns=["diagnosis"])
Y = df["diagnosis"]

# Împarte datele în seturi de antrenament și testare
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Normalizează datele
X_train = X_train / X_train.max()
X_test = X_test / X_test.max()

# Definirea Autoencoderului
input_dim = X_train.shape[1]
encoding_dim = 10  # Numărul de dimensiuni reduse

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
encoder = Model(inputs=input_layer, outputs=encoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, shuffle=True, validation_split=0.2, verbose=0)

# Reduce dimensiunile datelor folosind Autoencoder
X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)

# Aplică PCA pentru reducerea dimensiunilor
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Antrenează un model Random Forest folosind datele reduse dimensional (Autoencoder)
model_rf_ae = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf_ae.fit(X_train_encoded, y_train)

# Antrenează un model Random Forest folosind datele reduse dimensional (PCA)
model_rf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf_pca.fit(X_train_pca, y_train)

# Calculează probabilitățile de clasificare pentru setul de testare (Autoencoder)
probs_rf_ae = model_rf_ae.predict_proba(X_test_encoded)
preds_rf_ae = probs_rf_ae[:, 1]

# Calculează probabilitățile de clasificare pentru setul de testare (PCA)
probs_rf_pca = model_rf_pca.predict_proba(X_test_pca)
preds_rf_pca = probs_rf_pca[:, 1]

# Calculează curba ROC și AUC (Autoencoder)
fpr_rf_ae, tpr_rf_ae, threshold_rf_ae = roc_curve(y_test, preds_rf_ae)
roc_auc_rf_ae = auc(fpr_rf_ae, tpr_rf_ae)

# Calculează curba ROC și AUC (PCA)
fpr_rf_pca, tpr_rf_pca, threshold_rf_pca = roc_curve(y_test, preds_rf_pca)
roc_auc_rf_pca = auc(fpr_rf_pca, tpr_rf_pca)

# Plotează curba ROC pentru Autoencoder și PCA
plt.figure()
plt.plot(fpr_rf_ae, tpr_rf_ae, color='darkorange', lw=2, label='Autoencoder ROC curve (area = %0.2f)' % roc_auc_rf_ae)
plt.plot(fpr_rf_pca, tpr_rf_pca, color='blue', lw=2, label='PCA ROC curve (area = %0.2f)' % roc_auc_rf_pca)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Rata Fals Pozitive')
plt.ylabel('Rata Adevărat Pozitive')
plt.title('Caracteristica de Operare a Receptorului')
plt.legend(loc="lower right")
plt.show()

# Calculează predicțiile pe datele de testare (Autoencoder)
y_pred_rf_ae = model_rf_ae.predict(X_test_encoded)

# Calculează predicțiile pe datele de testare (PCA)
y_pred_rf_pca = model_rf_pca.predict(X_test_pca)

# Calculează matricea de confuzie (Autoencoder)
conf_matrix_rf_ae = confusion_matrix(y_test, y_pred_rf_ae)

# Calculează matricea de confuzie (PCA)
conf_matrix_rf_pca = confusion_matrix(y_test, y_pred_rf_pca)

# Afișează matricea de confuzie (Autoencoder)
print("Matricea de confuzie (Autoencoder):")
print(conf_matrix_rf_ae)

# Afișează matricea de confuzie (PCA)
print("Matricea de confuzie (PCA):")
print(conf_matrix_rf_pca)

# Vizualizează matricea de confuzie (Autoencoder)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_rf_ae, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Etichete Prezise')
plt.ylabel('Etichete Adevărate')
plt.title('Matricea de Confuzie (Autoencoder)')
plt.show()

# Vizualizează matricea de confuzie (PCA)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_rf_pca, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Etichete Prezise')
plt.ylabel('Etichete Adevărate')
plt.title('Matricea de Confuzie (PCA)')
plt.show()

# Calculează și afișează scorul de acuratețe (Autoencoder)
accuracy_rf_ae = accuracy_score(y_test, y_pred_rf_ae)
print(f'Acuratețea (Autoencoder): {accuracy_rf_ae:.2f}')

# Calculează și afișează scorul de acuratețe (PCA)
accuracy_rf_pca = accuracy_score(y_test, y_pred_rf_pca)
print(f'Acuratețea (PCA): {accuracy_rf_pca:.2f}')
