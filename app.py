import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Titre de l'application
st.title("Prédiction de souscription bancaire")

# Charger les données
uploaded_file = st.file_uploader("Télécharger un fichier CSV avec les données", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Aperçu des données :", data.head())

    # Sélection des colonnes
    features = st.multiselect("Sélectionnez les colonnes explicatives :", data.columns)
    target = st.selectbox("Sélectionnez la colonne cible :", data.columns)

    if features and target:
        X = data[features]
        y = data[target]

        # Diviser les données en ensemble d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Créer et entraîner le modèle
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Faire des prédictions
        y_pred = model.predict(X_test)

        # Afficher les résultats
        st.subheader("Résultats du modèle")
        st.text("Rapport de classification :")
        st.text(classification_report(y_test, y_pred))
        st.text("Matrice de confusion :")
        st.write(confusion_matrix(y_test, y_pred))

        # Importance des caractéristiques
        st.subheader("Importance des caractéristiques")
        fig, ax = plt.subplots()
        ax.bar(features, model.feature_importances_)
        ax.set_title("Importance des variables explicatives")
        st.pyplot(fig)
