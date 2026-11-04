# Application Streamlit : 4 modèles (2 classifications + 2 régressions)
# Même structure que l'application du taxi vue en classe : Pred_func, Pred_func_csv, onglets simple / multiple
# Les fichiers .joblib sont ceux produits par nos propres notebooks (chargés depuis ce dépôt : source de confiance)
from pathlib import Path
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Configuration de la page
st.set_page_config(page_title="Machine Learning App", page_icon="🤖", layout="centered")

BASE = Path(__file__).parent

# Traduction d'affichage des valeurs des listes déroulantes (le modèle reçoit la valeur d'origine)
FR = {'Morning': 'Matin', 'Afternoon': 'Après-midi', 'Evening': 'Soir', 'Night': 'Nuit',
      'Weekday': 'Semaine', 'Weekend': 'Week-end', 'Low': 'Faible', 'Medium': 'Moyen', 'High': 'Élevé',
      'Clear': 'Dégagé', 'Rain': 'Pluie', 'Snow': 'Neige',
      'Petrol': 'Essence', 'Diesel': 'Diesel', 'CNG': 'GNC', 'Dealer': 'Concessionnaire', 'Individual': 'Particulier',
      'Automatic': 'Automatique', 'Manual': 'Manuelle'}


def fr(valeur):
    return FR.get(valeur, valeur)
# Nom du fichier du meilleur modèle sauvegardé dans chaque dossier (voir la fin de chaque notebook)
MODEL_FILES = {'Car_Data': 'dt_model.joblib', 'Telephone': 'rf_model.joblib',
               'Car_Dataset': 'rf_model.joblib', 'Taxi_Trip': 'rf_model.joblib'}


# Charger les objets sauvegardés (une seule fois grâce au cache)
@st.cache_resource
def load_objects(folder):
    # importer les encodeurs
    encoders = joblib.load(BASE / folder / 'encoders.joblib')
    # importer le best modèle
    model = joblib.load(BASE / folder / MODEL_FILES[folder])
    # importer les valeurs uniques
    uniques = joblib.load(BASE / folder / 'uniques.joblib')
    # importer le scaler
    scaler = joblib.load(BASE / folder / 'scaler.joblib')
    return encoders, model, uniques, scaler


# =====================================================================================
# 1) CLASSIFICATION : Car_Data
# =====================================================================================
def app_car_data():
    encoders, model, uniques, scaler = load_objects('Car_Data')

    # Fonction de prédiction
    def Pred_func(Marque, Annee, Boite_Vitesse, Adresse, Prix):
        # Encoder les valeurs de Marque, Boite_Vitesse et Adresse
        Marque = encoders[0].transform([Marque])[0]
        Boite_Vitesse = encoders[1].transform([Boite_Vitesse])[0]
        Adresse = encoders[2].transform([Adresse])[0]
        # vecteur des valeurs
        x_new = np.array([Marque, Annee, Boite_Vitesse, Adresse, Prix], dtype=float)
        x_new = x_new.reshape(1, -1)  # convertir en un 2D array
        # normaliser
        x_new = scaler.transform(x_new)
        # Prédire
        y_pred = model.predict(x_new)
        # Décoder la classe prédite
        return encoders[-1].inverse_transform([int(y_pred[0])])[0]

    # Fonction de prédiction multiple
    def Pred_func_csv(df):
        predictions = []
        # Boucle sur les lignes du dataframe
        for row in df.iloc[:, :5].values:
            y_pred = Pred_func(row[0], row[1], row[2], row[3], row[4])
            predictions.append(y_pred)
        df['Venant_Occasion_Prediction'] = predictions
        return df

    tab1, tab2 = st.tabs(['Simple Prédiction', 'Prédiction multiple'])

    with tab1:
        st.subheader("Prédire si une voiture est « Venant » ou « D'occasion » (une seule entrée)")
        st.write("""Ce modèle de machine learning permet de prédire si une voiture est Venant ou D'occasion
                 à partir de la marque, de l'année, de la boîte de vitesse, de l'adresse et du prix.""")
        with st.form('form_simple_car_data'):
            col1, col2 = st.columns(2)
            with col1:
                Marque = st.selectbox('Marque', options=uniques[0])
                Annee = st.number_input('Année', min_value=1990, max_value=2030, value=2015, step=1)
                Boite_Vitesse = st.selectbox('Boîte de vitesse', options=uniques[1])
            with col2:
                Adresse = st.selectbox('Adresse', options=uniques[2])
                Prix = st.number_input('Prix', min_value=0.0, value=7800000.0, step=100000.0)
            submit = st.form_submit_button('Prédire')
        if submit:
            y_pred = Pred_func(Marque, Annee, Boite_Vitesse, Adresse, Prix)
            st.success(f"Résultat : la voiture est **{y_pred}**")

    with tab2:
        st.subheader("Prédire si des voitures sont « Venant » ou « D'occasion » (plusieurs entrées)")
        st.write("""Le fichier csv doit contenir les colonnes : Marque, Année, Boite_Vitesse, Adresse, Prix.""")
        file = st.file_uploader('Téléverser un fichier csv', type=['csv'], key='upload_car_data')
        if file is not None:
            # Lire le fichier csv
            df = pd.read_csv(file)
            st.write("Aperçu des données chargées :")
            st.dataframe(df.head())
            if st.button('Prédire', key='predire_multiple_car_data'):
                try:
                    df_pred = Pred_func_csv(df)
                except Exception as e:
                    st.error(f"Erreur : vérifiez les colonnes et les valeurs du fichier ({e})")
                else:
                    st.write("Résultats des prédictions :")
                    st.dataframe(df_pred)
                    # Télécharger le fichier des prédictions
                    st.download_button(label='Télécharger le fichier csv',
                                       data=df_pred.to_csv(index=False).encode('utf-8'),
                                       file_name='predictions.csv', mime='text/csv',
                                       key='telecharger_car_data')


# =====================================================================================
# 2) CLASSIFICATION : Telephone
# =====================================================================================
def app_telephone():
    encoders, model, uniques, scaler = load_objects('Telephone')

    # Fonction de prédiction
    def Pred_func(prix, adresse, marque, dim_ecr, ram, stockage):
        # Encoder les valeurs de adresse et marque
        adresse = encoders[0].transform([adresse])[0]
        marque = encoders[1].transform([marque])[0]
        # vecteur des valeurs
        x_new = np.array([prix, adresse, marque, dim_ecr, ram, stockage], dtype=float)
        x_new = x_new.reshape(1, -1)  # convertir en un 2D array
        # normaliser
        x_new = scaler.transform(x_new)
        # Prédire
        y_pred = model.predict(x_new)
        # Décoder la classe prédite
        return encoders[-1].inverse_transform([int(y_pred[0])])[0]

    # Fonction de prédiction multiple
    def Pred_func_csv(df):
        predictions = []
        # Boucle sur les lignes du dataframe
        for row in df.iloc[:, :6].values:
            y_pred = Pred_func(row[0], row[1], row[2], row[3], row[4], row[5])
            predictions.append(y_pred)
        df['etat_Prediction'] = predictions
        return df

    tab1, tab2 = st.tabs(['Simple Prédiction', 'Prédiction multiple'])

    with tab1:
        st.subheader("Prédire l'état d'un téléphone (une seule entrée)")
        st.write("""Ce modèle de machine learning permet de prédire l'état d'un téléphone
                 (Neuf, Réconditionné, D'occasion ou Venant) à partir du prix, de l'adresse, de la marque, de l'écran, de la RAM et du stockage.""")
        with st.form('form_simple_telephone'):
            col1, col2 = st.columns(2)
            with col1:
                prix = st.number_input('Prix', min_value=0.0, value=360000.0, step=10000.0)
                adresse = st.selectbox('Adresse', options=uniques[0])
                marque = st.selectbox('Marque', options=uniques[1])
            with col2:
                dim_ecr = st.number_input("Dimension de l'écran", min_value=0.0, value=6.0, step=0.1)
                ram = st.number_input('RAM', min_value=1, value=8, step=1)
                stockage = st.number_input('Stockage', min_value=1, value=128, step=32)
            submit = st.form_submit_button('Prédire')
        if submit:
            y_pred = Pred_func(prix, adresse, marque, dim_ecr, ram, stockage)
            st.success(f"Résultat : l'état du téléphone est **{y_pred}**")

    with tab2:
        st.subheader("Prédire l'état de plusieurs téléphones (plusieurs entrées)")
        st.write("""Le fichier csv doit contenir les colonnes : prix, adresse, marque, dim_ecr, ram, stockage.""")
        file = st.file_uploader('Téléverser un fichier csv', type=['csv'], key='upload_telephone')
        if file is not None:
            # Lire le fichier csv
            df = pd.read_csv(file)
            st.write("Aperçu des données chargées :")
            st.dataframe(df.head())
            if st.button('Prédire', key='predire_multiple_telephone'):
                try:
                    df_pred = Pred_func_csv(df)
                except Exception as e:
                    st.error(f"Erreur : vérifiez les colonnes et les valeurs du fichier ({e})")
                else:
                    st.write("Résultats des prédictions :")
                    st.dataframe(df_pred)
                    # Télécharger le fichier des prédictions
                    st.download_button(label='Télécharger le fichier csv',
                                       data=df_pred.to_csv(index=False).encode('utf-8'),
                                       file_name='predictions.csv', mime='text/csv',
                                       key='telecharger_telephone')


# =====================================================================================
# 3) RÉGRESSION : Car_dataset
# =====================================================================================
def app_car_dataset():
    encoders, model, uniques, scaler = load_objects('Car_Dataset')

    # Fonction de prédiction
    def Pred_func(Kms_Driven, Present_Price, Fuel_Type, Seller_Type, Transmission, Age):
        # Encoder les valeurs de Fuel_Type, Seller_Type et Transmission
        Fuel_Type = encoders[0].transform([Fuel_Type])[0]
        Seller_Type = encoders[1].transform([Seller_Type])[0]
        Transmission = encoders[2].transform([Transmission])[0]
        # vecteur des valeurs
        x_new = np.array([Kms_Driven, Present_Price, Fuel_Type, Seller_Type, Transmission, Age], dtype=float)
        x_new = x_new.reshape(1, -1)  # convertir en un 2D array
        # normaliser
        x_new = scaler.transform(x_new)
        # Prédire
        y_pred = model.predict(x_new)
        # Arrondir
        return round(float(y_pred[0]), 2)

    # Fonction de prédiction multiple
    def Pred_func_csv(df):
        predictions = []
        # Boucle sur les lignes du dataframe
        for row in df.iloc[:, :6].values:
            y_pred = Pred_func(row[0], row[1], row[2], row[3], row[4], row[5])
            predictions.append(y_pred)
        df['Selling_Price_Prediction'] = predictions
        return df

    tab1, tab2 = st.tabs(['Simple Prédiction', 'Prédiction multiple'])

    with tab1:
        st.subheader("Prédire le prix de vente d'une voiture (une seule entrée)")
        st.write("""Ce modèle de machine learning permet de prédire le prix de vente d'une voiture
                 à partir du kilométrage, du prix actuel, du carburant, du type de vendeur, de la transmission et de l'âge.""")
        with st.form('form_simple_car_dataset'):
            col1, col2 = st.columns(2)
            with col1:
                Kms_Driven = st.number_input('Kilométrage (Kms_Driven)', min_value=0.0, value=70000.0, step=1000.0)
                Present_Price = st.number_input('Prix actuel (Present_Price)', min_value=0.0, value=15.0, step=0.5)
                Fuel_Type = st.selectbox('Carburant (Fuel_Type)', options=uniques[0], format_func=fr)
            with col2:
                Seller_Type = st.selectbox('Type de vendeur (Seller_Type)', options=uniques[1], format_func=fr)
                Transmission = st.selectbox('Transmission', options=uniques[2], format_func=fr)
                Age = st.number_input('Âge de la voiture (Age)', min_value=0, value=8, step=1)
            submit = st.form_submit_button('Prédire')
        if submit:
            y_pred = Pred_func(Kms_Driven, Present_Price, Fuel_Type, Seller_Type, Transmission, Age)
            st.success(f"Résultat : prix de vente estimé = **{y_pred}**")

    with tab2:
        st.subheader("Prédire le prix de vente de plusieurs voitures (plusieurs entrées)")
        st.write("""Le fichier csv doit contenir les colonnes : Kms_Driven, Present_Price, Fuel_Type, Seller_Type, Transmission, Age.""")
        file = st.file_uploader('Téléverser un fichier csv', type=['csv'], key='upload_car_dataset')
        if file is not None:
            # Lire le fichier csv
            df = pd.read_csv(file)
            st.write("Aperçu des données chargées :")
            st.dataframe(df.head())
            if st.button('Prédire', key='predire_multiple_car_dataset'):
                try:
                    df_pred = Pred_func_csv(df)
                except Exception as e:
                    st.error(f"Erreur : vérifiez les colonnes et les valeurs du fichier ({e})")
                else:
                    st.write("Résultats des prédictions :")
                    st.dataframe(df_pred)
                    # Télécharger le fichier des prédictions
                    st.download_button(label='Télécharger le fichier csv',
                                       data=df_pred.to_csv(index=False).encode('utf-8'),
                                       file_name='predictions.csv', mime='text/csv',
                                       key='telecharger_car_dataset')


# =====================================================================================
# 4) RÉGRESSION : Taxi trip (application vue en classe)
# =====================================================================================
def app_taxi():
    encoders, model, uniques, scaler = load_objects('Taxi_Trip')

    # Fonction de prédiction
    def Pred_func(Trip_Distance_km, Time_of_Day, Day_of_Week, Passenger_Count, Traffic_Conditions,
                  Weather, Base_Fare, Per_Km_Rate, Per_Minute_Rate, Trip_Duration_Minutes):
        # Encoder les valeurs des Time_of_Day, Day_of_Week, Traffic_Conditions et Weather
        Time_of_Day = encoders[0].transform([Time_of_Day])[0]
        Day_of_Week = encoders[1].transform([Day_of_Week])[0]
        Traffic_Conditions = encoders[2].transform([Traffic_Conditions])[0]
        Weather = encoders[3].transform([Weather])[0]
        # vecteur des valeurs numériques
        x_new = np.array([Trip_Distance_km, Time_of_Day, Day_of_Week, Passenger_Count, Traffic_Conditions,
                          Weather, Base_Fare, Per_Km_Rate, Per_Minute_Rate, Trip_Duration_Minutes], dtype=float)
        x_new = x_new.reshape(1, -1)  # convertir en un 2D array
        # normaliser
        x_new = scaler.transform(x_new)
        # Prédire
        y_pred = model.predict(x_new)
        # Arrondir
        return round(float(y_pred[0]), 2)

    # Fonction de prédiction multiple
    def Pred_func_csv(df):
        predictions = []
        # Boucle sur les lignes du dataframe
        for row in df.iloc[:, :10].values:
            y_pred = Pred_func(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9])
            predictions.append(y_pred)
        df['Trip_Price_Prediction'] = predictions
        return df

    tab1, tab2 = st.tabs(['Simple Prédiction', 'Prédiction multiple'])

    with tab1:
        st.subheader("Prédire le prix d'une course de taxi (une seule entrée)")
        st.write("""Ce modèle de machine learning permet de prédire le prix d'une course de taxi
                 à partir de la distance, du moment de la journée, du jour de la semaine, du nombre de passagers, du trafic, de la météo,
                 du tarif de base, du tarif par km, du tarif par minute et de la durée du trajet.""")
        with st.form('form_simple_taxi'):
            col1, col2 = st.columns(2)
            with col1:
                Trip_Distance_km = st.number_input('Distance du trajet (km)', min_value=0.0, value=25.0, step=0.5)
                Time_of_Day = st.selectbox('Moment de la journée', options=uniques[0], format_func=fr)
                Day_of_Week = st.selectbox('Jour de la semaine', options=uniques[1], format_func=fr)
                Passenger_Count = st.number_input('Nombre de passagers', min_value=1, max_value=4, value=2, step=1)
                Traffic_Conditions = st.selectbox('Conditions de circulation', options=uniques[2], format_func=fr)
            with col2:
                Weather = st.selectbox('Météo', options=uniques[3], format_func=fr)
                Base_Fare = st.number_input('Tarif de base', min_value=0.0, value=3.5, step=0.1)
                Per_Km_Rate = st.number_input('Tarif par km', min_value=0.0, value=1.2, step=0.05)
                Per_Minute_Rate = st.number_input('Tarif par minute', min_value=0.0, value=0.3, step=0.01)
                Trip_Duration_Minutes = st.number_input('Durée du trajet (minutes)', min_value=0.0, value=60.0, step=1.0)
            submit = st.form_submit_button('Prédire')
        if submit:
            y_pred = Pred_func(Trip_Distance_km, Time_of_Day, Day_of_Week, Passenger_Count, Traffic_Conditions,
                               Weather, Base_Fare, Per_Km_Rate, Per_Minute_Rate, Trip_Duration_Minutes)
            st.success(f"Résultat : prix estimé de la course = **{y_pred} $**")

    with tab2:
        st.subheader("Prédire le prix de plusieurs courses de taxi (plusieurs entrées)")
        st.write("""Ce modèle de machine learning permet de prédire le prix d'une course de taxi
                 à partir de la distance, du moment de la journée, du jour de la semaine, du nombre de passagers, du trafic, de la météo,
                 du tarif de base, du tarif par km, du tarif par minute et de la durée du trajet.""")
        file = st.file_uploader('Téléverser un fichier csv', type=['csv'], key='upload_taxi')
        if file is not None:
            # Lire le fichier csv
            df = pd.read_csv(file)
            st.write("Aperçu des données chargées :")
            st.dataframe(df.head())
            if st.button('Prédire', key='predire_multiple_taxi'):
                try:
                    df_pred = Pred_func_csv(df)
                except Exception as e:
                    st.error(f"Erreur : vérifiez les colonnes et les valeurs du fichier ({e})")
                else:
                    st.write("Résultats des prédictions :")
                    st.dataframe(df_pred)
                    # Télécharger le fichier des prédictions
                    st.download_button(label='Télécharger le fichier csv',
                                       data=df_pred.to_csv(index=False).encode('utf-8'),
                                       file_name='predictions.csv', mime='text/csv',
                                       key='telecharger_taxi')


# =====================================================================================
# Menu : choix du modèle
# =====================================================================================
st.sidebar.title("🤖 Machine Learning")
choix = st.sidebar.radio("Choisissez un modèle", [
    "🚗 Car Data (Classification)",
    "📱 Telephone (Classification)",
    "🚙 Car Dataset (Régression)",
    "🚕 Taxi Trip (Régression)"])

st.title(choix)
if choix.startswith("🚗"):
    app_car_data()
elif choix.startswith("📱"):
    app_telephone()
elif choix.startswith("🚙"):
    app_car_dataset()
else:
    app_taxi()
