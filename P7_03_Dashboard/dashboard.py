import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
import lime
from lime import lime_tabular
from urllib.request import urlopen
import streamlit.components.v1 as components
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# Chargement des données :

df = pd.read_csv('data_modelisation.csv')
y = df['TARGET']
df.drop(columns='TARGET', inplace=True)

# Chargement des données originals des clients :
df_original = pd.read_csv('original_data.csv')

# Chargement du modèle :

model = pickle.load(open('credit_final.pkl', 'rb'))

liste_id = df['SK_ID_CURR'].values


def main():
    st.title('Dashboard Scoring Credit')
    st.markdown("Prédictions de scoring client, notre seuil de choix est de 50 %")
    # hobby = st.selectbox(" Veuillez choisir un identifiant à saisir: ", liste_id)

    id_input = st.number_input(label='Veuillez saisir l\'identifiant du client demandeur de crédit:', format="%i",
                               value=0)
    if id_input not in liste_id:

        st.write("L'identifiant client n'est pas bon")


    elif (int(id_input) in liste_id):
        #API_url = "http://127.0.0.1:5000/credit/" + str(id_input)
        API_url = "https://creditprediction.herokuapp.com/credit/"+str(id_input)
        with st.spinner('Chargement du score du client...'):
            json_url = urlopen(API_url)
            API_data = json.loads(json_url.read())
            classe_predite = API_data['prediction']
            pred_prob = API_data['probability']

            st.subheader('Le statut de la demande de crédit')
            if classe_predite == 'Acceptation de la demande de credit':
                html_temp1 = "<html><span style='color:green;font-weight: bold;font-size: 18px;'>" + classe_predite + "</span></html>"
                st.markdown(html_temp1, unsafe_allow_html=True)

            else:
                html_temp2 = "<html><span style='color:red;font-weight: bold;font-size: 18px;'>" + classe_predite + "</span></html>"
                st.markdown(html_temp2, unsafe_allow_html=True)

            # html_temp3 = "<html><span style='color:red'>" +chaine2+ "</span></html>"
            st.subheader('La probabilité de scoring du client')
            chaine = round(pred_prob * 100, 0)
            chaine2 = str(chaine)
            if chaine < 50:
                html_temp3 = "<html><span style='color:red;font-weight: bold;font-size: 18px;'>" + chaine2 + "</span></html>"
                st.markdown(html_temp3, unsafe_allow_html=True)
            else:
                html_temp4 = "<html><span style='color:green;font-weight: bold;font-size: 18px;'>" + chaine2 + "</span></html>"
                st.markdown(html_temp4, unsafe_allow_html=True)

            # Jauge colorée :
            data = df.copy()
            data.drop(columns = 'SK_ID_CURR' , inplace=True)
            pred_prob_all = model.predict_proba(data.values)[:, 1].tolist()

            all_pred_prod = []
            for i in pred_prob_all:
                percent_all = round(i * 100, 0)
                all_pred_prod.append(percent_all)
            data['pred_prob'] = all_pred_prod
            data['threshold'] = data['pred_prob'].apply(lambda x: 1 if x > 50 else 0)
            fig0 = px.histogram(data, data.pred_prob, color='threshold', color_discrete_sequence=['red', 'green'])
            fig0.add_vline(x=round(pred_prob * 100, 0), line_width=3, line_color="blue")
            st.plotly_chart(fig0)

            # Affichage des données des clients :
            st.subheader('Les données du client')
            X1 = df_original[df_original['SK_ID_CURR'] == id_input]
            X1.drop(columns='SK_ID_CURR', inplace=True)
            df_original.drop(columns='SK_ID_CURR', inplace=True)
            st.write(X1)

            # Feature importance pour l'ensemble des clients :
            dataframe = df.drop(['SK_ID_CURR'], axis=1)
            st.subheader("Importance globale des variables")
            coefficients = model.steps[1][1].coef_.tolist().pop()
            coefficients_variables = pd.concat([pd.DataFrame(dataframe.columns, columns=['Caractéristiques']),
                                                pd.DataFrame(coefficients, columns=['Coefficients'])],
                                               axis=1)
            coefficients_variables = coefficients_variables.sort_values(by='Coefficients', ascending=True).tail(10)
            fig1 = px.bar(coefficients_variables, x='Coefficients', y='Caractéristiques')
            st.plotly_chart(fig1)


            #Interprétabilité locale
            X = df[df.SK_ID_CURR == id_input]
            X.drop(columns=['SK_ID_CURR'], inplace=True)

            st.subheader("Importance locale des variables")

            lime_explainer = lime_tabular.LimeTabularExplainer(dataframe.values,
                                                               feature_names=dataframe.columns,
                                                               class_names=list(y.unique()),
                                                               mode="classification")
            idx = X.index[0]
            exp = lime_explainer.explain_instance(X.loc[idx], model.predict_proba, num_features=10)

            components.html(exp.as_html(), height=800)
            exp.as_list()

            # Exploration des caractéristiques du client :
            df_original['Target'] = y
            st.subheader('Exploration des caractéristiques du client/des clients')
            container = st.container()
            select_box = container.multiselect(label='Features', options=df_original.columns)
            if len(select_box)!=0 :
                for i in range(0, len(select_box)):
                    figi = px.histogram(df_original, x=df_original[select_box[i]], color='Target')
                    figi.add_vline(x=X1[select_box[i]].values[0], line_width=3, line_color="green")
                    st.plotly_chart(figi)
            

                

            # Analyse bivariée des caractéristiques des clients :
            st.subheader("Analyse Bivariée des caractéristiques de nos clients")
            select_box2 = st.selectbox(label='Axe des abscisses', options=df_original.columns)
            select_box3 = st.selectbox(label='Axe des ordonnées', options=df_original.columns)
            fig3 = px.scatter(df_original, x=select_box2, y=select_box3)
            st.plotly_chart(fig3)


if __name__ == "__main__":
    main()
