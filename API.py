from turtle import st
import pandas as pd
import numpy as np
import pickle
from pycaret.classification import setup, compare_models, create_model, tune_model, plot_model, evaluate_model, finalize_model, predict_model, save_model, load_model

path = "C:/Users/57321/Downloads/Python/code_APP/"
path2 = "C:/Users/57321/Downloads/Python/Taller1/"
with open(path2 + 'best_model.pkl', 'rb') as model_file:
    modelo = pickle.load(model_file)

def generar_numericas(Avg, Time_App, Time_Web, Length):
    numericas = pd.DataFrame({
        'Avg. Session Length': [Avg],
        'Time on App': [Time_App],
        'Time on Website': [Time_Web],
        'Length of Membership': [Length]
    })
    numericas["x1"] = numericas["Time on Website"] / numericas["Avg. Session Length"]
    numericas["x2"] = numericas["Time on App"] / numericas["Avg. Session Length"]
    numericas["x3"] = numericas["Avg. Session Length"] ** 2
    numericas["x4"] = numericas["Time on App"] * numericas["Length of Membership"]
    return numericas

def generar_dummies(dominio, Address, Tec, numericas):
    # Crear variables dummy para los dominios
    dm = pd.DataFrame({'gmail': [0], 'hotmail': [0], 'yahoo': [0]})
    if dominio in dm.columns:
        dm[dominio] = 1

    # Crear variables dummy para las combinaciones de dirección y dominio
    dm2 = pd.DataFrame({
        'Ausburgo_Otro':[0], 'Ausburgo_gmail': [0], 'Ausburgo_hotmail': [0], 'Ausburgo_yahoo': [0], 
        'Berlin_Otro': [0], 'Berlin_gmail': [0], 'Berlin_hotmail': [0], 'Berlin_yahoo': [0], 
        'Frankfurt_Otro': [0], 'Frankfurt_gmail': [0], 'Frankfurt_hotmail': [0], 'Frankfurt_yahoo': [0], 
        'Munich_Otro': [0], 'Munich_gmail': [0], 'Munich_hotmail': [0], 'Munich_yahoo': [0]
    })
    dominio2 = Address + "_" + dominio
    if dominio2 in dm2.columns:
        dm2[dominio2] = 1

    # Concatenar dummies de dominio y dirección
    numericas = pd.concat([numericas, dm], axis=1)
    numericas = pd.concat([numericas, dm2], axis=1)

# Variables dummy adicionales para dominios y tecnología
    dm = pd.DataFrame({'Otro_dm': [0], 'gmail_dm': [0], 'hotmail_dm': [0], 'yahoo_dm': [0]})
    if dominio + "_dm" in dm.columns:
        dm[dominio + "_dm"] = 1

    dum = pd.DataFrame({})
    dum = pd.concat([dum, dm], axis=1)

    dm = pd.DataFrame({'Iphone_dm': [0], 'PC_dm': [0], 'Portatil_dm': [0], 'Smartphone_dm': [0]})
    if Tec + "_dm" in dm.columns:
        dm[Tec + "_dm"] = 1

    dum = pd.concat([dum, dm], axis=1)
    return numericas, dum

# Aplicación de Streamlit
st.title("Predicción del precio de la APP basada en características de usuario")

# Entradas del usuario
dominio = st.selectbox("Seleccione el dominio:", ['gmail', 'Otro', 'hotmail', 'yahoo'])
Tec = st.selectbox("Seleccione el tipo de dispositivo:", ['Smartphone', 'Portatil', 'PC', 'Iphone'])
Address = st.selectbox("Seleccione la dirección:", ['Munich', 'Ausburgo', 'Berlin', 'Frankfurt'])

# Campos numéricos sin valor predeterminado
Avg = st.text_input("Ingrese Avg. Session Length:", value="")
Time_App = st.text_input("Ingrese Time on App:", value="")
Time_Web = st.text_input("Ingrese Time on Website:", value="")
Length = st.text_input("Ingrese Length of Membership:", value="")

# Convertir los valores de texto a números si es posible
if st.button("Calcular"):
    try:
        Avg = float(Avg)
        Time_App = float(Time_App)
        Time_Web = float(Time_Web)
        Length = float(Length)

        # Generar datos de entrada
        numericas = generar_numericas(Avg, Time_App, Time_Web, Length)
        numericas, dum = generar_dummies(dominio, Address, Tec, numericas)

        # Concatenar los dataframes
        covariables = ['Avg. Session Length', 'Time on App', 'Time on Website',
                       'Length of Membership', 'x1', 'x2', 'x3', 'x4', 'gmail', 'hotmail',
                       'yahoo', 'Ausburgo_gmail', 'Ausburgo_hotmail', 'Ausburgo_yahoo',
                       'Berlin_Otro', 'Berlin_gmail', 'Berlin_hotmail', 'Berlin_yahoo',
                       'Frankfurt_Otro', 'Frankfurt_gmail', 'Frankfurt_hotmail',
                       'Frankfurt_yahoo', 'Munich_Otro', 'Munich_gmail', 'Munich_hotmail',
                       'Munich_yahoo', 'Otro_dm', 'gmail_dm', 'hotmail_dm', 'yahoo_dm',
                       'Iphone_dm', 'PC_dm', 'Portatil_dm', 'Smartphone_dm']

        base_modelo2 = pd.concat([numericas, dum], axis=1)
        base_modelo2 = base_modelo2[covariables]

        # Predicción
        yhat = modelo.predict(base_modelo2)
        st.markdown(f"<p class='big-font'>Predicción: {np.round(float(yhat), 2)}</p>", unsafe_allow_html=True)
    except ValueError:
        st.error("Por favor, ingrese valores numéricos válidos en todos los campos.")

# Botón para reiniciar la consulta
if st.button("Reiniciar"):
    st.experimental_rerun()