from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import streamlit as st

# Cargar variables de entorno
load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Configuración del modelo de LangChain
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Interfaz con Streamlit
st.title("Asistente de Preguntas con LangChain")
st.write("Introduce una pregunta y recibe la respuesta generada por el modelo.")

# Entrada de la pregunta por el usuario
pregunta = st.text_input("Pregunta:")

if pregunta:
    st.write("Pregunta: ", pregunta)

    # Obtener la respuesta del modelo
    respuesta = llm.invoke(pregunta)
    st.write("Respuesta del modelo: ", respuesta.content)

