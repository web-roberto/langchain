# ROB
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
#from langchain.schema import AIMessage, HumanMessage, SystemMessage
# Ahora estan en estas librerias:
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.system import SystemMessage
#from langchain.prompts import PromptTemplate # ANTES versión 1.0.2
from langchain_core.prompts import PromptTemplate # DESPUES versión 1.0.2

#en una variable de entorno del sistems debo tener la OPENAI_API_KEY=
load_dotenv()
# Configuración inicial
st.set_page_config(page_title="Chatbot Básico", page_icon="🤖")
st.title("🤖 Chatbot Básico con LangChain")
st.markdown("Este es un *chatbot de ejemplo* construido con LangChain + Streamlit. ¡Escribe tu mensaje abajo para comenzar!")

with st.sidebar:
    st.header("Configuración")
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.5, 0.1)
    model_name = st.selectbox("Modelo", ["gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"])
    # Recrear el modelo con nuevos parámetros seleccionados en pantalla
    chat_model = ChatOpenAI(model=model_name, temperature=temperature)
# MEMORIA a base de guardar el historial de mensajes
# Inicializar el historial de mensajes en session_state si no hay mensaje creando una lista vacía
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []
# Crear el template de prompt con comportamiento específico
prompt_template = PromptTemplate(
    input_variables=["mensaje", "historial"],
    template="""Eres un asistente útil y amigable llamado ChatBot Pro. 

Historial de conversación:
{historial}

Responde de manera clara y concisa a la siguiente pregunta: {mensaje}"""
)
# Crear cadena usando LCEL (LangChain Expression Language)
cadena = prompt_template | chat_model # el la cadena que le paso el modelo
# Pinto el historial existente pq esto se ejecuta con cada interacción del usuario (el streamlit)
for msg in st.session_state.mensajes:
    if isinstance(msg, SystemMessage):
        continue  # no mostrar al usuario los mensajes del sistema
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(msg.content) # va pintando los mensajes por la pantalla
if st.button("🗑️ Nueva conversación"):
    st.session_state.mensajes = [] # vacío los mensajes y empiezao de nuevo
    st.rerun()
# Input de usuario
pregunta = st.chat_input("Escribe tu mensaje:")
if pregunta:
    with st.chat_message("user"): # Mostrar y almacenar mensaje del usuario
        st.markdown(pregunta) #muestro la pregunta por pantalla
    # Generar y mostrar respuesta del asistente a la pregunta del usuario
    try:
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            # Streaming de la respuesta: va juntando los chats que me da openAI como respuesta
            # le pido una respuesta = f(mensaje, historial)
            # en lugar de .inkove uso .stream y le asigno valor a las variables
            for chunk in cadena.stream({"mensaje": pregunta, "historial": st.session_state.mensajes}):
                full_response += chunk.content
                # el símbolo ▌ es un cursor y parece que voy escribiendo los chunks a medida que los recibo
                response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
        # creo un obj HumanMessage y meto la pregunta en el historial de mensajes como mensaje humano
        st.session_state.mensajes.append(HumanMessage(content=pregunta))
        # creo un obj AIMessage y meto la respuesta en el historial de mensajes como mensaje de AI
        st.session_state.mensajes.append(AIMessage(content=full_response))
    except Exception as e:
        st.error(f"Error al generar respuesta: {str(e)}")
        st.info("Verifica que tu API Key de OpenAI esté configurada correctamente.")