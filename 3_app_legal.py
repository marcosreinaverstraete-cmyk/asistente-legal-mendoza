import os
import warnings
import zipfile
import gdown  # <-- Agregamos esta herramienta
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURACIÓN DE LA PÁGINA WEB Y API KEY
# ==========================================
st.set_page_config(page_title="Socio Legal AI", page_icon="⚖️", layout="centered")
st.title("⚖️ Asistente Legal AI")
st.markdown("Consulta tus expedientes y libros jurídicos locales de forma privada.")

# Cargar API Key de Streamlit Secrets o archivo local
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    st.error("🚨 ERROR: No se encontró la GOOGLE_API_KEY en los secretos ni en el archivo .env")
    st.stop()

# ==========================================
# 2. CONECTAR LA BASE DE DATOS (El truco de Drive)
# ==========================================
@st.cache_resource
def cargar_base_de_datos():
    # --- EL TRUCO DE GOOGLE DRIVE ---
    if not os.path.exists("./db_vectorial"):
        # 1. Si no está el ZIP, lo bajamos de Drive
        if not os.path.exists("db_vectorial.zip"):
            print("📥 Descargando cerebro desde Google Drive...")
            file_id = '1T3Ij70FpL8qvFj-X6ts2kviMjdvOCncv' 
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, 'db_vectorial.zip', quiet=False)
            
        # 2. Descomprimimos el ZIP
        print("📦 Descomprimiendo...")
        with zipfile.ZipFile("db_vectorial.zip", 'r') as zip_ref:
            zip_ref.extractall("./")
    # --------------------------------
    
    persist_directory = "./db_vectorial"
    modelo_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return Chroma(persist_directory=persist_directory, embedding_function=modelo_embeddings)

vector_db = cargar_base_de_datos()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

# ==========================================
# 3. INTERFAZ DE CHAT
# ==========================================
# Guardamos la memoria del chat para que no se borre al hacer otra pregunta
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# Dibujamos los mensajes anteriores en la pantalla
for mensaje in st.session_state.mensajes:
    with st.chat_message(mensaje["rol"]):
        st.markdown(mensaje["contenido"])

# ==========================================
# 4. EL MOTOR DE BÚSQUEDA
# ==========================================
# Esta es la barrita de abajo donde el usuario escribe
pregunta = st.chat_input("Escribí tu pregunta legal acá...")

if pregunta:
    # Mostramos la pregunta del usuario en la web
    with st.chat_message("user"):
        st.markdown(pregunta)
    st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})

    # Mostramos a la IA pensando
    with st.chat_message("assistant"):
        with st.spinner("🕵️‍♂️ Buscando en los libros y redactando dictamen..."):
            
            # Buscamos los k=8 mejores fragmentos
            documentos = vector_db.similarity_search(pregunta, k=8)
            
            contexto = ""
            for doc in documentos:
                rama = doc.metadata.get("rama", "Desconocida")
                archivo = doc.metadata.get("source", "Desconocido")
                contexto += f"\n--- [Rama: {rama}, Archivo: {archivo}] ---\n{doc.page_content}\n"
            
            # El Prompt
            prompt_final = f"""Sos un abogado experto en derecho administrativo, ambiental, hídrico y minero de Argentina.
            Usá EXCLUSIVAMENTE los siguientes fragmentos para responder.

            REGLAS ESTRICTAS:
            1. Si la respuesta no está en el contexto, decí: "No encontré esta información en mis registros". NO INVENTES NADA.
            2. Al final de cada dato clave, CITÁ la rama y el archivo exacto.

            Contexto recuperado:
            {contexto}

            Pregunta del abogado: {pregunta}
            Dictamen Legal:"""

            try:
                # Llamamos a Gemini
                respuesta = llm.invoke(prompt_final)
                
                # Escribimos la respuesta en la web
                st.markdown(respuesta.content)
                
                # Le agregamos un botón desplegable (acordeón) para ver las fuentes
                with st.expander("👁️ Ver fragmentos originales usados por la IA"):
                    st.text(contexto)
                    
                st.session_state.mensajes.append({"rol": "assistant", "contenido": respuesta.content})
                
            except Exception as e:
                st.error(f"❌ Error al consultar a Gemini: {e}")