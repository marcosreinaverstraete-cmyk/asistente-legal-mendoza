import os
import warnings
import zipfile
import gdown
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURACIÓN Y MODELOS
# ==========================================
st.set_page_config(page_title="Socio Legal AI", page_icon="⚖️", layout="wide")
st.title("⚖️ Asistente Legal AI: Modo Socio")

if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    load_dotenv()

# Inicializamos Gemini 2.5 Flash (Rápido y con mucha memoria)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
# ==========================================
# 2. CARGA DE BASE DE DATOS (Dinamismo de Drive)
# ==========================================
@st.cache_resource
def cargar_base_de_datos():
    if not os.path.exists("./db_vectorial"):
        if not os.path.exists("db_vectorial.zip"):
            file_id = '1gF8cHQYPoK8YX14KPJXr9-eir2RtF7iq' 
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, 'db_vectorial.zip', quiet=False)
        with zipfile.ZipFile("db_vectorial.zip", 'r') as zip_ref:
            zip_ref.extractall("./")
    
    persist_directory = "./db_vectorial"
    modelo_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return Chroma(persist_directory=persist_directory, embedding_function=modelo_embeddings)

vector_db = cargar_base_de_datos()

# ==========================================
# 3. GESTIÓN DE MEMORIA (EXPEDIENTE VIRTUAL)
# ==========================================
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []
if "resumen_hechos" not in st.session_state:
    st.session_state.resumen_hechos = "No hay hechos registrados aún."
if "textos_legales_vault" not in st.session_state:
    st.session_state.textos_legales_vault = []

def actualizar_resumen_incremental(nueva_pregunta, nueva_respuesta):
    """Mantiene un resumen compacto de la charla para no saturar con 'ruido'."""
    prompt_resumen = f"""Actualizá el RESUMEN DE HECHOS del caso con la nueva información. 
    Mantené solo datos duros, fechas y objetivos. Sé breve (viñetas).Es importante mantener en este resumen sobre todo las primeras y últimas partes de la conversación que suelen ser las mas importates
    
    RESUMEN ANTERIOR: {st.session_state.resumen_hechos}
    NUEVO INTERCAMBIO:
    Abogado: {nueva_pregunta}
    IA: {nueva_respuesta}
    
    NUEVO RESUMEN ACTUALIZADO:"""
    resumen = llm.invoke(prompt_resumen)
    st.session_state.resumen_hechos = resumen.content

# ==========================================
# 4. BARRA LATERAL (SELECTOR DE MODO)
# ==========================================
with st.sidebar:
    st.header("⚙️ Panel de Control")
    modo = st.radio(
        "Seleccioná el modo operativo:",
        ("📚 Búsqueda (HyDE)", "🧠 Estrategia y Análisis"),
        help="Búsqueda usa los libros con IA avanzada. Estrategia usa la memoria del caso para debatir."
    )
    st.markdown("---")
    if st.button("🗑️ Limpiar Memoria del Caso"):
        st.session_state.mensajes = []
        st.session_state.resumen_hechos = "No hay hechos registrados aún."
        st.session_state.textos_legales_vault = []
        st.rerun()
    
    with st.expander("📝 Expediente en Memoria"):
        st.write("**Hechos resumidos:**")
        st.info(st.session_state.resumen_hechos)
        st.write(f"**Textos legales en caja fuerte:** {len(st.session_state.textos_legales_vault)}")

# ==========================================
# 5. LÓGICA DE CHAT
# ==========================================
for mensaje in st.session_state.mensajes:
    with st.chat_message(mensaje["rol"]):
        st.markdown(mensaje["contenido"])

pregunta = st.chat_input("Planteá tu duda o caso aquí...")

if pregunta:
    with st.chat_message("user"):
        st.markdown(pregunta)
    st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})

    with st.chat_message("assistant"):
        # --- MODO 📚 BÚSQUEDA (HyDE) ---
        if modo == "📚 Búsqueda (HyDE)":
            with st.spinner("🔍 Generando hipótesis legal y buscando en biblioteca..."):
                # PASO HyDE: Generamos respuesta ficticia para mejorar el match
                hyde_prompt = f"Como abogado experto, escribí un párrafo técnico legal que responda a: {pregunta}"
                ficticio = llm.invoke(hyde_prompt).content
                
                # Buscamos en Chroma usando la respuesta ficticia
                docs = vector_db.similarity_search(ficticio, k=6)
                contexto_crudo = ""
                for d in docs:
                    fuente = f"[Rama: {d.metadata.get('rama')}, Archivo: {d.metadata.get('source')}]"
                    contexto_crudo += f"\n{fuente}\n{d.page_content}\n"
                    # Guardamos el texto crudo en la caja fuerte si no está
                    if d.page_content not in st.session_state.textos_legales_vault:
                        st.session_state.textos_legales_vault.append(f"{fuente}: {d.page_content}")

                prompt_rag = f"""Usá estos fragmentos EXACTOS para responder. 
                Contexto: {contexto_crudo}
                Pregunta: {pregunta}
                Respuesta Técnica:"""
                
                respuesta = llm.invoke(prompt_rag).content
                st.markdown(respuesta)
                with st.expander("👁️ Ver fuentes originales"):
                    st.text(contexto_crudo)

        # --- MODO 🧠 ESTRATEGIA (Memoria Dual) ---
        else:
            with st.spinner("🧠 Consultando expediente y analizando estrategia..."):
                # Construimos el expediente desde la memoria
                vault_text = "\n".join(st.session_state.textos_legales_vault[-5:]) # Últimos 5 textos crudos
                
                prompt_socio = f"""Actuá como mi socio legal senior. Analizá mi planteo usando el expediente actual.
                
                [RESUMEN DE HECHOS]: {st.session_state.resumen_hechos}
                
                [TEXTOS LEGALES RELEVANTES EN MEMORIA]:
                {vault_text}
                
                [NUEVA CONSULTA]: {pregunta}
                
                Tu análisis estratégico:"""
                
                respuesta = llm.invoke(prompt_socio).content
                st.markdown(respuesta)

        # Actualizamos la memoria incremental al final
        actualizar_resumen_incremental(pregunta, respuesta)
        st.session_state.mensajes.append({"rol": "assistant", "contenido": respuesta})
