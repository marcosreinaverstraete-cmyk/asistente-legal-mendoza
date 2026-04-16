import os
import warnings
import zipfile
import gdown
import re
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

# Inicializamos los dos motores (Flash para buscar rápido, Pro para pensar la estrategia)
llm_flash = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
llm_pro = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2)

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
if "mensajes_iniciales" not in st.session_state:
    st.session_state.mensajes_iniciales = [] # Para guardar el planteo original
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []
if "resumen_hechos" not in st.session_state:
    st.session_state.resumen_hechos = "No hay hechos registrados aún."
if "textos_legales_vault" not in st.session_state:
    st.session_state.textos_legales_vault = []

def actualizar_resumen_incremental(nueva_pregunta, nueva_respuesta):
    """Extrae solo hechos y objetivos, ignorando leyes."""
    prompt_resumen = f"""Tu única tarea es extraer LOS HECHOS DEL CASO y EL OBJETIVO DEL CLIENTE.
    PROHIBIDO resumir leyes, doctrina o jurisprudencia (eso se guarda en otro lado).
    Si la charla fue solo una consulta teórica sin un cliente real, mantené el resumen anterior sin sumar nada.
    
    RESUMEN ANTERIOR: {st.session_state.resumen_hechos}
    
    NUEVO INTERCAMBIO:
    Abogado: {nueva_pregunta}
    IA: {nueva_respuesta}
    
    HECHOS Y OBJETIVOS ACTUALIZADOS (en viñetas):"""
    
    # Usamos Flash porque es una tarea mecánica y rápida
    resumen = llm_flash.invoke(prompt_resumen)
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
        st.session_state.mensajes_iniciales = []
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
        # --- MODO 📚 BÚSQUEDA (HyDE con FLASH) ---
        if modo == "📚 Búsqueda (HyDE)":
            with st.spinner("🔍 Generando hipótesis legal y buscando en biblioteca..."):
                hyde_prompt = f"Como abogado experto, escribí un párrafo técnico legal que responda a: {pregunta}"
                ficticio = llm_flash.invoke(hyde_prompt).content
                
                docs = vector_db.similarity_search(ficticio, k=6)
                
                contexto_con_indices = ""
                for i, d in enumerate(docs):
                    fuente = f"[Rama: {d.metadata.get('rama')}, Archivo: {d.metadata.get('source')}]"
                    contexto_con_indices += f"\n--- FRAGMENTO {i+1} ---\n{fuente}\n{d.page_content}\n"

                prompt_rag = f"""
                Analizá estos fragmentos legales para responder la pregunta del abogado.
                
                [FRAGMENTOS DISPONIBLES]:
                {contexto_con_indices}
                
                [PREGUNTA ACTUAL A RESPONDER]:
                {pregunta}
                
                INSTRUCCIONES ESTRICTAS:
                1. Respondé ÚNICAMENTE a la [PREGUNTA ACTUAL].
                2. Basate EXCLUSIVAMENTE en los fragmentos. Si no está la info, decí "No tengo doctrina sobre esto en mis libros".
                3. PROHIBIDO decir "En el fragmento 1...". Tenés que citar el AUTOR o la LEY usando el nombre del archivo (Ej: "Según el archivo Borda_Obligaciones.pdf...").
                4. La última línea de tu respuesta debe ser EXACTAMENTE: RELEVANTES: [números separados por comas]
                
                RESPUESTA:"""
                
                full_res = llm_flash.invoke(prompt_rag).content
                
                match = re.search(r"RELEVANTES:\s*\[?([\d\s,]+)\]?", full_res)
                
                if match:
                    respuesta = full_res[:match.start()].strip()
                    indices_str = match.group(1)
                    indices = [int(x.strip()) for x in indices_str.split(",") if x.strip().isdigit()]
                    
                    for idx in indices:
                        if 1 <= idx <= len(docs):
                            d = docs[idx-1]
                            fuente = f"[Rama: {d.metadata.get('rama')}, Archivo: {d.metadata.get('source')}]"
                            if d.page_content not in st.session_state.textos_legales_vault:
                                st.session_state.textos_legales_vault.append(f"{fuente}: {d.page_content}")
                else:
                    respuesta = full_res

                st.markdown(respuesta)
                with st.expander("👁️ Ver los 6 fragmentos analizados por la IA"):
                    st.text(contexto_con_indices)

        # --- MODO 🧠 ESTRATEGIA (Memoria Dual con PRO) ---
        else:
            with st.spinner("🧠 El Socio Senior está analizando el expediente completo..."):
                contexto_inicial = "\n".join([f"{m['rol'].upper()}: {m['contenido']}" for m in st.session_state.mensajes_iniciales])
                mensajes_recientes = st.session_state.mensajes[-6:]
                contexto_reciente = "\n".join([f"{m['rol'].upper()}: {m['contenido']}" for m in mensajes_recientes])
                vault_text = "\n".join(st.session_state.textos_legales_vault)
                
                prompt_socio = f"""
                Actuá como mi socio legal senior. 
                
                [PLANTEO INICIAL]: {contexto_inicial}
                [HILO RECIENTE]: {contexto_reciente}
                [DOCTRINA Y LEYES EN EXPEDIENTE]: {vault_text}
                [PREGUNTA ACTUAL]: {pregunta}
                
                REGLAS DE ORO ESTRICTAS:
                1. Respuestas DIRECTAS, al grano y en formato viñetas. Eliminá cualquier introducción o conclusión de cortesía.
                2. Cruzá los hechos con la [DOCTRINA Y LEYES EN EXPEDIENTE].
                3. Si la doctrina guardada NO sirve para responder la pregunta, tu única respuesta debe ser: "No hay información legal en el expediente para analizar esto." 
                4. PROHIBIDO irte por las ramas, divagar o usar conocimientos externos no guardados. Si no está en el expediente, no existe.
                
                ANÁLISIS ESTRATÉGICO CONCISO:"""
                
                respuesta = llm_pro.invoke(prompt_socio).content
                st.markdown(respuesta)

        # --- ACTUALIZACIÓN DE MEMORIA FINAL ---
        st.session_state.mensajes.append({"rol": "assistant", "contenido": respuesta})
        
        # Guardamos los primeros 2 intercambios (4 mensajes: user, asst, user, asst)
        if len(st.session_state.mensajes) <= 4:
            st.session_state.mensajes_iniciales = st.session_state.mensajes.copy()
            
        actualizar_resumen_incremental(pregunta, respuesta)
