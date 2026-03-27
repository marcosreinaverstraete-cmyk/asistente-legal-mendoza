import os
import shutil
from langchain_community.document_loaders import PyPDFLoader  # <-- Cambiamos al loader individual
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

ruta_raiz = "."
persist_directory = "./db_vectorial"

ramas_objetivo = ["Administrativo", "Ambiental", "Minero", "Recursos_Hidricos"]
todos_los_documentos = []

print("🕵️‍♂️ Iniciando escaneo BLINDADO de carpetas locales...")

for rama in ramas_objetivo:
    ruta_carpeta = os.path.join(ruta_raiz, rama)
    
    if os.path.exists(ruta_carpeta):
        print(f"\n📚 Entrando a la rama: [{rama}]...")
        archivos_pdf = [f for f in os.listdir(ruta_carpeta) if f.lower().endswith('.pdf')]
        paginas_rama = 0
        
        # Leemos archivo por archivo en lugar de toda la carpeta de golpe
        for archivo in archivos_pdf:
            ruta_pdf = os.path.join(ruta_carpeta, archivo)
            try:
                loader = PyPDFLoader(ruta_pdf)
                docs = loader.load()
                
                for doc in docs:
                    doc.metadata["rama"] = rama
                    doc.metadata["source"] = archivo

                todos_los_documentos.extend(docs)
                paginas_rama += len(docs)
                
            except Exception as e:
                # ¡EL ESCUDO! Si el PDF está roto, no explota, solo avisa.
                print(f"   ❌ ALERTA: El archivo '{archivo}' está roto o protegido. Lo salteamos.")
        
        print(f"   ✅ Se cargaron {paginas_rama} páginas válidas de {rama}.")
    else:
        print(f"   ⚠️ La carpeta {rama} no existe. Saltando...")

if not todos_los_documentos:
    print("🚨 ¡Error! No se cargaron documentos válidos.")
    exit()

print("\n✂️ Cortando los textos...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
fragmentos = splitter.split_documents(todos_los_documentos)

print("🧠 Cargando modelo matemático...")
modelo_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)

print("💾 Guardando vectores en tu disco duro (ChromaDB)... esto puede demorar.")
vector_db = Chroma.from_documents(
    documents=fragmentos,
    embedding=modelo_embeddings,
    persist_directory=persist_directory
)

print(f"\n🎉 ¡Base de datos creada con éxito en la carpeta {persist_directory}!")