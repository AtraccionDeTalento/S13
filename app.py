import streamlit as st
import numpy as np
from PIL import Image
import json
import os
from ultralytics import YOLO
from deepface import DeepFace

# ---------------------------------------------------------------
# configuracion
# ---------------------------------------------------------------
CARPETA_DATOS = "rostros_guardados"
RUTA_BASE_DATOS = os.path.join(CARPETA_DATOS, "base_datos.json")
MODELO_EMBEDDING = "Facenet"     # modelo de deepface para sacar el vector del rostro
UMBRAL_RECONOCIMIENTO = 0.50     # similitud minima para reconocer a alguien (0 a 1)
CLASE_PERSONA = 0                # en COCO la clase 0 es "persona"

os.makedirs(CARPETA_DATOS, exist_ok=True)


# ---------------------------------------------------------------
# modelo YOLO26 (se carga una sola vez gracias al cache de streamlit)
# ---------------------------------------------------------------
@st.cache_resource
def cargar_modelo_yolo():
    return YOLO("yolo26n.pt")


modelo_yolo = cargar_modelo_yolo()


# ---------------------------------------------------------------
# base de datos de rostros (un json simple: {"nombre": [vector...]})
# ---------------------------------------------------------------
def cargar_base_datos():
    if os.path.exists(RUTA_BASE_DATOS):
        with open(RUTA_BASE_DATOS, "r") as f:
            return json.load(f)
    return {}


def guardar_base_datos(base_datos):
    with open(RUTA_BASE_DATOS, "w") as f:
        json.dump(base_datos, f)


# ---------------------------------------------------------------
# formula de vectorizacion: convertir un rostro en numeros y comparar
# ---------------------------------------------------------------
def recortar_persona_principal(imagen_np):
    """Usa YOLO26 para encontrar a la persona mas grande en la imagen.
    Si no detecta a nadie, devuelve la imagen completa tal cual."""
    resultados = modelo_yolo(imagen_np, classes=[CLASE_PERSONA], verbose=False)[0]
    if len(resultados.boxes) == 0:
        return imagen_np
    caja = max(resultados.boxes, key=lambda c: float((c.xyxy[0][2] - c.xyxy[0][0]) * (c.xyxy[0][3] - c.xyxy[0][1])))
    x1, y1, x2, y2 = map(int, caja.xyxy[0])
    recorte = imagen_np[y1:y2, x1:x2]
    return recorte if recorte.size > 0 else imagen_np


def calcular_vector_rostro(imagen_np):
    """Recibe una imagen (foto o captura) y devuelve el vector de su rostro."""
    recorte = recortar_persona_principal(imagen_np)
    try:
        resultado = DeepFace.represent(
            img_path=recorte,
            model_name=MODELO_EMBEDDING,
            detector_backend="opencv",
            enforce_detection=False,
        )
        return np.array(resultado[0]["embedding"])
    except Exception:
        return None


def similitud_coseno(vector_a, vector_b):
    """Formula de vectorizacion: que tan parecidos son dos vectores.
    1.0 = identicos, 0.0 = sin relacion, negativo = opuestos."""
    a = np.array(vector_a)
    b = np.array(vector_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def buscar_persona(vector_nuevo, base_datos):
    """Compara el vector nuevo contra la base de datos y devuelve el nombre mas parecido."""
    mejor_nombre = "Desconocido"
    mejor_similitud = 0.0
    for nombre, vector_guardado in base_datos.items():
        similitud = similitud_coseno(vector_nuevo, vector_guardado)
        if similitud > mejor_similitud:
            mejor_similitud = similitud
            mejor_nombre = nombre
    if mejor_similitud >= UMBRAL_RECONOCIMIENTO:
        return mejor_nombre, mejor_similitud
    return "Desconocido", mejor_similitud


# ---------------------------------------------------------------
# interfaz
# ---------------------------------------------------------------
st.set_page_config(page_title="Reconocimiento Facial", page_icon="🙂")
st.title("Reconocimiento Facial")
st.caption("YOLO26 ubica a la persona en la imagen, DeepFace calcula su vector facial y lo compara.")

base_datos_rostros = cargar_base_datos()

tab_registrar, tab_reconocer = st.tabs(["Registrar rostro", "Reconocer imagen"])

with tab_registrar:
    st.subheader("Captura tu rostro con la camara")
    nombre = st.text_input("Tu nombre", key="nombre_registro")
    foto = st.camera_input("Toma una foto", key="camara_registro")

    if st.button("Guardar rostro"):
        if not nombre:
            st.error("Escribe un nombre antes de guardar")
        elif not foto:
            st.error("Toma una foto primero")
        else:
            imagen = np.array(Image.open(foto).convert("RGB"))
            vector = calcular_vector_rostro(imagen)
            if vector is None:
                st.error("No se pudo distinguir el rostro, intenta de nuevo")
            else:
                base_datos_rostros[nombre] = vector.tolist()
                guardar_base_datos(base_datos_rostros)
                st.success(f"{nombre} fue registrado correctamente")

    if base_datos_rostros:
        st.write("**Rostros guardados:**", ", ".join(sorted(base_datos_rostros.keys())))
    else:
        st.write("Todavia no hay nadie registrado")

with tab_reconocer:
    st.subheader("Sube una imagen para reconocer")
    imagen_subida = st.file_uploader("Imagen", type=["jpg", "jpeg", "png"], key="imagen_reconocer")

    if imagen_subida:
        imagen = np.array(Image.open(imagen_subida).convert("RGB"))
        st.image(imagen, caption="Imagen subida", width=300)

        if st.button("Reconocer"):
            if not base_datos_rostros:
                st.warning("Todavia no hay nadie registrado, ve a la pestana Registrar rostro")
            else:
                vector = calcular_vector_rostro(imagen)
                if vector is None:
                    st.error("No se encontro un rostro en la imagen")
                else:
                    nombre_encontrado, similitud = buscar_persona(vector, base_datos_rostros)
                    if nombre_encontrado == "Desconocido":
                        st.error(f"No reconocido (similitud maxima: {similitud:.2f})")
                    else:
                        st.success(f"Es **{nombre_encontrado}** (similitud: {similitud:.2f})")
