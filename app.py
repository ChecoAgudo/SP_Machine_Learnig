"""
app.py - Interfaz Streamlit
Diagnostico de enfermedades foliares en tomate (PlantVillage) con una CNN en PyTorch.

Ejecutar desde la raiz del proyecto:
    streamlit run app.py
"""

import json

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

from src import preprocesamiento as prep
from src import modelo_dl as dl

st.set_page_config(page_title="Diagnóstico foliar - Tomate", layout="wide")

# ===========================================================================
# TEXTOS DE AYUDA (burbujas "?"). Editalos aqui si quieres cambiar una explicacion.
# ===========================================================================
AYUDA = {
    "secciones": "Dataset: revisa las imágenes y cómo se preparan. Aprendizaje profundo: entrena la red neuronal (CNN) y diagnostica hojas. Aprendizaje no supervisado: agrupa hojas parecidas sin usar etiquetas.",
    "resolucion": "Tamaño (en píxeles) al que se reduce cada imagen antes de entrar a la red.\n\n**128**: rápido, con menos detalle.\n\n**224**: más detalle (manchas pequeñas) pero unas 3 veces más lento.\n\nSin tarjeta gráfica (GPU), usa 128.",
    "max_por_clase": "Máximo de imágenes que se usan de cada clase.\n\n**Valor bajo (100)**: entrena rápido, pero el modelo ve pocos ejemplos y rinde peor.\n\n**Valor alto (2000)**: usa casi todo el dataset y rinde mejor, pero tarda mucho más.\n\n600 es un buen equilibrio en un computador sin GPU.",
    "epocas": "Una época = el modelo ve todas las imágenes de entrenamiento una vez.\n\n**Pocas (1-3)**: aprende poco (subajuste).\n\n**Muchas (20-30)**: aprende más, pero puede empezar a memorizar las fotos (sobreajuste) y tarda más.\n\nPunto de partida recomendado: 8 a 10.",
    "lr": "Learning rate (tasa de aprendizaje): qué tan grandes son los pasos con que el modelo corrige sus errores.\n\n**Muy bajo (0.0001)**: aprende lento pero seguro; necesita más épocas.\n\n**Muy alto (0.003)**: aprende rápido, pero puede 'pasarse' y volverse inestable.\n\n0.001 es un valor estándar con el optimizador Adam.",
    "batch": "Cuántas imágenes procesa el modelo antes de actualizar sus pesos.\n\n**Bajo (16)**: usa poca memoria; el aprendizaje es más 'ruidoso'.\n\n**Alto (64)**: aprendizaje más estable, pero usa más memoria RAM.\n\n32 es un valor equilibrado.",
    "dispositivo": "Procesador con el que se entrena. **cpu** = procesador común (más lento). **cuda** = tarjeta gráfica NVIDIA (mucho más rápida para redes neuronales).",
    "balance": "Cuántas veces más imágenes tiene la clase más grande que la más pequeña. 1.0 significa que todas tienen lo mismo. Si es mayor a 1.5 las clases están desbalanceadas; por eso se usa la división estratificada y las métricas 'macro'.",
    "distribucion": "Cantidad de imágenes que hay en el dataset por cada clase (estado de la hoja).",
    "division": "Las imágenes se separan en tres grupos:\n\n**Entrenamiento (70%)**: con estas imágenes el modelo aprende.\n\n**Validación (15%)**: sirve para vigilar el aprendizaje y elegir la mejor época.\n\n**Prueba (15%)**: imágenes que el modelo nunca vio; dan la nota final honesta.\n\nLa división es **estratificada**: cada grupo conserva la misma proporción de cada clase que el dataset completo. Así ningún grupo se queda sin ejemplos de una enfermedad y la evaluación es justa.",
    "preproceso": "Antes de entrar a la red, cada imagen se redimensiona al mismo tamaño y sus píxeles se pasan de 0-255 a 0-1 (normalización).\n\nAdemás, durante el **entrenamiento** se aplica *data augmentation*: se modifican al azar las imágenes (giros, rotación, cambios de brillo y contraste) para que el modelo vea más variedad y aprenda a reconocer la enfermedad en distintas posiciones y luces, en lugar de memorizar fotos. Nunca se aplica a validación ni a prueba.",
    "cnn": "CNN (Red Neuronal Convolucional): tipo de red neuronal especializada en imágenes. Sus capas 'convolucionales' detectan patrones visuales (bordes, manchas, texturas) y las capas siguientes los combinan para decidir la clase de la hoja.",
    "pytorch": "PyTorch: librería de Python que se usa para construir y entrenar redes neuronales.",
    "parametros": "Cantidad de números internos (pesos) que la red ajusta mientras aprende. Más parámetros = red más grande y más lenta.",
    "capas": "Cantidad de bloques convolucionales: cada uno detecta patrones más complejos que el anterior.",
    "perdida": "CrossEntropy (entropía cruzada): función que mide qué tan equivocadas son las predicciones. Castiga más cuando el modelo está muy seguro de una respuesta incorrecta. El entrenamiento intenta hacerla bajar.",
    "accuracy": "Porcentaje de hojas clasificadas correctamente sobre el total. Ejemplo: 0.95 = acertó 95 de cada 100. Puede engañar si las clases están desbalanceadas.",
    "precision": "De todas las hojas que el modelo marcó como una clase, cuántas realmente lo eran. Precision alta = pocas falsas alarmas.",
    "recall": "De todas las hojas que realmente eran de una clase, cuántas logró encontrar. Recall alto = casi no se le escapan casos enfermos.",
    "f1": "Promedio equilibrado entre Precision y Recall. Resume ambos en un solo número; solo es alto si los dos son buenos.",
    "macro": "Promedio macro: la métrica se calcula por separado para cada clase y luego se promedia, de modo que cada clase pesa igual aunque tenga menos imágenes.",
    "support": "Cantidad de imágenes reales de esa clase en el conjunto de prueba.",
    "curvas": "**Loss (pérdida)**: mide cuánto se equivoca el modelo; debe BAJAR con las épocas.\n\n**Accuracy**: porcentaje de aciertos; debe SUBIR.\n\nRojo = entrenamiento, verde = validación. Si el entrenamiento mejora pero la validación se estanca o empeora, hay sobreajuste.",
    "matriz": "Tabla que cruza lo REAL (filas) con lo PREDICHO (columnas).\n\nLa diagonal (celdas más oscuras) son los **aciertos**. Fuera de la diagonal están los **errores**: por ejemplo, la celda fila 'Tizón temprano' / columna 'Tizón tardío' cuenta hojas de tizón temprano que el modelo confundió con tizón tardío.",
    "sobreajuste": "Sobreajuste = el modelo memoriza las fotos de entrenamiento y falla con fotos nuevas. Se detecta comparando la accuracy de entrenamiento con la de validación en la mejor época: si la de entrenamiento es mucho mayor, hay sobreajuste.",
    "confianza": "Probabilidad que el modelo asigna a su respuesta. No es una garantía: un modelo puede estar muy seguro y equivocarse, sobre todo con imágenes distintas a las que vio al entrenar.",
    "ns": "**K-Means**: algoritmo que agrupa imágenes parecidas en 'clusters' sin usar las etiquetas.\n\n**PCA**: técnica que reduce muchas características a 2 o 3 para poder graficarlas.",
}

# ===========================================================================
# INFORMACION DE CADA CLASE (orientativa)
# ===========================================================================
COLOR_CLASE = {
    "Tomato_healthy": "#10B981",
    "Tomato_Early_blight": "#F59E0B",
    "Tomato_Late_blight": "#E11D48",
    "Tomato_Bacterial_spot": "#F97316",
}
INFO_CLASE = {
    "Tomato_healthy": {
        "causa": "No aplica.",
        "sintomas": "Color verde uniforme, sin manchas, halos ni deformaciones.",
        "recomendacion": "Mantener el monitoreo periódico, regar sin mojar el follaje y asegurar buena ventilación entre plantas.",
    },
    "Tomato_Early_blight": {
        "causa": "Hongo <i>Alternaria solani</i>.",
        "sintomas": "Manchas marrones con anillos concéntricos (como un blanco de tiro), a menudo con halo amarillo. Suelen aparecer primero en las hojas más viejas, las de abajo.",
        "recomendacion": "Retirar y eliminar las hojas afectadas, rotar cultivos, evitar mojar el follaje al regar y consultar a un técnico agrónomo sobre el tratamiento.",
    },
    "Tomato_Late_blight": {
        "causa": "Oomiceto <i>Phytophthora infestans</i> (organismo parecido a un hongo).",
        "sintomas": "Manchas grandes e irregulares, de aspecto aceitoso o empapado, que se oscurecen rápido. Con humedad puede verse un moho blanquecino por el envés. Avanza muy rápido con clima fresco y húmedo.",
        "recomendacion": "Actuar de inmediato: aislar o retirar plantas afectadas, mejorar la ventilación, reducir la humedad y consultar a un técnico agrónomo porque se propaga rápido.",
    },
    "Tomato_Bacterial_spot": {
        "causa": "Bacterias del género <i>Xanthomonas</i>.",
        "sintomas": "Manchas pequeñas, oscuras y de aspecto empapado en las hojas, a veces con halo amarillo. También pueden aparecer en los frutos con textura rugosa.",
        "recomendacion": "Usar semillas sanas, evitar el riego por aspersión sobre las hojas, no manipular plantas mojadas y consultar a un técnico agrónomo.",
    },
}

RUTA_RES = dl.RUTA_MODELO.parent / "resultados_cnn.json"

# ===========================================================================
# ESTILO
# ===========================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500&family=Space+Grotesk:wght@600;700&family=JetBrains+Mono&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }
code, pre { font-family: 'JetBrains Mono', monospace !important; }
div[data-testid="stHeading"] h2 { border-bottom: 2px solid #10B981; padding-bottom: 6px; }
.hero { display:flex; align-items:center; gap:18px; padding:20px 24px; margin-bottom:18px;
        background:linear-gradient(120deg,#18181B 0%,#1d1418 55%,#2b0f17 100%);
        border:1px solid #27272A; border-left:6px solid #E11D48; border-radius:14px; }
.hero-title { font-family:'Space Grotesk',sans-serif; font-size:2rem; font-weight:700; color:#FAFAFA; line-height:1.15; }
.hero-title span { color:#E11D48; }
.hero-sub { color:#A1A1AA; margin:4px 0 10px 0; }
.badge { display:inline-block; padding:3px 12px; margin:0 6px 4px 0; border:1px solid #27272A;
         border-radius:999px; background:#0D0D0D; color:#A1A1AA; font-size:0.8rem; cursor:help; }
.badge b { color:#10B981; }
.side-logo { display:flex; align-items:center; gap:10px; margin-bottom:6px; }
.side-logo div { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.05rem; color:#FAFAFA; line-height:1.1; }
.side-logo small { color:#10B981; font-weight:500; font-size:0.75rem; }
div[data-testid="stMetric"] { background:#18181B; border:1px solid #27272A; border-left:4px solid #10B981;
                              border-radius:10px; padding:14px 16px; }
.tarjeta { background:#18181B; border:1px solid #27272A; border-radius:10px; padding:16px 20px; margin-bottom:12px; }
.tarjeta h4 { margin:0 0 6px 0; font-family:'Space Grotesk',sans-serif; color:#FAFAFA; }
.tarjeta p { margin:4px 0; color:#D4D4D8; font-size:0.95rem; }
.tarjeta .nota { color:#A1A1AA; font-size:0.8rem; margin-top:10px; }
.barra-fondo { background:#27272A; border-radius:6px; height:12px; width:100%; }
.barra { height:12px; border-radius:6px; }
.fila-barra { display:flex; justify-content:space-between; font-size:0.9rem; margin:10px 0 4px 0; color:#FAFAFA; }
.chip { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:8px; }
.mono { font-family:'JetBrains Mono',monospace; color:#A1A1AA; }
.nombre-clase { font-family:'Space Grotesk',sans-serif; font-weight:600; margin-bottom:4px; }
</style>
""", unsafe_allow_html=True)

LOGO_SVG = ("<svg width='{s}' height='{s}' viewBox='0 0 64 64' xmlns='http://www.w3.org/2000/svg'>"
            "<circle cx='32' cy='38' r='22' fill='#E11D48'/>"
            "<ellipse cx='24' cy='30' rx='6' ry='4' fill='#ffffff' opacity='0.18' transform='rotate(-30 24 30)'/>"
            "<path d='M32 19C27 19 22 16 19 12C25 11 29 14 32 17C35 14 39 11 45 12C42 16 37 19 32 19Z' fill='#10B981'/>"
            "<rect x='30' y='7' width='4' height='11' rx='2' fill='#059669'/></svg>")


# ===========================================================================
# FUNCIONES AUXILIARES
# ===========================================================================
def mostrar_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


def barra_html(texto, pct, color, derecha=None):
    derecha = derecha if derecha is not None else f"{pct:.1f}%"
    return (f"<div class='fila-barra'><span><span class='chip' style='background:{color}'></span>{texto}</span>"
            f"<span class='mono'>{derecha}</span></div>"
            f"<div class='barra-fondo'><div class='barra' style='width:{min(pct, 100):.1f}%;background:{color}'></div></div>")


def guardar_resultados(res):
    RUTA_RES.parent.mkdir(parents=True, exist_ok=True)
    with open(RUTA_RES, "w", encoding="utf-8") as f:
        json.dump(res, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))


def cargar_resultados():
    if not RUTA_RES.exists():
        return None
    try:
        with open(RUTA_RES, encoding="utf-8") as f:
            res = json.load(f)
        res["met"]["matriz"] = np.array(res["met"]["matriz"])
        return res
    except Exception:
        return None


# ===========================================================================
# ENCABEZADO
# ===========================================================================
st.markdown(
    "<div class='hero'>" + LOGO_SVG.format(s=64) +
    "<div><div class='hero-title'>Diagnóstico de <span>enfermedades foliares</span> en tomate</div>"
    "<div class='hero-sub'>Clasificación de hojas con visión por computadora</div>"
    "<span class='badge'><b>Área:</b> Visión por computadora</span>"
    f"<span class='badge' title='{AYUDA['cnn'].replace(chr(39), '')}'><b>Modelo:</b> CNN</span>"
    f"<span class='badge' title='{AYUDA['pytorch']}'><b>Librería:</b> PyTorch</span>"
    "<span class='badge' title='Imágenes de hojas de tomate etiquetadas por especialistas (Kaggle).'><b>Dataset:</b> PlantVillage</span>"
    "</div></div>", unsafe_allow_html=True)

# ===========================================================================
# BARRA LATERAL
# ===========================================================================
SEC_DATA = "Dataset y preprocesamiento"
SEC_DL = "Aprendizaje profundo (CNN)"
SEC_NS = "Aprendizaje no supervisado"

with st.sidebar:
    st.markdown("<div class='side-logo'>" + LOGO_SVG.format(s=34) +
                "<div>Visión Foliar<br><small>Tomate</small></div></div>", unsafe_allow_html=True)
    seccion = st.radio("Secciones", [SEC_DATA, SEC_DL, SEC_NS], help=AYUDA["secciones"])
    st.markdown("---")
    with st.expander("Parámetros del modelo", expanded=(seccion != SEC_NS)):
        img_size = st.selectbox("Resolución de entrada (px)", [128, 224], index=0, help=AYUDA["resolucion"])
        max_por_clase = st.slider("Imágenes por clase (máximo)", 100, 2000, 600, 100, help=AYUDA["max_por_clase"])
        epocas = st.slider("Épocas", 1, 30, 8, help=AYUDA["epocas"])
        lr = st.select_slider("Learning rate", options=[1e-4, 3e-4, 1e-3, 3e-3], value=1e-3,
                              format_func=lambda v: f"{v:g}", help=AYUDA["lr"])
        batch = st.select_slider("Batch size", options=[16, 32, 64], value=32, help=AYUDA["batch"])
        st.caption(f"Dispositivo: **{dl.obtener_dispositivo()}**", help=AYUDA["dispositivo"])

hay_dataset = prep.dataset_disponible()

# ===========================================================================
# SECCION 1: DATASET Y PREPROCESAMIENTO
# ===========================================================================
if seccion == SEC_DATA:
    st.header("Dataset y preprocesamiento")

    if not hay_dataset:
        st.error("No se encontró el dataset en `data/PlantVillage_Tomate/`.")
        st.code("python -m src.descargar_dataset", language="bash")
        st.stop()

    conteos = prep.conteo_por_clase()
    total = sum(conteos.values())
    desbalance = max(conteos.values()) / max(1, min(conteos.values()))
    c = st.columns(5)
    c[0].metric("Imágenes totales", f"{total:,}", help="Cantidad total de fotos de hojas de tomate disponibles en las 4 clases.")
    c[1].metric("Clases", len(prep.CLASES), help="Estados posibles de una hoja: sana y tres enfermedades.")
    c[2].metric("Clase más grande", f"{max(conteos.values()):,}", help="Cantidad de imágenes de la clase que más tiene.")
    c[3].metric("Clase más pequeña", f"{min(conteos.values()):,}", help="Cantidad de imágenes de la clase que menos tiene.")
    c[4].metric("Balance de clases", f"{desbalance:.2f}x", help=AYUDA["balance"])

    izq, der = st.columns(2)
    with izq:
        st.subheader("Distribución por clase", help=AYUDA["distribucion"])
        mayor = max(conteos.values())
        html = "".join(barra_html(prep.NOMBRES_LEGIBLES[k], 100 * v / mayor, COLOR_CLASE[k], f"{v:,}")
                       for k, v in conteos.items())
        st.markdown(html, unsafe_allow_html=True)
    with der:
        st.subheader("División de los datos", help=AYUDA["division"])
        rutas, etiquetas = prep.listar_imagenes(max_por_clase=max_por_clase)
        splits = prep.dividir_estratificado(rutas, etiquetas)
        filas = []
        for nombre, (r, y) in splits.items():
            fila = {"Conjunto": {"train": "Entrenamiento (70%)", "val": "Validación (15%)", "test": "Prueba (15%)"}[nombre],
                    "Total": len(r)}
            for i, k in enumerate(prep.CLASES):
                fila[prep.NOMBRES_LEGIBLES[k]] = int(np.sum(np.array(y) == i))
            filas.append(fila)
        st.dataframe(pd.DataFrame(filas), hide_index=True)
        st.caption(f"Se usan hasta {max_por_clase} imágenes por clase (se cambia en la barra lateral).")

    st.subheader("Cómo ve el modelo las imágenes", help=AYUDA["preproceso"])
    st.caption(f"Cada imagen se reduce a {img_size}×{img_size} px. Abajo, un ejemplo de las variaciones "
               "aleatorias que se aplican solo durante el entrenamiento.")
    st.button("Generar otras variaciones")
    cols = st.columns(4)
    for col, clase in zip(cols, prep.CLASES):
        archivo = sorted((prep.RUTA_DATASET / clase).glob("*"))[0]
        img = Image.open(archivo).convert("RGB")
        with col:
            st.markdown(f"<div class='nombre-clase'><span class='chip' style='background:{COLOR_CLASE[clase]}'></span>"
                        f"{prep.NOMBRES_LEGIBLES[clase]}</div>", unsafe_allow_html=True)
            st.image(img, width=200, caption="Original")
            st.image(prep.imagen_con_augmentation(img, img_size), width=200, caption="Variación de entrenamiento")

# ===========================================================================
# SECCION 2: APRENDIZAJE PROFUNDO
# ===========================================================================
elif seccion == SEC_DL:
    st.header("Aprendizaje profundo: CNN", help=AYUDA["cnn"])
    st.caption("Red neuronal convolucional (con PyTorch) que clasifica una hoja de tomate en 4 estados.",
               help=AYUDA["pytorch"])
    tab_entrena, tab_valida, tab_diag = st.tabs(["Entrenamiento", "Validación", "Diagnóstico de hoja"])

    # ------------------------------ ENTRENAMIENTO -------------------------
    with tab_entrena:
        modelo_tmp = dl.CNNFoliar(n_clases=len(prep.CLASES))
        k = st.columns(4)
        k[0].metric("Tipo de modelo", "CNN", help=AYUDA["cnn"])
        k[1].metric("Capas convolucionales", 4, help=AYUDA["capas"])
        k[2].metric("Parámetros", f"{dl.contar_parametros(modelo_tmp):,}", help=AYUDA["parametros"])
        k[3].metric("Clases de salida", len(prep.CLASES), help="Las 4 respuestas posibles: hoja sana, tizón temprano, tizón tardío o mancha bacteriana.")

        with st.expander("Ver estructura de la red"):
            st.markdown(
                "| Capa | Qué hace |\n|---|---|\n"
                "| Conv2D (×4: 32, 64, 128 y 128 filtros) | Detecta patrones visuales: primero bordes y colores, luego manchas y texturas |\n"
                "| BatchNorm | Estabiliza y acelera el aprendizaje |\n"
                "| ReLU | Deja pasar solo las señales positivas; permite aprender patrones complejos |\n"
                "| MaxPool | Reduce el tamaño de la imagen quedándose con lo más importante |\n"
                "| Flatten + Dense (256) | Combina todo lo detectado en una sola decisión |\n"
                "| Dropout (0.4) | Apaga neuronas al azar al entrenar para evitar que memorice |\n"
                "| Dense (4) + Softmax | Da una probabilidad para cada una de las 4 clases |\n")
            st.caption("Función de pérdida: CrossEntropy. Optimizador: Adam.", help=AYUDA["perdida"])

        st.caption(f"Configuración actual: {img_size} px · {epocas} épocas · learning rate {lr:g} · "
                   f"batch {batch} · hasta {max_por_clase} imágenes por clase (se cambia en la barra lateral)")

        if not hay_dataset:
            st.error("Primero descarga el dataset: `python -m src.descargar_dataset`")
        elif st.button("Entrenar CNN", type="primary"):
            with st.spinner("Preparando datos..."):
                loaders, splits = prep.crear_dataloaders(
                    img_size=img_size, batch_size=batch, max_por_clase=max_por_clase)
            barra = st.progress(0.0, text="Iniciando entrenamiento...")
            grafica = st.empty()

            def progreso(e, total, h):
                barra.progress(e / total, text=f"Época {e}/{total} | loss {h['loss'][-1]:.3f} | "
                                               f"accuracy validación {h['val_acc'][-1]:.3f}")
                fig = dl.graficar_curvas(h)
                grafica.pyplot(fig)
                plt.close(fig)

            modelo = dl.CNNFoliar(n_clases=len(prep.CLASES))
            hist = dl.entrenar(modelo, loaders, epocas=epocas, lr=lr, callback=progreso,
                               ruta_guardado=dl.RUTA_MODELO, clases=prep.CLASES, img_size=img_size)
            y_real, y_pred, _ = dl.evaluar(modelo, loaders["test"])
            res = {
                "hist": hist,
                "met": dl.calcular_metricas(y_real, y_pred, prep.CLASES),
                "n_test": int(len(y_real)),
                "cfg": f"{img_size} px · {epocas} épocas · lr {lr:g} · batch {batch} · {max_por_clase} imágenes/clase",
            }
            st.session_state["resultado"] = res
            guardar_resultados(res)
            st.cache_resource.clear()
            barra.empty()
            st.success(f"Entrenamiento terminado en {hist['tiempo_s']:.0f} s. Modelo guardado. "
                       "Revisa la pestaña Validación.")

    # ------------------------------ VALIDACION ----------------------------
    with tab_valida:
        res = st.session_state.get("resultado") or cargar_resultados()
        if not res:
            st.info("Aún no hay resultados. Entrena el modelo en la pestaña Entrenamiento.")
        else:
            st.session_state["resultado"] = res
            m, hist = res["met"], res["hist"]
            st.caption(f"Configuración usada: {res['cfg']}. Resultados sobre {res['n_test']} imágenes de prueba "
                       "que el modelo nunca vio.")
            k = st.columns(4)
            k[0].metric("Accuracy", f"{m['Accuracy']:.3f}", help=AYUDA["accuracy"])
            k[1].metric("Precision", f"{m['Precision']:.3f}", help=AYUDA["precision"])
            k[2].metric("Recall", f"{m['Recall']:.3f}", help=AYUDA["recall"])
            k[3].metric("F1-Score", f"{m['F1-Score']:.3f}", help=AYUDA["f1"])
            st.caption("Precision, Recall y F1 son promedio macro.", help=AYUDA["macro"])

            g1, g2 = st.columns([1.3, 1])
            with g1:
                st.subheader("Curvas de aprendizaje", help=AYUDA["curvas"])
                mostrar_fig(dl.graficar_curvas(hist))
            with g2:
                st.subheader("Matriz de confusión", help=AYUDA["matriz"])
                etiquetas = [prep.NOMBRES_LEGIBLES[c] for c in prep.CLASES]
                mostrar_fig(dl.graficar_matriz_confusion(m["matriz"], etiquetas))

            st.subheader("Reporte por clase")
            rep = pd.DataFrame(m["reporte"]).T.loc[prep.CLASES, ["precision", "recall", "f1-score", "support"]]
            rep.index = [prep.NOMBRES_LEGIBLES[c] for c in rep.index]
            rep.columns = ["Precision", "Recall", "F1-Score", "Support"]
            rep["Support"] = rep["Support"].astype(int)
            st.dataframe(rep.round(3), column_config={
                "Precision": st.column_config.NumberColumn("Precision", help=AYUDA["precision"]),
                "Recall": st.column_config.NumberColumn("Recall", help=AYUDA["recall"]),
                "F1-Score": st.column_config.NumberColumn("F1-Score", help=AYUDA["f1"]),
                "Support": st.column_config.NumberColumn("Support", help=AYUDA["support"]),
            })

            
    # ------------------------------ DIAGNOSTICO ---------------------------
    with tab_diag:
        @st.cache_resource
        def _cargar():
            return dl.cargar_modelo()

        cargado = _cargar()
        if cargado is None:
            st.warning("Todavía no hay un modelo entrenado. Entrénalo en la pestaña Entrenamiento.")
        else:
            modelo, clases, size_modelo = cargado
            izq, der = st.columns(2)
            with izq:
                st.subheader("Hoja a analizar")
                subida = st.file_uploader("Sube una foto de una hoja (JPG/PNG)", type=["jpg", "jpeg", "png"])
                muestras = sorted(p.name for p in (prep.RAIZ_PROYECTO / "muestras").glob("*")
                                  if p.suffix.lower() in prep.EXTENSIONES)
                elegida = st.selectbox("...o elige una muestra de prueba", ["(ninguna)"] + muestras)
                img = None
                if subida is not None:
                    img = Image.open(subida)
                elif elegida != "(ninguna)":
                    img = Image.open(prep.RAIZ_PROYECTO / "muestras" / elegida)
                    real = elegida.split("__")[0]
                    st.caption(f"Clase real (según el nombre del archivo): {prep.NOMBRES_LEGIBLES.get(real, real)}")
                if img is not None:
                    tensor, vista = prep.preprocesar_imagen(img, size_modelo)
                    a, b = st.columns(2)
                    a.image(img.convert("RGB"), width=220, caption="Original")
                    b.image(vista, width=220, caption=f"Lo que ve el modelo ({size_modelo}×{size_modelo})")

            with der:
                st.subheader("Diagnóstico")
                if img is None:
                    st.info("Sube una imagen o elige una muestra para ver el diagnóstico.")
                else:
                    probs = dl.predecir(modelo, tensor)
                    mejor = int(np.argmax(probs))
                    clase = clases[mejor]
                    color = COLOR_CLASE[clase]
                    estado = "HOJA SANA" if clase == "Tomato_healthy" else "ENFERMEDAD DETECTADA"
                    st.markdown(
                        f"<div class='tarjeta' style='border-left:6px solid {color}'>"
                        f"<span class='mono'>{estado}</span>"
                        f"<h2 style='margin:4px 0;color:{color}'>{prep.NOMBRES_LEGIBLES[clase]}</h2>"
                        f"<span class='mono'>Confianza: {probs[mejor] * 100:.1f}%</span></div>", unsafe_allow_html=True)
                    st.caption("Probabilidad por clase", help=AYUDA["confianza"])
                    orden = sorted(zip(clases, probs), key=lambda t: -t[1])
                    st.markdown("".join(barra_html(prep.NOMBRES_LEGIBLES[k], p * 100, COLOR_CLASE[k])
                                        for k, p in orden), unsafe_allow_html=True)
                    if probs[mejor] < 0.6:
                        st.warning("Confianza baja: la imagen puede no parecerse a las del entrenamiento "
                                   "(fondo distinto, otra planta o mala iluminación).")

            if img is not None:
                info = INFO_CLASE[clase]
                st.markdown(
                    f"<div class='tarjeta' style='border-left:6px solid {color}'>"
                    f"<h4>Sobre: {prep.NOMBRES_LEGIBLES[clase]}</h4>"
                    f"<p><b>Causa:</b> {info['causa']}</p>"
                    f"<p><b>Cómo se reconoce:</b> {info['sintomas']}</p>"
                    f"<p><b>Qué suele recomendarse:</b> {info['recomendacion']}</p>"
                    f"<p class='nota'>Información orientativa. El diagnóstico definitivo debe confirmarlo un especialista.</p></div>",
                    unsafe_allow_html=True)

# ===========================================================================
# SECCION 3: APRENDIZAJE NO SUPERVISADO
# ===========================================================================
else:
    st.header("Aprendizaje no supervisado", help=AYUDA["ns"])
    st.info("Sección en construcción. Aquí se mostrará el agrupamiento visual de las hojas "
            "(K-Means y PCA) sin usar las etiquetas.")
