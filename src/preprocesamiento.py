"""
preprocesamiento.py
-------------------
Lectura, redimensionado, normalizacion, data augmentation y division
estratificada (70/15/15) de las imagenes de hojas de tomate.

Estructura esperada en disco:
    data/PlantVillage_Tomate/<nombre_clase>/*.jpg
"""

from pathlib import Path
import random

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_DATASET = RAIZ_PROYECTO / "data" / "PlantVillage_Tomate"

CLASES = [
    "Tomato_healthy",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Bacterial_spot",
]

NOMBRES_LEGIBLES = {
    "Tomato_healthy": "Hoja sana",
    "Tomato_Early_blight": "Tizón temprano",
    "Tomato_Late_blight": "Tizón tardío",
    "Tomato_Bacterial_spot": "Mancha bacteriana",
}

EXTENSIONES = {".jpg", ".jpeg", ".png"}
SEMILLA = 42


# ---------------------------------------------------------------------------
# LECTURA DEL DATASET
# ---------------------------------------------------------------------------

def dataset_disponible(ruta=RUTA_DATASET) -> bool:
    """True si existen las carpetas de las 4 clases con al menos una imagen."""
    ruta = Path(ruta)
    return all(
        (ruta / c).is_dir() and any(p.suffix.lower() in EXTENSIONES for p in (ruta / c).iterdir())
        for c in CLASES
    )


def listar_imagenes(ruta=RUTA_DATASET, max_por_clase=None, semilla=SEMILLA):
    """
    Recorre las carpetas de clases y devuelve (rutas, etiquetas).
    Si max_por_clase no es None, toma una muestra aleatoria reproducible
    por clase (util para entrenar rapido en CPU).
    """
    ruta = Path(ruta)
    rng = random.Random(semilla)
    rutas, etiquetas = [], []
    for idx, clase in enumerate(CLASES):
        archivos = sorted(p for p in (ruta / clase).iterdir() if p.suffix.lower() in EXTENSIONES)
        if max_por_clase is not None and len(archivos) > max_por_clase:
            archivos = rng.sample(archivos, max_por_clase)
        rutas.extend(archivos)
        etiquetas.extend([idx] * len(archivos))
    return rutas, etiquetas


def conteo_por_clase(ruta=RUTA_DATASET) -> dict:
    """Cantidad total de imagenes por clase (sin muestreo)."""
    ruta = Path(ruta)
    return {
        c: sum(1 for p in (ruta / c).iterdir() if p.suffix.lower() in EXTENSIONES)
        for c in CLASES
    }


# ---------------------------------------------------------------------------
# DIVISION ESTRATIFICADA 70 / 15 / 15
# ---------------------------------------------------------------------------

def dividir_estratificado(rutas, etiquetas, semilla=SEMILLA):
    """
    Divide en train (70%), validacion (15%) y prueba (15%) manteniendo la
    proporcion de cada clase en los tres conjuntos.
    Devuelve un diccionario {"train": (rutas, etiquetas), "val": ..., "test": ...}.
    """
    r_train, r_temp, y_train, y_temp = train_test_split(
        rutas, etiquetas, test_size=0.30, stratify=etiquetas, random_state=semilla
    )
    r_val, r_test, y_val, y_test = train_test_split(
        r_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=semilla
    )
    return {
        "train": (r_train, y_train),
        "val": (r_val, y_val),
        "test": (r_test, y_test),
    }


# ---------------------------------------------------------------------------
# TRANSFORMACIONES
# ---------------------------------------------------------------------------

def crear_transformaciones(img_size=128):
    """
    - entrenamiento: redimensiona + data augmentation + ToTensor ([0,1])
    - evaluacion:    redimensiona + ToTensor ([0,1])  (sin aumentos)
    ToTensor convierte a float32 y escala los pixeles de [0,255] a [0,1].
    """
    t_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])
    t_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    return t_train, t_eval


class HojasDataset(Dataset):
    """Dataset de PyTorch: lee la imagen desde disco, la pasa a RGB y la transforma."""

    def __init__(self, rutas, etiquetas, transform):
        self.rutas = list(rutas)
        self.etiquetas = list(etiquetas)
        self.transform = transform

    def __len__(self):
        return len(self.rutas)

    def __getitem__(self, i):
        img = Image.open(self.rutas[i]).convert("RGB")
        return self.transform(img), self.etiquetas[i]


def crear_dataloaders(img_size=128, batch_size=32, max_por_clase=None,
                      ruta=RUTA_DATASET, num_workers=0):
    """
    Pipeline completo: lista imagenes -> divide 70/15/15 -> crea DataLoaders.
    num_workers=0 es lo mas seguro en Windows (y con Streamlit).
    Devuelve (loaders, splits) con loaders = {"train","val","test"}.
    """
    rutas, etiquetas = listar_imagenes(ruta, max_por_clase)
    splits = dividir_estratificado(rutas, etiquetas)
    t_train, t_eval = crear_transformaciones(img_size)

    loaders = {}
    for nombre, (r, y) in splits.items():
        ds = HojasDataset(r, y, t_train if nombre == "train" else t_eval)
        loaders[nombre] = DataLoader(
            ds, batch_size=batch_size, shuffle=(nombre == "train"),
            num_workers=num_workers,
        )
    return loaders, splits


# ---------------------------------------------------------------------------
# UTILIDADES PARA LA INTERFAZ
# ---------------------------------------------------------------------------

def preprocesar_imagen(img_pil: Image.Image, img_size=128):
    """
    Aplica a UNA imagen subida por el usuario exactamente el mismo
    preprocesamiento de evaluacion usado en el entrenamiento.
    Devuelve (tensor [1,3,H,W], array uint8 HxWx3 para mostrar).
    """
    img = img_pil.convert("RGB").resize((img_size, img_size))
    tensor = transforms.ToTensor()(img).unsqueeze(0)
    return tensor, np.array(img)


def imagen_con_augmentation(img_pil: Image.Image, img_size=128):
    """Aplica una vez el pipeline de aumento (para mostrar ejemplos en la interfaz)."""
    t_train, _ = crear_transformaciones(img_size)
    t = t_train(img_pil.convert("RGB"))
    return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
