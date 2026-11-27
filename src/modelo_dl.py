"""
modelo_dl.py  (Avance 1 - Aprendizaje Profundo)
-----------------------------------------------
CNN propia (sin YOLO ni redes preentrenadas) para clasificar hojas de tomate.

Arquitectura:
    [Conv2D -> BatchNorm -> ReLU -> MaxPool2D] x 4  (32, 64, 128, 128 filtros)
    -> AdaptiveAvgPool (4x4) -> Flatten -> Dense(256) -> ReLU -> Dropout
    -> Dense(n_clases)  (Softmax se aplica en CrossEntropyLoss y al predecir)

El AdaptiveAvgPool permite usar 128x128 o 224x224 sin cambiar el codigo.
"""

import copy
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

import torch
import torch.nn as nn

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_MODELO = RAIZ_PROYECTO / "models_saved" / "modelo_cnn_foliar.pth"

# Colores de la interfaz (ver guia de diseno)
C_FONDO, C_TARJETA, C_ROJO, C_TEXTO, C_VERDE = "#0B0E0C", "#151A17", "#E11D48", "#F5F5F4", "#16C172"


def obtener_dispositivo():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# ARQUITECTURA
# ---------------------------------------------------------------------------

def _bloque(entrada, salida):
    return nn.Sequential(
        nn.Conv2d(entrada, salida, kernel_size=3, padding=1),
        nn.BatchNorm2d(salida),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CNNFoliar(nn.Module):
    def __init__(self, n_clases=4, dropout=0.4):
        super().__init__()
        self.caracteristicas = nn.Sequential(
            _bloque(3, 32), _bloque(32, 64), _bloque(64, 128), _bloque(128, 128),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.clasificador = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_clases),
        )

    def forward(self, x):
        return self.clasificador(self.caracteristicas(x))


def contar_parametros(modelo) -> int:
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# ENTRENAMIENTO
# ---------------------------------------------------------------------------

def _epoca(modelo, loader, criterio, dispositivo, optimizador=None):
    """Una pasada completa. Si hay optimizador entrena, si no solo evalua."""
    entrenando = optimizador is not None
    modelo.train(entrenando)
    perdida_total, aciertos, total = 0.0, 0, 0
    with torch.set_grad_enabled(entrenando):
        for x, y in loader:
            x, y = x.to(dispositivo), y.to(dispositivo)
            salida = modelo(x)
            perdida = criterio(salida, y)
            if entrenando:
                optimizador.zero_grad()
                perdida.backward()
                optimizador.step()
            perdida_total += perdida.item() * x.size(0)
            aciertos += (salida.argmax(1) == y).sum().item()
            total += x.size(0)
    return perdida_total / total, aciertos / total


def entrenar(modelo, loaders, epocas=10, lr=1e-3, dispositivo=None,
             callback=None, ruta_guardado=None, clases=None, img_size=128):
    """
    Entrena con CrossEntropyLoss + Adam y conserva los pesos de la mejor
    epoca segun accuracy de validacion (evita quedarse con una epoca
    sobreajustada).

    callback(epoca, epocas, historial) se llama al final de cada epoca
    (la interfaz lo usa para barra de progreso y curvas en vivo).
    Devuelve el historial con loss/acc de train y val y el tiempo total.
    """
    dispositivo = dispositivo or obtener_dispositivo()
    modelo.to(dispositivo)
    criterio = nn.CrossEntropyLoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=lr)

    historial = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
    mejores_pesos, mejor_val = copy.deepcopy(modelo.state_dict()), -1.0
    t0 = time.time()

    for e in range(1, epocas + 1):
        l, a = _epoca(modelo, loaders["train"], criterio, dispositivo, optimizador)
        vl, va = _epoca(modelo, loaders["val"], criterio, dispositivo)
        historial["loss"].append(l)
        historial["acc"].append(a)
        historial["val_loss"].append(vl)
        historial["val_acc"].append(va)

        if va > mejor_val:
            mejor_val = va
            mejores_pesos = copy.deepcopy(modelo.state_dict())

        if callback:
            callback(e, epocas, historial)

    modelo.load_state_dict(mejores_pesos)
    historial["tiempo_s"] = time.time() - t0
    historial["mejor_val_acc"] = mejor_val

    if ruta_guardado:
        guardar_modelo(modelo, ruta_guardado, clases, img_size)
    return historial


# ---------------------------------------------------------------------------
# VALIDACION
# ---------------------------------------------------------------------------

def evaluar(modelo, loader, dispositivo=None):
    """Devuelve (y_real, y_predicho, probabilidades) sobre un DataLoader."""
    dispositivo = dispositivo or obtener_dispositivo()
    modelo.to(dispositivo).eval()
    reales, preds, probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            p = torch.softmax(modelo(x.to(dispositivo)), dim=1).cpu()
            probs.append(p.numpy())
            preds.append(p.argmax(1).numpy())
            reales.append(y.numpy())
    return np.concatenate(reales), np.concatenate(preds), np.concatenate(probs)


def calcular_metricas(y_real, y_pred, clases):
    """Accuracy y Precision/Recall/F1 macro (promedio igualitario entre clases)."""
    return {
        "Accuracy": accuracy_score(y_real, y_pred),
        "Precision": precision_score(y_real, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_real, y_pred, average="macro", zero_division=0),
        "F1-Score": f1_score(y_real, y_pred, average="macro", zero_division=0),
        "matriz": confusion_matrix(y_real, y_pred, labels=list(range(len(clases)))),
        "reporte": classification_report(
            y_real, y_pred, labels=list(range(len(clases))),
            target_names=clases, output_dict=True, zero_division=0),
    }


# ---------------------------------------------------------------------------
# GRAFICOS (estilo oscuro coherente con la interfaz)
# ---------------------------------------------------------------------------

def _estilo(fig, ax):
    fig.patch.set_facecolor(C_TARJETA)
    for a in np.atleast_1d(ax):
        a.set_facecolor(C_FONDO)
        a.tick_params(colors=C_TEXTO)
        a.xaxis.label.set_color(C_TEXTO)
        a.yaxis.label.set_color(C_TEXTO)
        a.title.set_color(C_TEXTO)
        for s in a.spines.values():
            s.set_color("#27272A")
        a.grid(alpha=0.15)


def graficar_curvas(historial):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ep = range(1, len(historial["loss"]) + 1)
    ax[0].plot(ep, historial["loss"], color=C_ROJO, marker="o", label="Entrenamiento")
    ax[0].plot(ep, historial["val_loss"], color=C_VERDE, marker="s", label="Validación")
    ax[0].set_title("Loss (CrossEntropy)")
    ax[1].plot(ep, historial["acc"], color=C_ROJO, marker="o", label="Entrenamiento")
    ax[1].plot(ep, historial["val_acc"], color=C_VERDE, marker="s", label="Validación")
    ax[1].set_title("Accuracy")
    for a in ax:
        a.set_xlabel("Época")
        a.legend(facecolor=C_TARJETA, labelcolor=C_TEXTO, edgecolor="#27272A")
    _estilo(fig, ax)
    fig.tight_layout()
    return fig


def graficar_matriz_confusion(matriz, etiquetas):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Reds", ax=ax,
                xticklabels=etiquetas, yticklabels=etiquetas,
                cbar=False, annot_kws={"color": "white", "fontsize": 11},
                linecolor=C_FONDO, linewidths=1)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor real")
    ax.set_title("Matriz de confusión (conjunto de prueba)")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    _estilo(fig, ax)
    ax.grid(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# GUARDAR / CARGAR / PREDECIR
# ---------------------------------------------------------------------------

def guardar_modelo(modelo, ruta, clases, img_size):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": modelo.state_dict(),
        "clases": list(clases),
        "img_size": int(img_size),
    }, ruta)


def cargar_modelo(ruta=RUTA_MODELO, dispositivo=None):
    """Devuelve (modelo en eval, clases, img_size) o None si no existe el archivo."""
    ruta = Path(ruta)
    if not ruta.exists():
        return None
    dispositivo = dispositivo or obtener_dispositivo()
    ck = torch.load(ruta, map_location=dispositivo, weights_only=False)
    modelo = CNNFoliar(n_clases=len(ck["clases"]))
    modelo.load_state_dict(ck["state_dict"])
    modelo.to(dispositivo).eval()
    return modelo, ck["clases"], ck["img_size"]


def predecir(modelo, tensor, dispositivo=None):
    """tensor [1,3,H,W] -> vector de probabilidades (softmax) por clase."""
    dispositivo = dispositivo or obtener_dispositivo()
    modelo.to(dispositivo).eval()
    with torch.no_grad():
        return torch.softmax(modelo(tensor.to(dispositivo)), dim=1)[0].cpu().numpy()
