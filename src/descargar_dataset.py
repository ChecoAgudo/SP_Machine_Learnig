"""
descargar_dataset.py
--------------------
Descarga PlantVillage (Kaggle: emmarex/plantdisease) y copia SOLO las 4 clases
de tomate a data/PlantVillage_Tomate/.

Uso:
    python -m src.descargar_dataset                      # descarga con kagglehub
    python -m src.descargar_dataset --origen RUTA        # usa una carpeta ya descargada/descomprimida
    python -m src.descargar_dataset --max-por-clase 800  # copia solo 800 imagenes por clase

Desde la raiz del proyecto.
"""

import argparse
import shutil
from pathlib import Path

from src.preprocesamiento import CLASES, EXTENSIONES, RUTA_DATASET, RAIZ_PROYECTO

SLUG_KAGGLE = "emmarex/plantdisease"


def _normalizar(nombre: str) -> str:
    """'Tomato___Early_blight' y 'Tomato_Early_blight' -> 'tomato_early_blight'."""
    partes = [p for p in nombre.lower().split("_") if p]
    return "_".join(partes)


def buscar_carpetas_clases(origen: Path) -> dict:
    """Busca recursivamente las carpetas de las 4 clases dentro de 'origen'."""
    objetivo = {_normalizar(c): c for c in CLASES}
    encontradas = {}
    for carpeta in origen.rglob("*"):
        if carpeta.is_dir():
            clave = _normalizar(carpeta.name)
            if clave in objetivo and objetivo[clave] not in encontradas:
                tiene_imgs = any(p.suffix.lower() in EXTENSIONES for p in carpeta.iterdir())
                if tiene_imgs:
                    encontradas[objetivo[clave]] = carpeta
    return encontradas


def copiar_clases(origen: Path, destino: Path, max_por_clase=None):
    encontradas = buscar_carpetas_clases(origen)
    faltan = [c for c in CLASES if c not in encontradas]
    if faltan:
        raise FileNotFoundError(
            f"No se encontraron las carpetas: {faltan} dentro de {origen}. "
            "Revisa que sea el dataset PlantVillage (emmarex/plantdisease)."
        )

    destino.mkdir(parents=True, exist_ok=True)
    for clase, carpeta in encontradas.items():
        archivos = sorted(p for p in carpeta.iterdir() if p.suffix.lower() in EXTENSIONES)
        if max_por_clase:
            archivos = archivos[:max_por_clase]
        sub = destino / clase
        sub.mkdir(exist_ok=True)
        for a in archivos:
            shutil.copy2(a, sub / a.name)
        print(f"  {clase}: {len(archivos)} imagenes copiadas")


def copiar_muestras(destino_datos: Path, n_por_clase=2):
    """Copia unas pocas imagenes a muestras/ para la demostracion (esta carpeta SI va a GitHub)."""
    muestras = RAIZ_PROYECTO / "muestras"
    muestras.mkdir(exist_ok=True)
    total = 0
    for clase in CLASES:
        archivos = sorted(p for p in (destino_datos / clase).iterdir()
                          if p.suffix.lower() in EXTENSIONES)
        # se toman del final de la lista para que sean distintas a las primeras
        for a in archivos[-n_por_clase:]:
            shutil.copy2(a, muestras / f"{clase}__{a.name}")
            total += 1
    print(f"  {total} imagenes de muestra en {muestras}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origen", type=str, default=None,
                    help="Carpeta del dataset ya descargado (omite kagglehub).")
    ap.add_argument("--max-por-clase", type=int, default=None)
    args = ap.parse_args()

    if args.origen:
        origen = Path(args.origen)
    else:
        try:
            import kagglehub
        except ImportError:
            raise SystemExit("Falta kagglehub: pip install kagglehub")
        print(f"Descargando {SLUG_KAGGLE} con kagglehub (puede tardar unos minutos)...")
        origen = Path(kagglehub.dataset_download(SLUG_KAGGLE))
        print(f"Descargado en: {origen}")

    print(f"Copiando las 4 clases de tomate a {RUTA_DATASET} ...")
    copiar_clases(origen, RUTA_DATASET, args.max_por_clase)
    copiar_muestras(RUTA_DATASET)
    print("Listo.")


if __name__ == "__main__":
    main()
