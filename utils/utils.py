import re
from unicodedata import normalize, category
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    hamming_loss,
    accuracy_score,
    f1_score,
)

import pandas as pd

comunes = {
    "el",
    "la",
    "de",
    "que",
    "y",
    "a",
    "en",
    "un",
    "ser",
    "se",
    "no",
    "haber",
    "por",
    "con",
    "su",
    "para",
    "es",
    "una",
    "del",
    "las",
    "como",
    "este",
    "o",
    "fue",
}


def preprocess_text(text: str) -> str:
    """Normaliza el texto del campo `motivos`

    Args:
        text (str): texto a normalizar

    Returns:
        str: texto normalizado.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r"\s+", " ", text).strip()  # quitar espacios inecesarios
    tokens = [
        t for t in text.split() if t not in comunes and len(t) > 2
    ]  # tokenización, omitiendo las comunes
    return " ".join(tokens)


def norm_key(t):
    t = clean_text(t).lower()
    t = "".join(c for c in normalize("NFD", t) if category(c) != "Mn")
    return re.sub(r"[^a-z0-9 ]", "", t).strip()


def clean_text(t):
    t = str(t).replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", t).strip()


THRESHOLD_GRID = np.arange(0.1, 0.9, 0.05)


def tune_thresholds(
    y_val: pd.DataFrame, probs_val: pd.DataFrame, grid: np.ndarray = THRESHOLD_GRID
):
    """Para cada etiqueta elige el umbral que maximiza su F1 en validación.

    Args:
        y_val (pd.DataFrame): etiquetas de validación.
        probs_val (pd.DataFrame): probabilidades de salida de validación
        grid (np.ndarray, optional): Espacio de valores a probar. Defaults to THRESHOLD_GRID.

    Returns:
        _type_: _description_
    """
    y_val = np.asarray(y_val)
    thresholds = []
    for j in range(y_val.shape[1]):
        if (
            y_val[:, j].sum() == 0
        ):  # sin positivos en validación no hay señal: se deja 0.5
            thresholds.append(0.5)
            continue
        # ante empates de F1 se prefiere el umbral más cercano a 0.5
        best_t = max(
            grid,
            key=lambda t: (
                f1_score(y_val[:, j], probs_val[:, j] > t, zero_division=0),
                -abs(t - 0.5),
            ),
        )
        thresholds.append(best_t)
    return np.array(thresholds)


def ablation_lr(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_valid: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
) -> pd.DataFrame:
    """Genera los experimentos de ablación para la regresión logistica.

    Args:
        name (str): nombre del modelo
        X_train (np.ndarray): conj entrenamiento.
        X_val (np.ndarray): conj validación.
        C (float, optional): mejor valor de C escogido. Defaults to 1.0.
        max_iter (int, optional): Defaults to 1000.

    Returns:
        pd.Dataframe: resultados.
    """
    rows = []
    for cw in [None, "balanced"]:
        clf = OneVsRestClassifier(
            LogisticRegression(C=C, class_weight=cw, max_iter=max_iter)
        ).fit(X_train, y_train)
        predictions = clf.predict_proba(X_val)
        th = tune_thresholds(y_valid, predictions)  # aprendidos en validación
        for use_th in [False, True]:
            t = th if use_th else 0.5
            y_val = (predictions > t).astype(int)
            rows.append(
                {
                    "enfoque": name,
                    "class_weight": cw or "none",
                    "umbrales": "calculados" if use_th else "0.5",
                    "val_hamming": hamming_loss(y_valid, y_val),
                    "val_accuracy": accuracy_score(y_valid, y_val),
                    "val_macroF1": f1_score(
                        y_valid, y_val, average="macro", zero_division=0
                    ),
                    "val_microF1": f1_score(
                        y_valid, y_val, average="micro", zero_division=0
                    ),
                }
            )
    return pd.DataFrame(rows)
