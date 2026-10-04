import re
from unicodedata import normalize, category
import numpy as np
from sklearn.metrics import f1_score

comunes = {
    'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'ser', 'se', 'no', 'haber',
    'por', 'con', 'su', 'para', 'es', 'una', 'del', 'las', 'como', 'este', 'o', 'fue'
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
    #text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip() # quitar espacios inecesarios
    tokens = [t for t in text.split() if t not in comunes and len(t) > 2] # tokenización, omitiendo las comunes
    return ' '.join(tokens)

def norm_key(t):
    t = clean_text(t).lower()
    t = ''.join(c for c in normalize('NFD', t) if category(c) != 'Mn')
    return re.sub(r'[^a-z0-9 ]', '', t).strip()

def clean_text(t):
    t = str(t).replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", t).strip()


THRESHOLD_GRID = np.arange(0.1, 0.9, 0.05)

def tune_thresholds(y_val, probs_val, grid=THRESHOLD_GRID):
    """Para cada etiqueta elige el umbral que maximiza su F1 en validación."""
    y_val = np.asarray(y_val)
    thresholds = []
    for j in range(y_val.shape[1]):
        if y_val[:, j].sum() == 0:   # sin positivos en validación no hay señal: se deja 0.5
            thresholds.append(0.5)
            continue
        # ante empates de F1 se prefiere el umbral más cercano a 0.5
        best_t = max(
            grid,
            key=lambda t: (f1_score(y_val[:, j], probs_val[:, j] > t, zero_division=0),
                           -abs(t - 0.5))
        )
        thresholds.append(best_t)
    return np.array(thresholds)