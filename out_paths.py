#!/usr/bin/env python3
# Creación y validación de la estructura de salidas del benchmark

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional

def asegurar_dirs_de_salidas(rutas: Iterable[Optional[str]]) -> None:
    """
    Crea el directorio padre de cada ruta (si aplica).
    - Ignora None / "".
    - Ignora rutas sin directorio (ej: 'metricas.json').
    """
    for r in rutas:
        if not r:
            continue
        p = Path(r)
        parent = p.parent
        if str(parent) not in (".", ""):
            parent.mkdir(parents=True, exist_ok=True)
