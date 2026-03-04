#!/usr/bin/env python3
import heapq
from typing import Dict, List, Optional, Tuple
import numpy as np

LIBRE = 0
ESTACION = 2  # transitable
# ANAQUEL y BLOQUEADO son obstáculos

Celda = Tuple[int, int]

def a_estrella(grid: np.ndarray, inicio: Celda, meta: Celda) -> Optional[List[Celda]]:
    """
    Algoritmo A* sobre un grid 4-conectado (arriba/abajo/izquierda/derecha).

    Regresa:
      - Una ruta (lista de celdas) que incluye inicio y meta, o
      - None si la meta no es alcanzable.

    Convención del grid:
      - LIBRE (=0) y ESTACION (=2) son transitables.
      - Cualquier otro valor se considera obstáculo.
    """
    alto, ancho = grid.shape

    def en_rango(x: int, y: int) -> bool:
        return 0 <= x < ancho and 0 <= y < alto

    def transitable(x: int, y: int) -> bool:
        return grid[y, x] in (LIBRE, ESTACION)

    ix, iy = inicio
    mx, my = meta

    # Validación rápida
    if (
        (not en_rango(ix, iy))
        or (not en_rango(mx, my))
        or (not transitable(ix, iy))
        or (not transitable(mx, my))
    ):
        return None

    def heuristica(x: int, y: int) -> int:
        # Distancia Manhattan
        return abs(x - mx) + abs(y - my)

    # heap: (f = g + h, g, (x,y))
    abiertos: List[Tuple[int, int, Celda]] = []
    heapq.heappush(abiertos, (heuristica(ix, iy), 0, (ix, iy)))

    vino_de: Dict[Celda, Celda] = {}
    costo_g: Dict[Celda, int] = {(ix, iy): 0}
    cerrados = set()

    while abiertos:
        _, g_actual, actual = heapq.heappop(abiertos)

        if actual in cerrados:
            continue
        cerrados.add(actual)

        if actual == (mx, my):
            # Reconstrucción de ruta
            ruta = [actual]
            while actual in vino_de:
                actual = vino_de[actual]
                ruta.append(actual)
            ruta.reverse()
            return ruta

        x, y = actual
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy

            if (not en_rango(nx, ny)) or (not transitable(nx, ny)):
                continue

            nuevo_g = g_actual + 1
            if nuevo_g < costo_g.get((nx, ny), 10**9):
                costo_g[(nx, ny)] = nuevo_g
                vino_de[(nx, ny)] = (x, y)
                heapq.heappush(abiertos, (nuevo_g + heuristica(nx, ny), nuevo_g, (nx, ny)))

    return None
