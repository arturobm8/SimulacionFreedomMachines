#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from out_paths import asegurar_dirs_de_salidas

LIBRE = 0
ANAQUEL = 1
ESTACION = 2
BLOQUEADO = 3

Celda = Tuple[int, int]

@dataclass(frozen=True)
class Estacion:
    estacion_id: int
    dock: Celda     
    cell: Celda     

def _ruta_por_escenario(escenario: str, nombre_archivo: str) -> str:
    return os.path.join("outputs", escenario, nombre_archivo)

def _recortar_rectangulo(grid: np.ndarray, x0: int, y0: int, ancho: int, alto: int, valor: int) -> None:
    alto_grid, ancho_grid = grid.shape
    x1 = max(0, min(ancho_grid, x0 + ancho))
    y1 = max(0, min(alto_grid, y0 + alto))
    x0 = max(0, min(ancho_grid, x0))
    y0 = max(0, min(alto_grid, y0))
    grid[y0:y1, x0:x1] = valor

def _en_rango(grid: np.ndarray, x: int, y: int) -> bool:
    alto_grid, ancho_grid = grid.shape
    return 0 <= x < ancho_grid and 0 <= y < alto_grid

def _bfs_alcanzable(grid: np.ndarray, celdas_inicio: List[Celda]) -> np.ndarray:
    """
    Regresa matriz booleana de alcanzabilidad para celdas LIBRE/ESTACION.
    ANAQUEL/BLOQUEADO se tratan como muros.
    """
    alto_grid, ancho_grid = grid.shape
    transitable = (grid == LIBRE) | (grid == ESTACION)
    visto = np.zeros((alto_grid, ancho_grid), dtype=bool)

    cola: List[Celda] = []
    for (x, y) in celdas_inicio:
        if _en_rango(grid, x, y) and transitable[y, x]:
            visto[y, x] = True
            cola.append((x, y))

    indice = 0
    while indice < len(cola):
        x, y = cola[indice]
        indice += 1
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if _en_rango(grid, nx, ny) and (not visto[ny, nx]) and transitable[ny, nx]:
                visto[ny, nx] = True
                cola.append((nx, ny))
    return visto

def generar_layout(seed: int, ancho: int, alto: int, estaciones: int) -> Dict:
    """
    Concepto del layout:
    - Frente sur: fila de estaciones (y=H-2) y carril buffer (y=H-3)
    - Zona de parking/carga: rectángulo abajo-izquierda (LIBRE)
    - Almacenaje: macro-bloques de racks con pasillos internos + pasillos principales
    - Cross-aisles: corredores horizontales (2 celdas) cada ~10 filas en región de almacenaje
    """
    rng = np.random.default_rng(seed)
    grid = np.full((alto, ancho), LIBRE, dtype=np.int8)

    # Borde exterior bloqueado (margen de 1 celda)
    _recortar_rectangulo(grid, 0, 0, ancho, 1, BLOQUEADO)
    _recortar_rectangulo(grid, 0, alto - 1, ancho, 1, BLOQUEADO)
    _recortar_rectangulo(grid, 0, 0, 1, alto, BLOQUEADO)
    _recortar_rectangulo(grid, ancho - 1, 0, 1, alto, BLOQUEADO)

    # Estaciones en el sur
    y_estacion = alto - 2
    y_buffer = alto - 3

    ancho_total_estaciones = estaciones * 2
    x_inicio = (ancho - ancho_total_estaciones) // 2
    lista_estaciones: List[Estacion] = []

    # Carril buffer como LIBRE
    for x in range(1, ancho - 1):
        if grid[y_buffer, x] != BLOQUEADO:
            grid[y_buffer, x] = LIBRE

    for i in range(estaciones):
        sx = x_inicio + i * 2

        # estación ocupa 2 celdas de ancho
        for dx in (0, 1):
            if _en_rango(grid, sx + dx, y_estacion):
                grid[y_estacion, sx + dx] = ESTACION

        dock = (sx, y_buffer)  
        cell = (sx, y_estacion)
        lista_estaciones.append(Estacion(estacion_id=i, dock=dock, cell=cell))

    # Apron frontal arriba del buffer (para incorporaciones)
    _recortar_rectangulo(grid, 1, y_buffer - 2, ancho - 2, 2, LIBRE)

    # Zona de parking/carga
    ancho_parking, alto_parking = 12, 8
    x_parking0, y_parking0 = 2, alto - (alto_parking + 5)
    _recortar_rectangulo(grid, x_parking0, y_parking0, ancho_parking, alto_parking, LIBRE)

    # Región de almacenaje
    y_top = 2
    y_bottom = y_buffer - 4
    x_left = 2
    x_right = ancho - 3

    alto_storage = y_bottom - y_top
    ancho_storage = x_right - x_left + 1

    # Macro-bloques 3x2 con pasillos principales
    ancho_pasillo_principal = 2
    cols = 3
    filas = 2

    ancho_total_pasillos_verticales = (cols + 1) * ancho_pasillo_principal
    ancho_bloque = (ancho_storage - ancho_total_pasillos_verticales) // cols

    alto_pasillo_horizontal = 2
    alto_total_pasillos_horiz = (filas + 1) * alto_pasillo_horizontal
    alto_bloque = (alto_storage - alto_total_pasillos_horiz) // filas

    # Pasillos principales verticales
    x = x_left
    for _ in range(cols + 1):
        _recortar_rectangulo(grid, x, y_top, ancho_pasillo_principal, alto_storage, LIBRE)
        x += ancho_pasillo_principal + ancho_bloque

    # Pasillos principales horizontales
    y = y_top
    for _ in range(filas + 1):
        _recortar_rectangulo(grid, x_left, y, ancho_storage, alto_pasillo_horizontal, LIBRE)
        y += alto_pasillo_horizontal + alto_bloque

    # Rellenar bloques con patrón de anaqueles
    anaqueles: Dict[int, Celda] = {}
    anaquel_id = 0

    def _llenar_bloque_con_anaqueles(x0: int, y0: int, w: int, h: int) -> None:
        nonlocal anaquel_id

        # margen interno
        ix0, iy0 = x0 + 1, y0 + 1
        iw, ih = max(0, w - 2), max(0, h - 2)
        if iw <= 0 or ih <= 0:
            return

        col = 0
        xx = ix0
        while xx < ix0 + iw:
            # patrón: 2 columnas de anaquel, 1 columna libre (pasillo interno)
            if col % 3 in (0, 1):
                for yy in range(iy0, iy0 + ih):
                    # huecos raros (mantenimiento)
                    if (yy - iy0) % 17 == 0 and (xx - ix0) % 11 == 0:
                        continue
                    if grid[yy, xx] == LIBRE:
                        grid[yy, xx] = ANAQUEL
                        anaqueles[anaquel_id] = (xx, yy)
                        anaquel_id += 1
            else:
                for yy in range(iy0, iy0 + ih):
                    if grid[yy, xx] != BLOQUEADO:
                        grid[yy, xx] = LIBRE
            col += 1
            xx += 1

    # Iterar posiciones de bloques
    y = y_top + alto_pasillo_horizontal
    for _ in range(filas):
        x = x_left + ancho_pasillo_principal
        for _ in range(cols):
            _llenar_bloque_con_anaqueles(x, y, ancho_bloque, alto_bloque)
            x += ancho_bloque + ancho_pasillo_principal
        y += alto_bloque + alto_pasillo_horizontal

    # Cross-aisles cada 10 filas (corredor de 2 filas)
    cada = 10
    for yy in range(y_top + 5, y_bottom, cada):
        _recortar_rectangulo(grid, x_left, yy, ancho_storage, 2, LIBRE)

    # Conectar parking con apron
    x_corredor = x_parking0 + ancho_parking + 2
    _recortar_rectangulo(grid, x_corredor, y_parking0 - 15, 2, (alto - 2) - (y_parking0 - 15), LIBRE)

    # puntos de spawn dentro del parking
    spawn_points: List[Celda] = []
    for yy in range(y_parking0, y_parking0 + alto_parking):
        for xx in range(x_parking0, x_parking0 + ancho_parking):
            if grid[yy, xx] == LIBRE:
                spawn_points.append((xx, yy))
    rng.shuffle(spawn_points)
    spawn_points = spawn_points[:200]

    # Validar alcanzabilidad: desde el primer spawn, docks alcanzables
    if not spawn_points:
        raise RuntimeError("No se generaron spawn points.")
    alcanzable = _bfs_alcanzable(grid, [spawn_points[0]])

    for est in lista_estaciones:
        dx, dy = est.dock
        if not alcanzable[dy, dx]:
            _recortar_rectangulo(grid, dx, dy - 5, 2, 6, LIBRE)

    # Ensayar de nuevo alcanzabilidad
    _ = _bfs_alcanzable(grid, [spawn_points[0]])

    layout = {
        "seed": seed,
        "width": ancho,
        "height": alto,
        "grid": grid,
        # JSONs
        "estaciones": [{"estacion_id": e.estacion_id, "dock": e.dock, "cell": e.cell} for e in lista_estaciones],
        "anaqueles": [{"anaquel_id": aid, "home": home} for aid, home in anaqueles.items()],
        "spawn_points": spawn_points,
        "constants": {"LIBRE": LIBRE, "ANAQUEL": ANAQUEL, "ESTACION": ESTACION, "BLOQUEADO": BLOQUEADO},
    }
    return layout

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ancho", type=int, default=120)
    parser.add_argument("--alto", type=int, default=80)
    parser.add_argument("--estaciones", type=int, default=20)

    parser.add_argument(
        "--escenario",
        type=str,
        default="seed42",
        help="Nombre del escenario. Escribe en outputs/<escenario>/",
    )

    # Overrides (compatibilidad / modo avanzado)
    parser.add_argument("--salida_layout", type=str, default=None,
                        help="(Opcional) Ruta explícita para layout.npy")
    parser.add_argument("--salida_estaciones", type=str, default=None,
                        help="(Opcional) Ruta explícita para estaciones.json")
    parser.add_argument("--salida_anaqueles", type=str, default=None,
                        help="(Opcional) Ruta explícita para anaqueles.json")
    parser.add_argument("--salida_spawn", type=str, default=None,
                        help="(Opcional) Ruta explícita para spawn.json")

    # parámetro viejo, sin uso oficial
    parser.add_argument(
        "--prefijo",
        type=str,
        default=None,
        help="(Deprecated) Use --escenario. Si se especifica, se usará como escenario si --escenario no fue dado.",
    )

    args = parser.parse_args()

    # este código deberá eliminarse eventualmente, pero lo dejo por compatibilidad con las primeras versioens (seed42)
    # (Con default seed42, este if solo aplica si pasan --prefijo explícitamente)
    if args.prefijo is not None and args.escenario == "seed42":
        args.escenario = args.prefijo

    # Resolver rutas estándar
    ruta_layout = args.salida_layout or _ruta_por_escenario(args.escenario, "layout.npy")
    ruta_estaciones = args.salida_estaciones or _ruta_por_escenario(args.escenario, "estaciones.json")
    ruta_anaqueles = args.salida_anaqueles or _ruta_por_escenario(args.escenario, "anaqueles.json")
    ruta_spawn = args.salida_spawn or _ruta_por_escenario(args.escenario, "spawn.json")

    asegurar_dirs_de_salidas([ruta_layout, ruta_estaciones, ruta_anaqueles, ruta_spawn])

    layout = generar_layout(args.seed, args.ancho, args.alto, args.estaciones)
    grid = layout["grid"]

    # Guardar grid
    np.save(ruta_layout, grid)

    # Guardar JSONs
    with open(ruta_estaciones, "w", encoding="utf-8") as f:
        json.dump(layout["estaciones"], f, indent=2, ensure_ascii=False)

    with open(ruta_anaqueles, "w", encoding="utf-8") as f:
        json.dump(layout["anaqueles"], f, indent=2, ensure_ascii=False)

    with open(ruta_spawn, "w", encoding="utf-8") as f:
        json.dump(layout["spawn_points"], f, indent=2, ensure_ascii=False)

    # Resumen
    libres = int(np.sum(grid == LIBRE))
    anaquel_celdas = int(np.sum(grid == ANAQUEL))
    estacion_celdas = int(np.sum(grid == ESTACION))
    bloqueadas = int(np.sum(grid == BLOQUEADO))

    print(f"[OK] Escenario: {args.escenario}")
    print(f"[OK] Layout     : {ruta_layout}")
    print(f"[OK] Estaciones : {ruta_estaciones}")
    print(f"[OK] Anaqueles  : {ruta_anaqueles}")
    print(f"[OK] Spawn      : {ruta_spawn}")
    print("Layout generado:")
    print(f"  tamaño: {args.ancho}x{args.alto}")
    print(f"  celdas: LIBRE={libres} ANAQUEL={anaquel_celdas} ESTACION={estacion_celdas} BLOQUEADO={bloqueadas}")
    print(f"  anaqueles: {len(layout['anaqueles'])}")
    print(f"  estaciones: {len(layout['estaciones'])}")
    print(f"  spawn_points: {len(layout['spawn_points'])}")

if __name__ == "__main__":
    main()
