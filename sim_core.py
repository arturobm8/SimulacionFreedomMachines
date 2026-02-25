#!/usr/bin/env python3
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Tuple
import numpy as np
from a_estrella import a_estrella
from tabla_reservas import TablaReservas

LIBRE = 0
ESTACION = 2

Celda = Tuple[int, int]

# Utilidades de rutas (estándar outputs/<escenario>/...)

def _ruta_por_escenario(escenario: str, nombre_archivo: str) -> str:
    return os.path.join("outputs", escenario, nombre_archivo)

class RutasEscenario(NamedTuple):
    layout: str
    estaciones: str
    anaqueles: str
    spawn: str
    pedidos: str

def resolver_rutas_escenario(escenario: str) -> RutasEscenario:
    """
    Convención estándar:
      outputs/<escenario>/
        layout.npy
        estaciones.json
        anaqueles.json
        spawn.json
        pedidos.json
    """
    return RutasEscenario(
        layout=_ruta_por_escenario(escenario, "layout.npy"),
        estaciones=_ruta_por_escenario(escenario, "estaciones.json"),
        anaqueles=_ruta_por_escenario(escenario, "anaqueles.json"),
        spawn=_ruta_por_escenario(escenario, "spawn.json"),
        pedidos=_ruta_por_escenario(escenario, "pedidos.json"),
    )

# Modelo de datos

@dataclass
class Pedido:
    pedido_id: int
    anaquel_id: int
    estacion_id: int
    tick_creacion: int
    tick_asignacion: Optional[int] = None
    tick_completado: Optional[int] = None

@dataclass
class Robot:
    robot_id: int
    pos: Celda
    estado: str = "INACTIVO"  # INACTIVO, A_RECOGER, A_ESTACION, RETORNO
    pedido_id: Optional[int] = None
    anaquel_home: Optional[Celda] = None
    estacion_dock: Optional[Celda] = None
    ruta: List[Celda] = field(default_factory=list)
    idx_ruta: int = 0
    ticks_espera: int = 0
    celdas_movidas: int = 0
    ticks_ocupado: int = 0

def cargar_layout(ruta_grid: str, ruta_estaciones: str, ruta_anaqueles: str, ruta_spawn: str):
    """
    Carga el layout del CEDIS y sus entidades desde archivos.

    Regresa:
      - grid (np.ndarray)
      - estacion_dock: Dict[estacion_id -> (x,y)]
      - anaquel_home: Dict[anaquel_id -> (x,y)]
      - spawns: List[(x,y)]
    """
    grid = np.load(ruta_grid)

    with open(ruta_estaciones, "r", encoding="utf-8") as f:
        estaciones = json.load(f)
    with open(ruta_anaqueles, "r", encoding="utf-8") as f:
        anaqueles = json.load(f)
    with open(ruta_spawn, "r", encoding="utf-8") as f:
        spawns = json.load(f)

    estacion_dock = {int(e["estacion_id"]): tuple(e["dock"]) for e in estaciones}
    anaquel_home = {int(a["anaquel_id"]): tuple(a["home"]) for a in anaqueles}

    # Robustez: normalizar spawn a tuplas de int
    spawns_norm = [(int(p[0]), int(p[1])) for p in spawns]

    return grid, estacion_dock, anaquel_home, spawns_norm

def celdas_adyacentes(celda: Celda) -> List[Celda]:
    x, y = celda
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def elegir_objetivo_adyacente(grid: np.ndarray, celda_anaquel: Celda) -> Optional[Celda]:
    """
    Dado que el anaquel es un obstáculo (no transitable), la "recolección" se modela
    desde una celda adyacente transitable. Se elige la primera celda adyacente válida.
    """
    alto, ancho = grid.shape

    def en_rango(x: int, y: int) -> bool:
        return 0 <= x < ancho and 0 <= y < alto

    for c in celdas_adyacentes(celda_anaquel):
        x, y = c
        if en_rango(x, y) and grid[y, x] in (LIBRE, ESTACION):
            return c
    return None

class SimAlmacen:
    def __init__(
        self,
        grid: np.ndarray,
        estacion_dock: Dict[int, Celda],
        anaquel_home: Dict[int, Celda],
        robots: int,
        puntos_spawn: List[Celda],
        pedidos: List[Pedido],
        seed: int,
    ):
        self.grid = grid
        self.estacion_dock = estacion_dock
        self.anaquel_home = anaquel_home
        self.pedidos = pedidos
        self.seed = seed
        self.tick = 0

        self.tabla_reservas = TablaReservas()

        # Inicializar robots en puntos spawn únicos
        if len(puntos_spawn) < robots:
            raise RuntimeError("No hay suficientes puntos de spawn para la cantidad de robots.")
        self.lista_robots: List[Robot] = []
        for i in range(robots):
            self.lista_robots.append(Robot(robot_id=i, pos=puntos_spawn[i]))

        # Reservar posiciones iniciales en tick 0
        for r in self.lista_robots:
            self.tabla_reservas.confirmar_espera(r.robot_id, r.pos, 0)

        # Métricas
        self.colisiones_vertice = 0
        self.intercambios_arista = 0
        self.conteo_deadlock = 0
        self.eventos_alto = 0

        # Gestión de pedidos
        self.pendientes: List[int] = []  # índices en self.pedidos
        self.no_liberados = sorted(range(len(self.pedidos)), key=lambda i: self.pedidos[i].tick_creacion)

    def _liberar_pedidos(self) -> None:
        while self.no_liberados and self.pedidos[self.no_liberados[0]].tick_creacion <= self.tick:
            idx = self.no_liberados.pop(0)
            self.pendientes.append(idx)

    def _asignar_pedidos(self) -> None:
        # Greedy: para cada robot inactivo, asignar el anaquel más cercano a una celda adyacente de recolección
        inactivos = [r for r in self.lista_robots if r.estado == "INACTIVO"]
        if (not inactivos) or (not self.pendientes):
            return

        for r in inactivos:
            if not self.pendientes:
                break

            mejor_idx: Optional[int] = None
            mejor_dist = 10**9

            # Considerar solo los primeros N en cola (localidad / performance)
            for pi in self.pendientes[: min(50, len(self.pendientes))]:
                p = self.pedidos[pi]
                anaquel = self.anaquel_home[p.anaquel_id]
                pickup = elegir_objetivo_adyacente(self.grid, anaquel)
                if pickup is None:
                    continue
                dist = abs(r.pos[0] - pickup[0]) + abs(r.pos[1] - pickup[1])
                if dist < mejor_dist:
                    mejor_dist = dist
                    mejor_idx = pi

            if mejor_idx is None:
                continue

            p = self.pedidos[mejor_idx]
            p.tick_asignacion = self.tick
            self.pendientes.remove(mejor_idx)

            r.pedido_id = p.pedido_id
            r.anaquel_home = self.anaquel_home[p.anaquel_id]
            r.estacion_dock = self.estacion_dock[p.estacion_id]
            r.estado = "A_RECOGER"

            pickup = elegir_objetivo_adyacente(self.grid, r.anaquel_home)
            if pickup is None:
                # No debería ocurrir si la lista de anaqueles es válida; revertir asignación
                r.estado = "INACTIVO"
                r.pedido_id = None
                r.anaquel_home = None
                r.estacion_dock = None
                p.tick_asignacion = None
                self.pendientes.append(mejor_idx)
                continue

            ruta = a_estrella(self.grid, r.pos, pickup)
            if ruta is None:
                # No alcanzable; revertir asignación
                r.estado = "INACTIVO"
                r.pedido_id = None
                r.anaquel_home = None
                r.estacion_dock = None
                p.tick_asignacion = None
                self.pendientes.append(mejor_idx)
                continue

            r.ruta = ruta
            r.idx_ruta = 0

    def _completar_pedido(self, pedido_id: int) -> None:
        # Buscar pedido por ID (O(n) con 600 pedidos es aceptable).
        for p in self.pedidos:
            if p.pedido_id == pedido_id:
                p.tick_completado = self.tick
                return

    def _planear_siguiente_tramo_si_llego(self, r: Robot) -> None:
        if (not r.ruta) or (r.idx_ruta != len(r.ruta) - 1):
            return

        if r.estado == "A_RECOGER":
            r.estado = "A_ESTACION"
            ruta = a_estrella(self.grid, r.pos, r.estacion_dock)
            if ruta is None:
                # reintentar después
                r.estado = "A_RECOGER"
                return
            r.ruta = ruta
            r.idx_ruta = 0

        elif r.estado == "A_ESTACION":
            r.estado = "RETORNO"
            pickup = elegir_objetivo_adyacente(self.grid, r.anaquel_home) if r.anaquel_home else None
            if pickup is None:
                r.estado = "A_ESTACION"
                return
            ruta = a_estrella(self.grid, r.pos, pickup)
            if ruta is None:
                r.estado = "A_ESTACION"
                return
            r.ruta = ruta
            r.idx_ruta = 0

        elif r.estado == "RETORNO":
            if r.pedido_id is not None:
                self._completar_pedido(r.pedido_id)

            r.estado = "INACTIVO"
            r.pedido_id = None
            r.anaquel_home = None
            r.estacion_dock = None
            r.ruta = []
            r.idx_ruta = 0

    def step(self) -> None:
        self._liberar_pedidos()
        self._asignar_pedidos()

        # Proponer movimientos
        propuestas: Dict[int, Celda] = {}
        for r in self.lista_robots:
            if r.estado != "INACTIVO":
                r.ticks_ocupado += 1

            self._planear_siguiente_tramo_si_llego(r)

            if r.estado == "INACTIVO" or (not r.ruta):
                propuestas[r.robot_id] = r.pos
                continue

            if r.idx_ruta < len(r.ruta) - 1:
                propuestas[r.robot_id] = r.ruta[r.idx_ruta + 1]
            else:
                propuestas[r.robot_id] = r.pos

        tick_siguiente = self.tick + 1
        movio_alguien = False

        # Orden determinista: robot_id ascendente
        for r in self.lista_robots:
            actual = r.pos
            siguiente = propuestas[r.robot_id]

            if siguiente == actual:
                self.tabla_reservas.confirmar_espera(r.robot_id, actual, tick_siguiente)
                continue

            if self.tabla_reservas.puede_moverse(actual, siguiente, tick_siguiente):
                self.tabla_reservas.confirmar_movimiento(r.robot_id, actual, siguiente, tick_siguiente)
                r.pos = siguiente
                r.idx_ruta += 1
                r.celdas_movidas += 1
                movio_alguien = True
            else:
                r.ticks_espera += 1
                self.eventos_alto += 1
                self.tabla_reservas.confirmar_espera(r.robot_id, actual, tick_siguiente)

        # Heurística de deadlock: nadie se movió pero al menos un robot está ocupado
        if (not movio_alguien) and any(r.estado != "INACTIVO" for r in self.lista_robots):
            self.conteo_deadlock += 1

        self.tick = tick_siguiente

    def run(self, ticks: int) -> None:
        for _ in range(ticks):
            self.step()

    def obtener_posiciones_robots(self) -> List[Celda]:
        return [r.pos for r in self.lista_robots]

    def obtener_estados_robots(self) -> List[str]:
        return [r.estado for r in self.lista_robots]

    def obtener_ids_robots(self) -> List[int]:
        return [r.robot_id for r in self.lista_robots]

    def metricas(self) -> Dict:
        completados = [p for p in self.pedidos if p.tick_completado is not None]
        completados_n = len(completados)

        tiempo_promedio_pedido = None
        if completados_n > 0:
            tiempo_promedio_pedido = float(np.mean([(p.tick_completado - p.tick_creacion) for p in completados]))

        utilizacion = [r.ticks_ocupado / max(1, self.tick) for r in self.lista_robots]
        ticks_espera = [r.ticks_espera for r in self.lista_robots]

        throughput = 0.0
        if self.tick > 0:
            throughput = completados_n / (self.tick / 1000.0)

        return {
            "seed": self.seed,
            "tick_final": self.tick,
            "robots": len(self.lista_robots),
            "pedidos_totales": len(self.pedidos),
            "pedidos_completados": completados_n,
            "tiempo_promedio_pedido_ticks": tiempo_promedio_pedido,
            "throughput_pedidos_por_1000_ticks": throughput,
            "tiempo_promedio_espera_ticks": float(np.mean(ticks_espera)) if ticks_espera else 0.0,
            "utilizacion_promedio": float(np.mean(utilizacion)) if utilizacion else 0.0,
            "colisiones_vertice": self.colisiones_vertice,
            "intercambios_arista": self.intercambios_arista,
            "deadlock": self.conteo_deadlock,
            "eventos_alto": self.eventos_alto,
            "distancia_total_celdas": int(sum(r.celdas_movidas for r in self.lista_robots)),
        }
