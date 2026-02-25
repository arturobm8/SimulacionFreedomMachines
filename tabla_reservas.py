#!/usr/bin/env python3
from typing import Dict, Tuple

Celda = Tuple[int, int]

class TablaReservas:
    """
    Tabla de reservas para movimiento multi-robot en tiempo discreto sobre un grid.

    - Reservas de celdas: evitan colisiones de vértice
      (dos robots ocupando la misma celda en el mismo tick).
    - Reservas de aristas: evitan colisiones por intercambio (swap)
      (A->B y B->A en el mismo tick).

    Convenciones:
    - tick_siguiente es el tick en el que el robot ocupará la celda destino.
    - Se reserva la celda destino en tick_siguiente.
    - Se reserva la arista dirigida (actual -> siguiente) en tick_siguiente.
    - Para evitar swaps, se rechaza el movimiento si la arista opuesta
      (siguiente -> actual) ya está reservada en tick_siguiente.
    """

    def __init__(self) -> None:
        # (x, y, t) -> robot_id
        self.reserva_celdas: Dict[Tuple[int, int, int], int] = {}
        # (x1, y1, x2, y2, t) -> robot_id  (reserva de arista dirigida)
        self.reserva_aristas: Dict[Tuple[int, int, int, int, int], int] = {}

    def celda_libre(self, celda: Celda, tick: int) -> bool:
        x, y = celda
        return (x, y, tick) not in self.reserva_celdas

    def reservar_celda(self, robot_id: int, celda: Celda, tick: int) -> None:
        x, y = celda
        self.reserva_celdas[(x, y, tick)] = robot_id

    def arista_libre(self, a: Celda, b: Celda, tick: int) -> bool:
        x1, y1 = a
        x2, y2 = b
        return (x1, y1, x2, y2, tick) not in self.reserva_aristas

    def reservar_arista(self, robot_id: int, a: Celda, b: Celda, tick: int) -> None:
        x1, y1 = a
        x2, y2 = b
        self.reserva_aristas[(x1, y1, x2, y2, tick)] = robot_id

    def puede_moverse(self, actual: Celda, siguiente: Celda, tick_siguiente: int) -> bool:
        """
        Retorna True si el movimiento actual -> siguiente es factible en tick_siguiente.

        Verificaciones:
        1) La celda destino no está reservada en tick_siguiente (evita colisión de vértice).
        2) La arista opuesta (siguiente -> actual) no está reservada en tick_siguiente (evita swap).
        """
        if not self.celda_libre(siguiente, tick_siguiente):
            return False
        if not self.arista_libre(siguiente, actual, tick_siguiente):
            return False
        return True

    def confirmar_movimiento(self, robot_id: int, actual: Celda, siguiente: Celda, tick_siguiente: int) -> None:
        """
        Confirma un movimiento reservando:
        - la celda destino en tick_siguiente
        - la arista dirigida (actual -> siguiente) en tick_siguiente

        Debe llamarse solo si puede_moverse(...) regresó True.
        """
        self.reservar_celda(robot_id, siguiente, tick_siguiente)
        self.reservar_arista(robot_id, actual, siguiente, tick_siguiente)

    def confirmar_espera(self, robot_id: int, actual: Celda, tick_siguiente: int) -> None:
        """Reserva permanecer en la celda actual durante tick_siguiente."""
        self.reservar_celda(robot_id, actual, tick_siguiente)
