#!/usr/bin/env python3
import argparse
import json
import os
from typing import List
from out_paths import asegurar_dirs_de_salidas
from sim_core import Pedido, SimAlmacen, cargar_layout

def _ruta_por_escenario(escenario: str, nombre_archivo: str) -> str:
    return os.path.join("outputs", escenario, nombre_archivo)

def cargar_pedidos(ruta: str) -> List[Pedido]:
    """
    Lee pedidos desde un JSON con formato:
    {
      "seed": 42,
      "pedidos": [
        {"pedido_id": 0, "anaquel_id": 123, "estacion_id": 5, "tick_creacion": 10},
        ...
      ]
    }
    """
    with open(ruta, "r", encoding="utf-8") as f:
        data = json.load(f)

    pedidos: List[Pedido] = []
    for p in data.get("pedidos", []):
        pedidos.append(
            Pedido(
                pedido_id=int(p["pedido_id"]),
                anaquel_id=int(p["anaquel_id"]),
                estacion_id=int(p["estacion_id"]),
                tick_creacion=int(p.get("tick_creacion", 0)),
            )
        )

    return pedidos

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--robots", type=int, default=20)
    parser.add_argument("--ticks", type=int, default=10000)

    # Nuevo estándar
    parser.add_argument(
        "--escenario",
        type=str,
        default="seed42",
        help="Nombre del escenario. Lee de outputs/<escenario>/ y escribe resultados ahí.",
    )

    # Overrides de entradas (modo avanzado / compatibilidad)
    parser.add_argument("--layout", type=str, default=None, help="(Opcional) Ruta explícita a layout.npy")
    parser.add_argument("--estaciones", type=str, default=None, help="(Opcional) Ruta explícita a estaciones.json")
    parser.add_argument("--anaqueles", type=str, default=None, help="(Opcional) Ruta explícita a anaqueles.json")
    parser.add_argument("--spawn", type=str, default=None, help="(Opcional) Ruta explícita a spawn.json")
    parser.add_argument("--pedidos", type=str, default=None, help="(Opcional) Ruta explícita a pedidos.json")

    # Salida
    parser.add_argument("--salida_metricas", type=str, default=None, help="(Opcional) Ruta explícita a metricas.json")

    args = parser.parse_args()

    # Resolver rutas por escenario si no se dieron explícitamente
    ruta_layout = args.layout or _ruta_por_escenario(args.escenario, "layout.npy")
    ruta_estaciones = args.estaciones or _ruta_por_escenario(args.escenario, "estaciones.json")
    ruta_anaqueles = args.anaqueles or _ruta_por_escenario(args.escenario, "anaqueles.json")
    ruta_spawn = args.spawn or _ruta_por_escenario(args.escenario, "spawn.json")
    ruta_pedidos = args.pedidos or _ruta_por_escenario(args.escenario, "pedidos.json")

    ruta_metricas = args.salida_metricas or _ruta_por_escenario(args.escenario, "metricas.json")
    asegurar_dirs_de_salidas([ruta_metricas])

    grid, estacion_dock, anaquel_home, spawns = cargar_layout(
        ruta_layout, ruta_estaciones, ruta_anaqueles, ruta_spawn
    )

    pedidos = cargar_pedidos(ruta_pedidos)

    sim = SimAlmacen(
        grid=grid,
        estacion_dock=estacion_dock,
        anaquel_home=anaquel_home,
        robots=args.robots,
        puntos_spawn=spawns,
        pedidos=pedidos,
        seed=args.seed,
    )

    sim.run(args.ticks)
    m = sim.metricas()

    with open(ruta_metricas, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)

    print(f"[OK] Escenario: {args.escenario}")
    print(f"[OK] Métricas : {ruta_metricas}")
    print("Benchmark finalizado. Métricas:")
    for k, v in m.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
