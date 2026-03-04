#!/usr/bin/env python3
# Genera pedidos discretos para el benchmark de robots de almacén
import argparse
import json
import os
import numpy as np
from out_paths import asegurar_dirs_de_salidas

def _ruta_por_escenario(escenario: str, nombre_archivo: str) -> str:
    # Centraliza convención outputs/<escenario>/<archivo>
    return os.path.join("outputs", escenario, nombre_archivo)

def main():
    parser = argparse.ArgumentParser()

    # Un burst representa un pico temporal de demanda, donde múltiples pedidos
    # se generan en un intervalo corto para evaluar el comportamiento del sistema
    # bajo carga transitoria.
    
    # Parámetros base
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pedidos", type=int, default=600)
    parser.add_argument(
        "--burst",
        action="store_true",
        help=(
            "Si se activa, el 70% de los pedidos se crean temprano "
            "(ticks 0..2000) y el resto se distribuye hasta el tick 10000."
        ),
    )

    # Nuevo estándar
    parser.add_argument(
        "--escenario",
        type=str,
        default="seed42",
        help="Nombre del escenario. Lee/escribe en outputs/<escenario>/",
    )

    # Overrides (compatibilidad / modo avanzado)
    parser.add_argument(
        "--archivo_estaciones",
        type=str,
        default=None,
        help="(Opcional) Ruta explícita a estaciones.json. Si se omite, se usa outputs/<escenario>/estaciones.json",
    )
    parser.add_argument(
        "--archivo_anaqueles",
        type=str,
        default=None,
        help="(Opcional) Ruta explícita a anaqueles.json. Si se omite, se usa outputs/<escenario>/anaqueles.json",
    )
    parser.add_argument(
        "--salida",
        type=str,
        default=None,
        help="(Opcional) Ruta explícita de salida para pedidos.json. Si se omite, se usa outputs/<escenario>/pedidos.json",
    )

    args = parser.parse_args()

    # Resolver rutas estándar
    ruta_estaciones = args.archivo_estaciones or _ruta_por_escenario(args.escenario, "estaciones.json")
    ruta_anaqueles = args.archivo_anaqueles or _ruta_por_escenario(args.escenario, "anaqueles.json")
    ruta_salida = args.salida or _ruta_por_escenario(args.escenario, "pedidos.json")

    # Asegurar carpeta de salida (y cualquier subcarpeta)
    asegurar_dirs_de_salidas([ruta_salida])

    rng = np.random.default_rng(args.seed)

    # Cargar estaciones y anaqueles
    with open(ruta_estaciones, "r", encoding="utf-8") as f:
        estaciones = json.load(f)

    with open(ruta_anaqueles, "r", encoding="utf-8") as f:
        anaqueles = json.load(f)

    ids_estacion = [e["estacion_id"] for e in estaciones]
    ids_anaquel = [a["anaquel_id"] for a in anaqueles]

    pedidos = []

    for i in range(args.pedidos):
        estacion_id = int(rng.choice(ids_estacion))
        anaquel_id = int(rng.choice(ids_anaquel))

        if args.burst:
            if rng.random() < 0.70:
                tick_creacion = int(rng.integers(0, 2001))
            else:
                tick_creacion = int(rng.integers(0, 10001))
        else:
            tick_creacion = 0

        pedidos.append(
            {
                "pedido_id": i,
                "anaquel_id": anaquel_id,
                "estacion_id": estacion_id,
                "tick_creacion": tick_creacion,
            }
        )

    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "pedidos": pedidos,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[OK] Escenario: {args.escenario}")
    print(f"[OK] Estaciones: {ruta_estaciones}")
    print(f"[OK] Anaqueles : {ruta_anaqueles}")
    print(f"[OK] Salida   : {ruta_salida}")
    print(f"[OK] Se generaron {len(pedidos)} pedidos")

if __name__ == "__main__":
    main()
