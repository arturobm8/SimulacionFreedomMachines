#!/usr/bin/env python3
import argparse
import json
import os
from typing import List, Tuple, Dict
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from out_paths import asegurar_dirs_de_salidas
from sim_core import Pedido, SimAlmacen, cargar_layout

# ======== FFMPEG PATH ============================================
# Ruta fija, eventualmente se agregará el switch --ffmpeg_path.
mpl.rcParams["animation.ffmpeg_path"] = r"/opt/homebrew/bin/ffmpeg"
# =================================================================

LIBRE = 0
ANAQUEL = 1
ESTACION = 2
BLOQUEADO = 3

Celda = Tuple[int, int]

def _ruta_por_escenario(escenario: str, nombre_archivo: str) -> str:
    return os.path.join("outputs", escenario, nombre_archivo)

def cargar_pedidos(ruta: str) -> List[Pedido]:
    """
    Lee pedidos en formato:
      {
        "seed": 42,
        "pedidos": [
          {"pedido_id":0, "anaquel_id":123, "estacion_id":5, "tick_creacion":17},
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

def graficar_layout(grid: np.ndarray, salida_png: str) -> None:
    """
    Visualización estática del layout.
    Escala de grises:
      LIBRE     -> blanco
      ESTACION  -> gris claro
      ANAQUEL   -> gris oscuro
      BLOQUEADO -> negro
    """
    img = np.zeros_like(grid, dtype=float)
    img[grid == LIBRE] = 1.0
    img[grid == ESTACION] = 0.7
    img[grid == ANAQUEL] = 0.35
    img[grid == BLOQUEADO] = 0.0

    plt.figure(figsize=(10, 7))
    plt.title("Layout CEDIS")
    plt.imshow(img, origin="upper", interpolation="nearest")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(salida_png, dpi=200)
    plt.close()

def guardar_heatmaps(grid: np.ndarray, visitas: np.ndarray, esperas: np.ndarray, prefijo: str = "heatmap") -> None:
    """
    Escribe:
      - <prefijo>_visitas.png : conteo de visitas a celdas (intensidad de tráfico)
      - <prefijo>_esperas.png : conteo de eventos de espera (bloqueo/congestión)
      - <prefijo>_ratio.png   : esperas/visitas (congestión relativa)
    """
    transitable = (grid == LIBRE) | (grid == ESTACION)

    # Heatmap visitas
    v = np.where(transitable, visitas, 0)
    plt.figure(figsize=(10, 7))
    plt.title("Heatmap: Visitas a celdas (intensidad de tráfico)")
    plt.imshow(v, origin="upper", interpolation="nearest")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{prefijo}_visitas.png", dpi=200)
    plt.close()

    # Heatmap esperas
    e = np.where(transitable, esperas, 0)
    plt.figure(figsize=(10, 7))
    plt.title("Heatmap: Esperas (congestión / bloqueo)")
    plt.imshow(e, origin="upper", interpolation="nearest")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{prefijo}_esperas.png", dpi=200)
    plt.close()

    # Heatmap ratio
    ratio = np.zeros_like(e, dtype=float)
    mask = (v > 0)
    ratio[mask] = e[mask] / v[mask]

    plt.figure(figsize=(10, 7))
    plt.title("Heatmap: Ratio Espera/Visita (congestión relativa)")
    plt.imshow(np.where(transitable, ratio, 0.0), origin="upper", interpolation="nearest")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{prefijo}_ratio.png", dpi=200)
    plt.close()

    print(f"[OK] Heatmaps escritos: {prefijo}_visitas.png, {prefijo}_esperas.png, {prefijo}_ratio.png")

def _mapa_colores_estados() -> Tuple[Dict[str, float], mpl.colors.Colormap, mpl.colors.Normalize]:
    """
    Colorea por estado con valores discretos y un cmap discreto.
    """
    estado_a_val = {
        "INACTIVO": 0.0,
        "A_RECOGER": 1.0,
        "A_ESTACION": 2.0,
        "RETORNO": 3.0,
    }

    colores = ["#777777", "#ff7f0e", "#1f77b4", "#2ca02c"]
    cmap = mpl.colors.ListedColormap(colores, name="estado_robots")
    norm = mpl.colors.Normalize(vmin=0.0, vmax=3.0)
    return estado_a_val, cmap, norm

def animar(
    grid: np.ndarray,
    sim: SimAlmacen,
    ticks: int,
    pasos_por_frame: int,
    salida_video: str,
    fps: int,
    prefijo_heatmap: str,
) -> None:
    alto, ancho = grid.shape

    visitas = np.zeros((alto, ancho), dtype=np.int32)
    esperas = np.zeros((alto, ancho), dtype=np.int32)

    img = np.zeros_like(grid, dtype=float)
    img[grid == LIBRE] = 1.0
    img[grid == ESTACION] = 0.7
    img[grid == ANAQUEL] = 0.35
    img[grid == BLOQUEADO] = 0.0

    fig = plt.figure(figsize=(10, 7))
    ax = plt.gca()
    ax.set_title("Simulación de flota de robots (CEDIS)")
    ax.imshow(img, origin="upper", interpolation="nearest")
    ax.set_xlim(-0.5, ancho - 0.5)
    ax.set_ylim(alto - 0.5, -0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    estado_a_val, cmap, norm = _mapa_colores_estados()

    posiciones = sim.obtener_posiciones_robots()
    xs = [p[0] for p in posiciones]
    ys = [p[1] for p in posiciones]
    estados = sim.obtener_estados_robots()
    cvals = [estado_a_val.get(s, 0.0) for s in estados]

    scat = ax.scatter(xs, ys, s=30, c=cvals, cmap=cmap, norm=norm)

    cbar = plt.colorbar(scat, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(["INACTIVO", "A_RECOGER", "A_ESTACION", "RETORNO"])

    texto = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    total_frames = max(1, ticks // pasos_por_frame)

    pedidos_completados = 0
    completados_prev = set()

    def update(frame_idx: int):
        nonlocal visitas, esperas, pedidos_completados, completados_prev

        for _ in range(pasos_por_frame):
            prev_pos = sim.obtener_posiciones_robots()
            sim.step()
            cur_pos = sim.obtener_posiciones_robots()

            for (x, y) in cur_pos:
                if 0 <= x < ancho and 0 <= y < alto:
                    visitas[y, x] += 1

            for p0, p1 in zip(prev_pos, cur_pos):
                if p0 == p1:
                    x, y = p1
                    if 0 <= x < ancho and 0 <= y < alto:
                        esperas[y, x] += 1

            nuevos = 0
            for ped in sim.pedidos:
                if ped.tick_completado is not None and ped.pedido_id not in completados_prev:
                    completados_prev.add(ped.pedido_id)
                    nuevos += 1
            pedidos_completados += nuevos

        posiciones = sim.obtener_posiciones_robots()
        xs = [p[0] for p in posiciones]
        ys = [p[1] for p in posiciones]
        scat.set_offsets(np.c_[xs, ys])

        estados = sim.obtener_estados_robots()
        cvals = [estado_a_val.get(s, 0.0) for s in estados]
        scat.set_array(np.array(cvals))

        tick = sim.tick
        pedidos_totales = len(sim.pedidos)
        throughput = 0.0
        if tick > 0:
            throughput = pedidos_completados / (tick / 1000.0)

        texto.set_text(
            f"tick={tick} | "
            f"completados={pedidos_completados}/{pedidos_totales} | "
            f"throughput={throughput:.1f} pedidos/1000t | "
            f"altos={sim.eventos_alto} | "
            f"deadlock={sim.conteo_deadlock}"
        )
        return scat, texto

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=1000 / fps,
        blit=False,
        repeat=False,
    )

    if salida_video.lower().endswith(".gif"):
        anim.save(salida_video, writer=animation.PillowWriter(fps=fps))
    else:
        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=fps, metadata={"artist": "sim_almacen"}, bitrate=1800)
        anim.save(salida_video, writer=writer)

    plt.close(fig)

    guardar_heatmaps(grid, visitas, esperas, prefijo=prefijo_heatmap)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--robots", type=int, default=20)
    ap.add_argument("--ticks", type=int, default=10000)
    ap.add_argument("--pasos_por_frame", type=int, default=25)
    ap.add_argument("--fps", type=int, default=20)

    # Escenario base
    ap.add_argument(
        "--escenario",
        type=str,
        default="seed42",
        help="Nombre del escenario. Lee de outputs/<escenario>/ y escribe resultados ahí.",
    )

    # Override opcional de ffmpeg (evita hardcode)
    ap.add_argument(
        "--ffmpeg_path",
        type=str,
        default=None, #"C:\ffmpeg\bin\ffmpeg.exe",
        help="Ruta a ffmpeg.exe (Windows). Si se omite, se usa el valor configurado en el script.",
    )

    # Overrides de entradas
    ap.add_argument("--layout", type=str, default=None)
    ap.add_argument("--estaciones", type=str, default=None)
    ap.add_argument("--anaqueles", type=str, default=None)
    ap.add_argument("--spawn", type=str, default=None)
    ap.add_argument("--pedidos", type=str, default=None)

    # Overrides de salidas
    ap.add_argument("--layout_png", type=str, default=None)
    ap.add_argument("--salida_video", type=str, default=None)
    ap.add_argument("--prefijo_heatmap", type=str, default=None)

    args = ap.parse_args()

    if args.ffmpeg_path:
        mpl.rcParams["animation.ffmpeg_path"] = args.ffmpeg_path

    # Entradas por escenario si no se dieron explícitamente
    ruta_layout = args.layout or _ruta_por_escenario(args.escenario, "layout.npy")
    ruta_estaciones = args.estaciones or _ruta_por_escenario(args.escenario, "estaciones.json")
    ruta_anaqueles = args.anaqueles or _ruta_por_escenario(args.escenario, "anaqueles.json")
    ruta_spawn = args.spawn or _ruta_por_escenario(args.escenario, "spawn.json")
    ruta_pedidos = args.pedidos or _ruta_por_escenario(args.escenario, "pedidos.json")

    # Salidas por escenario si no se dieron explícitamente
    ruta_layout_png = args.layout_png or _ruta_por_escenario(args.escenario, "layout.png")
    ruta_video = args.salida_video or _ruta_por_escenario(args.escenario, "simulacion.mp4")
    prefijo_heatmap = args.prefijo_heatmap or _ruta_por_escenario(args.escenario, "heatmap")

    asegurar_dirs_de_salidas([
        ruta_layout_png,
        ruta_video,
        f"{prefijo_heatmap}_visitas.png",
        f"{prefijo_heatmap}_esperas.png",
        f"{prefijo_heatmap}_ratio.png",
    ])

    grid, estacion_dock, anaquel_home, spawns = cargar_layout(
        ruta_layout, ruta_estaciones, ruta_anaqueles, ruta_spawn
    )
    pedidos = cargar_pedidos(ruta_pedidos)

    graficar_layout(grid, ruta_layout_png)
    print(f"[OK] Layout escrito en {ruta_layout_png}")

    sim = SimAlmacen(
        grid=grid,
        estacion_dock=estacion_dock,
        anaquel_home=anaquel_home,
        robots=args.robots,
        puntos_spawn=spawns,
        pedidos=pedidos,
        seed=args.seed,
    )

    animar(
        grid=grid,
        sim=sim,
        ticks=args.ticks,
        pasos_por_frame=args.pasos_por_frame,
        salida_video=ruta_video,
        fps=args.fps,
        prefijo_heatmap=prefijo_heatmap,
    )
    print(f"[OK] Video escrito en {ruta_video}")

if __name__ == "__main__":
    main()
