### CALCULATION, FORMULATION, AND VISUALIZATION OF 1st ORDER REACTION


import json
import tkinter as tk
from tkinter import messagebox
import sys
import math
import os
import numpy as np
from matplotlib.widgets import Slider

# Simple Tkinter interface to input numeric values for:
# rho_b, R_o, b, C_Ag, k_g, D_e, k_s
# Saves values to a JSON file and prints them.


def _build_and_run_ui():
    fields = [
        ("rho_b", "20671.25"),
        ("R_o", "0.006"),
        ("b", "4"),
        ("P", "101400"),
        ("R", "8.314"),
        ("T", "1175.15"),
        ("k_g", "0.4"),
        ("tortuosity", "2"),
        ("porosity", "0.37"),
        ("k_s", "0.01311"),
    ]

    root = tk.Tk()
    root.title("Input parameters")
    entries = {}

    def _parse_float(s):
        try:
            return float(s.replace(",", "."))
        except Exception:
            raise ValueError(f"Invalid numeric value: {s}")

    for r, (name, default) in enumerate(fields):
        lbl = tk.Label(root, text=name)
        lbl.grid(row=r, column=0, padx=6, pady=4, sticky="w")
        var = tk.StringVar(value=default)
        ent = tk.Entry(root, textvariable=var, width=20)
        ent.grid(row=r, column=1, padx=6, pady=4)
        if name == "R":
            ent.config(state="readonly")
        entries[name] = var

    def save():
        try:
            params = {}
            for name in entries:
                val = entries[name].get().strip()
                if val == "":
                    raise ValueError(f"Value for '{name}' is required.")
                f = _parse_float(val)
                params[name] = f

            # basic domain checks
            if not (0.0 <= params["porosity"] <= 1.0):
                raise ValueError("porosity must be between 0 and 1")
            if not (0.0 < params["tortuosity"]):
                raise ValueError("tortuosity must be > 0")

            # save to JSON in current directory
            out_path = "params.json"
            with open(out_path, "w") as fh:
                json.dump(params, fh, indent=2)

            print(json.dumps(params, indent=2))
            messagebox.showinfo("Saved", f"Parameters saved to {out_path}")
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))

    btn_save = tk.Button(root, text="Save", command=save)
    btn_save.grid(row=len(fields), column=0, padx=6, pady=8, sticky="ew")

    btn_quit = tk.Button(root, text="Quit", command=root.destroy)
    btn_quit.grid(row=len(fields), column=1, padx=6, pady=8, sticky="ew")

    root.mainloop()

if __name__ == "__main__":
    _build_and_run_ui()


### input values to equation

import matplotlib.pyplot as plt

# load parameters saved by the UI
try:
    with open("params.json", "r") as fh:
        params = json.load(fh)
except Exception as e:
    raise SystemExit(f"Failed to load params.json: {e}")

# ensure required parameters are present
for key in ("rho_b", "R_o", "P", "R", "T", "k_s"):
    if key not in params:
        raise SystemExit(f"Missing parameter '{key}' in params.json")

rho_b = float(params["rho_b"])
R_o = float(params["R_o"])
P = float(params["P"])
R = float(params["R"])
T = float(params["T"])
k_s = float(params["k_s"])

# coefficient in the formula
C = ((rho_b * R_o * R * T) / (3 * 0.15969 * k_s * P))

# r from R_o down to 0 and compute t (descending r so plot shows r reducing):
r = np.linspace(R_o, 0.0, 500)

# time increases as radius decreases
# Kinetic equation for 1st order reaction
t = C * (1 - 3 * (r / R_o)**(2/3) + 2 * (r / R_o))

plt.figure(figsize=(7,4.5))
plt.plot(t, r, color="tab:blue", lw=2)
plt.xlabel("time, t(s)")
plt.ylabel("radius, r(m)")
plt.title("Reaction: t vs r")
plt.grid(True)
plt.tight_layout()
plt.savefig("t_vs_r.png", dpi=150)
plt.show()


### advanced visualization of t vs r in spherical shape

# advanced spherical 3D visualization with time slider

# uses R_o and C defined earlier in the file
phi = np.linspace(0, 2 * np.pi, 80)
theta = np.linspace(0, np.pi, 40)
phi, theta = np.meshgrid(phi, theta)

def sphere_coords(radius):
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return x, y, z

# initial values
t0 = 0.0
r0 = R_o * (max(0.0, 1.0 - t0 / C) ** 3)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
ax.set_title("Shrinking Sphere: radius vs time")
ax.set_box_aspect([1, 1, 1])  # nice equal aspect (matplotlib >=3.3)

# plot initial sphere
X, Y, Z = sphere_coords(r0)
surf = ax.plot_surface(X, Y, Z, cmap="magma_r", edgecolor="none", alpha=0.9)

# text showing current time and radius
info_txt = ax.text2D(0.02, 0.95, f"t = {t0:.4f} s\nr = {r0:.6e} m", transform=ax.transAxes, fontsize=14)

# set limits so sphere stays centered and scaled
lim = R_o * 1.05
ax.set_xlim3d(-lim, lim)
ax.set_ylim3d(-lim, lim)
ax.set_zlim3d(-lim, lim)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")

# Slider axis
slider_ax = fig.add_axes([0.15, 0.03, 0.7, 0.03])
time_slider = Slider(slider_ax, "t (s)", valmin=0.0, valmax=C, valinit=t0, valfmt="%.6g")

def update(val):
    tval = time_slider.val
    frac = max(0.0, 1.0 - tval / C)
    rnew = R_o * (frac ** 3)
    # clear and redraw sphere
    ax.cla()
    X, Y, Z = sphere_coords(rnew)
    ax.plot_surface(X, Y, Z, cmap="magma_r", edgecolor="black", alpha=0.9)
    # update axes, limits, labels and text
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim3d(-lim, lim)
    ax.set_ylim3d(-lim, lim)
    ax.set_zlim3d(-lim, lim)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Shrinking Sphere: radius vs time", fontsize=19)
    info_txt.set_text(f"t = {tval:.6g} s\nr = {rnew:.6e} m")
    # redraw the 2D text onto the cleared axes
    ax.text2D(0.05, 0.95, info_txt.get_text(), transform=ax.transAxes, fontsize=14)
    fig.canvas.draw_idle()

time_slider.on_changed(update)

plt.tight_layout()
plt.show()

