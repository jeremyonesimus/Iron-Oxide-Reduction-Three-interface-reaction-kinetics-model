# corrected_shrinking_core.py
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.interpolate import interp1d      # requires scipy
import tkinter as tk
from tkinter import messagebox

# --- (Your UI builder left mostly unchanged) ---
def _build_and_run_ui():
    fields = [
        ("rho_b", "20648.18"),
        ("R_o", "0.006"),
        ("b", "1"),               # use b=1 for FeO -> Fe
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

            if not (0.0 <= params["porosity"] <= 1.0):
                raise ValueError("porosity must be between 0 and 1")
            if not (0.0 < params["tortuosity"]):
                raise ValueError("tortuosity must be > 0")

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

# Only run the UI when executed directly (keep for convenience)
if __name__ == "__main__":
    _build_and_run_ui()

# ---------------------------
# Load parameters (same as you did)
# ---------------------------
try:
    with open("params.json", "r") as fh:
        params = json.load(fh)
except Exception as e:
    raise SystemExit(f"Failed to load params.json: {e}")

for key in ("rho_b", "R_o", "P", "R", "T", "k_s", "b", "k_g", "porosity", "tortuosity"):
    if key not in params:
        raise SystemExit(f"Missing parameter '{key}' in params.json")

# --- parameter extraction ---
rho_b_mass = float(params["rho_b"])   # in kg/m3
M_FeO = 0.071844                      # kg/mol
rho_b = rho_b_mass / M_FeO            # convert to mol/m3
R_o = float(params["R_o"])
P = float(params["P"])
R = float(params["R"])
T = float(params["T"])
k_s = float(params["k_s"])
b = float(params["b"])
k_g = float(params["k_g"])
porosity = float(params["porosity"])
tortuosity = float(params["tortuosity"])

D_0 = 1e-4
D_e = D_0 * (porosity / tortuosity)

# correct coefficient
C = rho_b * R_o / (b * (P / (R * T)))   # = rho_b * R_o * R * T / (b * P)


# prepare r grid (from R_o down to near zero)
r = np.linspace(R_o, 1e-9, 500)   # avoid exact zero to prevent numerical issues
term_1 = r / R_o
term_2 = 1.0 - term_1

# original formula for t(r) you used
t = C * (
    (1/(3*k_g)) * (1 - r/R_o)
    + (R_o/(6*D_e)) * (1 - 3*(r/R_o)**2 + 2*(r/R_o)**3)
    + (1/(3*k_s)) * (1 - r/R_o)
)


# --- Basic sanity printouts to help debugging ---
print(f"D_e = {D_e:.3e} m^2/s")
print(f"C   = {C:.6g}")
print(f"t range: min={np.nanmin(t):.6g}  max={np.nanmax(t):.6g}")

# --- 2D plot (t on x, r on y as you wanted) ---
plt.figure(figsize=(7, 4.5))
plt.plot(t, r, lw=2)
plt.xlabel("time, t (s)")
plt.ylabel("radius, r (m)")
plt.title("Reaction: t vs r")
plt.grid(True)
plt.tight_layout()
plt.savefig("t_vs_r.png", dpi=150)
plt.show()

# --- 3D sphere visualization: use interpolation so r(t) matches the computed t(r) ---
# Note: t may not be monotonic in r due to formula; we must sort by t for interpolation
# Keep only finite values and sort by t ascending
mask = np.isfinite(t)
t_valid = t[mask]
r_valid = r[mask]

# sort by time (ascending)
order = np.argsort(t_valid)
t_sorted = t_valid[order]
r_sorted = r_valid[order]

# remove duplicates in t_sorted (interp1d doesn't like identical x's)
# We'll create a small unique set
_, unique_idx = np.unique(np.round(t_sorted, 12), return_index=True)  # rounding tolerance
t_unique = t_sorted[unique_idx]
r_unique = r_sorted[unique_idx]

# build interpolation from time -> radius
if len(t_unique) < 2:
    raise SystemExit("Not enough valid points in computed t(r) to interpolate r(t).")
r_of_t = interp1d(t_unique, r_unique, kind="linear", bounds_error=False,
                  fill_value=(R_o, 0.0))  # before earliest time -> R_o, after last time -> 0

# 3D plot with slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (keeps 3D backend)
phi = np.linspace(0, 2 * np.pi, 80)
theta = np.linspace(0, np.pi, 40)
phi, theta = np.meshgrid(phi, theta)

def sphere_coords(radius):
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return x, y, z

# initial time: set to min(t_unique)
t0 = float(t_unique[0])
r0 = float(r_of_t(t0))

fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection="3d")
ax.set_box_aspect([1, 1, 1])

lim = R_o * 1.05
ax.set_xlim3d(-lim, lim)
ax.set_ylim3d(-lim, lim)
ax.set_zlim3d(-lim, lim)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Shrinking Sphere: radius vs time")

X, Y, Z = sphere_coords(r0)
surf = ax.plot_surface(X, Y, Z, cmap="cubehelix", edgecolor="none", alpha=0.9)

info_txt = ax.text2D(0.02, 0.95, f"t = {t0:.6g} s\nr = {r0:.6e} m", transform=ax.transAxes, fontsize=12)

# slider spanning computed time range
t_min = float(np.min(t_unique))
t_max = float(np.max(t_unique))
slider_ax = fig.add_axes([0.15, 0.03, 0.7, 0.03])
time_slider = Slider(slider_ax, "t (s)", valmin=t_min, valmax=t_max, valinit=t0, valfmt="%.6g")

def update(val):
    tval = float(time_slider.val)
    rnew = float(r_of_t(tval))

    # remove previous surface safely (cross-version)
    for coll in list(ax.collections):
        coll.remove()

    # plot new sphere
    X, Y, Z = sphere_coords(rnew)
    ax.plot_surface(X, Y, Z, cmap="cubehelix", edgecolor="black", alpha=0.9)

    # update info text
    for txt in ax.texts:
        txt.remove()
    ax.text2D(0.02, 0.95, f"t = {tval:.6g} s\nr = {rnew:.6e} m",
              transform=ax.transAxes, fontsize=12)

    fig.canvas.draw_idle()


time_slider.on_changed(update)

plt.tight_layout()
plt.show()
