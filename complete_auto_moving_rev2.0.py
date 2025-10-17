import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

# =====================================================
# 1️⃣ Define physical constants for each phase
# =====================================================

# --- PHASE 1: chemical reaction control ---
rho_b1 = 20671.25
R_o1 = 0.006
P1 = 101400
R_g = 8.314
T1 = 1175.15
k_s1 = 0.01311
C1 = ((rho_b1 * R_o1 * R_g * T1) / (3 * 0.15969 * k_s1 * P1))

# --- PHASE 2: diffusion control ---
rho_b2 = 32900.11
R_o2 = 0.006
P2 = 101400
T2 = 1175.15
tortuosity2 = 2
porosity2 = 0.37
D_0 = 1e-4
D_e2 = D_0 * (porosity2 * T2**1.75) / (tortuosity2 * P2)
C2 = (rho_b2 * R_o2**2 * R_g * T2) / (6 * D_e2 * P2 * (1 - porosity2))

# --- PHASE 3: gas-film + diffusion + surface ---
rho_b3 = 20648.18
R_o3 = 0.006
b3 = 1
P3 = 101400
T3 = 1175.15
k_g3 = 0.4
tortuosity3 = 2
porosity3 = 0.37
k_s3 = 0.01311
D_e3 = D_0 * (porosity3 / tortuosity3)
C3 = rho_b3 * R_o3 / (b3 * (P3 / (R_g * T3)))

# =====================================================
# 2️⃣ Build r(t) relationships for each phase
# =====================================================

# Common radial fraction
r_frac = np.linspace(1.0, 0.1, 200)

# Invert t(r) equations numerically
def t1_r(r): return C1 * (1 - 3 * (r/R_o1)**(2/3) + 2 * (r/R_o1))
def t2_r(r): return C2 * (1 - 3 * (r/R_o2)**(2/3) + 2 * (r/R_o2))
def t3_r(r): 
    return C3 * (
        (1/(3*k_g3)) * (1 - r/R_o3)
        + (R_o3/(6*D_e3)) * (1 - 3*(r/R_o3)**2 + 2*(r/R_o3)**3)
        + (1/(3*k_s3)) * (1 - r/R_o3)
    )

t1 = t1_r(R_o1 * r_frac)
t2 = t2_r(R_o2 * r_frac)
t3 = t3_r(R_o3 * r_frac)

# Normalize to same total duration for animation
max_t = max(t1.max(), t2.max(), t3.max())
t_common = np.linspace(0, max_t, 200)

# Interpolated radii as function of common time
r1 = np.interp(t_common, t1, R_o1 * r_frac)
r2 = np.interp(t_common, t2, R_o2 * r_frac)
r3 = np.interp(t_common, t3, R_o3 * r_frac)

# =====================================================
# 3️⃣ Visualization setup
# =====================================================
def sphere_coords(radius, n=30):
    phi = np.linspace(0, np.pi, n)
    theta = np.linspace(0, 2*np.pi, n)
    phi, theta = np.meshgrid(phi, theta)
    X = radius * np.sin(phi) * np.cos(theta)
    Y = radius * np.sin(phi) * np.sin(theta)
    Z = radius * np.cos(phi)
    return X, Y, Z

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection="3d")
ax.set_title("Simultaneous Shrinking of 3 Reaction Layers", fontsize=14)
ax.set_box_aspect([1,1,1])

lim = R_o1 * 1.1
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim, lim)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")

# Initial spheres
X1,Y1,Z1 = sphere_coords(r1[0])
X2,Y2,Z2 = sphere_coords(r2[0])
X3,Y3,Z3 = sphere_coords(r3[0])

surf1 = ax.plot_surface(X1, Y1, Z1, color="deepskyblue", alpha=0.4, edgecolor="none")
surf2 = ax.plot_surface(X2, Y2, Z2, color="silver", alpha=0.4, edgecolor="none")
surf3 = ax.plot_surface(X3, Y3, Z3, color="darkslategrey", alpha=0.4, edgecolor="none")

time_text = ax.text2D(0.05, 0.92, "", transform=ax.transAxes, fontsize=12)

# =====================================================
# 4️⃣ Animation function
# =====================================================
def update(frame):
    # Clear only previous surface collections
    for coll in list(ax.collections):
        coll.remove()

    # Draw all three updated spheres
    X1,Y1,Z1 = sphere_coords(r1[frame])
    X2,Y2,Z2 = sphere_coords(r2[frame])
    X3,Y3,Z3 = sphere_coords(r3[frame])

    ax.plot_surface(X1, Y1, Z1, color="deepskyblue", alpha=0.2, edgecolor="black")
    ax.plot_surface(X2, Y2, Z2, color="lime", alpha=0.6, edgecolor="seagreen")
    ax.plot_surface(X3, Y3, Z3, color="red", alpha=0.15, edgecolor="red")

    time_text.set_text(f"t = {t_common[frame]:.2e} s")

    return []

# =====================================================
# 5️⃣ Run animation
# =====================================================
anim = animation.FuncAnimation(
    fig, update, frames=len(t_common), interval=30, blit=False, repeat=True
)

plt.show()
