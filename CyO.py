# Instrucción para Binder/Jupyter: Descomentar la siguiente línea en la primera celda del Notebook
# %matplotlib widget

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================
# 1. FUNCIONES MATEMÁTICAS Y FÍSICAS
# ==========================================

def calcular_plf_db(theta_deg):
    theta_rad = np.radians(theta_deg)
    plf = np.maximum(np.cos(theta_rad)**2, 1e-10)
    return -10 * np.log10(plf)

def calcular_xpd_db(theta_deg):
    theta_rad = np.radians(theta_deg)
    sin_val = np.maximum(np.abs(np.sin(theta_rad)), 1e-5)
    return 20 * np.log10(np.abs(np.cos(theta_rad) / sin_val))

def fresnel(phi_deg, eps_r, eps_i):
    eps_complex = eps_r - 1j * eps_i 
    phi_rad = np.radians(phi_deg)
    raiz = np.sqrt(eps_complex - np.sin(phi_rad)**2, dtype=complex)
    R_te = (np.cos(phi_rad) - raiz) / (np.cos(phi_rad) + raiz)
    R_tm = (eps_complex * np.cos(phi_rad) - raiz) / (eps_complex * np.cos(phi_rad) + raiz)
    return R_te, R_tm

def stokes_completo(phi_deg, eps_r, eps_i, desfase_deg):
    R_te, R_tm = fresnel(phi_deg, eps_r, eps_i)
    E_te = R_te
    E_tm = R_tm
    
    desfase_rad = np.radians(desfase_deg)
    E_tm = E_tm * np.exp(1j * desfase_rad)
    
    s0 = np.abs(E_te)**2 + np.abs(E_tm)**2
    if np.isscalar(s0) and s0 == 0: s0 = 1e-10
    
    s1 = (np.abs(E_te)**2 - np.abs(E_tm)**2) / s0
    delta = np.angle(E_tm) - np.angle(E_te)
    s2 = 2 * np.abs(E_te) * np.abs(E_tm) * np.cos(delta) / s0
    s3 = 2 * np.abs(E_te) * np.abs(E_tm) * np.sin(delta) / s0
    return s1, s2, s3

# ==========================================
# 2. INTERFAZ UNIFICADA (LAYOUT PARA BINDER)
# ==========================================
# Se crea una figura maestra de alta resolución que contendrá todos los paneles
fig = plt.figure("Simulador de Polarización", figsize=(16, 8))
fig.canvas.manager.set_window_title('Análisis de Propagación y Esfera de Poincaré')

# -- COLUMNA IZQUIERDA: Antenas, Sliders y Botones --
ax_link = fig.add_axes([0.02, 0.75, 0.28, 0.20])
ax_link.set_title("Desalineación Mecánica")
ax_link.axis('off')

def dibujar_antena(ax, x, angle, label, color):
    r = 0.5
    dx = r * np.sin(np.radians(angle))
    dy = r * np.cos(np.radians(angle))
    ax.plot([x, x], [-0.8, 0.8], 'k--', alpha=0.2)
    ax.plot([x-dx, x+dx], [-dy, dy], color, linewidth=4, label=label)
    ax.text(x, 1, label, ha='center')

# Sliders (Posicionamiento absoluto para estabilidad web)
ax_s1 = fig.add_axes([0.08, 0.60, 0.22, 0.02])
ax_s2 = fig.add_axes([0.08, 0.54, 0.22, 0.02])
ax_s3 = fig.add_axes([0.08, 0.48, 0.22, 0.02])
ax_s4 = fig.add_axes([0.08, 0.42, 0.22, 0.02]) 
ax_s5 = fig.add_axes([0.08, 0.36, 0.22, 0.02]) 

s_theta = Slider(ax_s1, 'Theta (deg)', 0, 90, valinit=15)
s_phi = Slider(ax_s2, 'Incidencia', 0, 89, valinit=60)
s_eps = Slider(ax_s3, 'Permitividad', 1.0, 85.0, valinit=4.0)
s_epsi = Slider(ax_s4, 'Pérdidas', 0.0, 100.0, valinit=0.0)
s_desfase = Slider(ax_s5, 'Desfase Extra', -180.0, 180.0, valinit=0.0, color='purple')

# Botones de Escenarios
ax_b1 = fig.add_axes([0.02, 0.22, 0.13, 0.05])
ax_b2 = fig.add_axes([0.16, 0.22, 0.13, 0.05])
ax_b3 = fig.add_axes([0.02, 0.15, 0.13, 0.05])
ax_b4 = fig.add_axes([0.16, 0.15, 0.13, 0.05])
ax_b5 = fig.add_axes([0.09, 0.08, 0.13, 0.05])

btn1 = Button(ax_b1, 'Tierra Seca')
btn2 = Button(ax_b2, 'Asfalto')
btn3 = Button(ax_b3, 'Agua Mar')
btn4 = Button(ax_b4, 'Multitrayecto')
btn5 = Button(ax_b5, 'Ionósfera')

def set_tierra_seca(event): s_phi.set_val(60.0); s_eps.set_val(4.0); s_epsi.set_val(0.01); s_desfase.set_val(0.0)
def set_asfalto_mojado(event): s_phi.set_val(70.0); s_eps.set_val(8.0); s_epsi.set_val(5.0); s_desfase.set_val(0.0)
def set_agua_mar(event): s_phi.set_val(82.0); s_eps.set_val(81.0); s_epsi.set_val(60.0); s_desfase.set_val(0.0)
def set_multitrayecto(event): s_phi.set_val(45.0); s_eps.set_val(5.0); s_epsi.set_val(0.5); s_desfase.set_val(60.0)
def set_ionosfera(event): s_phi.set_val(30.0); s_eps.set_val(1.5); s_epsi.set_val(0.0); s_desfase.set_val(-75.0)

btn1.on_clicked(set_tierra_seca); btn2.on_clicked(set_asfalto_mojado); btn3.on_clicked(set_agua_mar)
btn4.on_clicked(set_multitrayecto); btn5.on_clicked(set_ionosfera)

# -- COLUMNA CENTRAL: Gráficas de Análisis --
angles = np.linspace(0, 90, 500)
ax_plf = fig.add_axes([0.38, 0.70, 0.25, 0.22])
ax_xpd = fig.add_axes([0.38, 0.40, 0.25, 0.22])
ax_refl = fig.add_axes([0.38, 0.10, 0.25, 0.22])

line_plf, = ax_plf.plot(angles, calcular_plf_db(angles), 'b')
p_plf, = ax_plf.plot([], [], 'ro')
ax_plf.set_title("PLF (dB)"); ax_plf.grid(True)

line_xpd, = ax_xpd.plot(angles, calcular_xpd_db(angles), 'g')
p_xpd, = ax_xpd.plot([], [], 'ro')
ax_xpd.set_title("XPD (dB)"); ax_xpd.grid(True)

line_te, = ax_refl.plot(angles, np.abs(fresnel(angles, 4, 0)[0]), 'b', label='TE')
line_tm, = ax_refl.plot(angles, np.abs(fresnel(angles, 4, 0)[1]), 'r', label='TM')
p_te, = ax_refl.plot([], [], 'bo'); p_tm, = ax_refl.plot([], [], 'ro')
ax_refl.set_title("Reflexión"); ax_refl.legend(); ax_refl.grid(True)

# -- COLUMNA DERECHA: Esfera de Poincaré --
ax_p = fig.add_axes([0.65, 0.10, 0.33, 0.80], projection='3d')

def setup_poincare(ax):
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:30j]
    ax.plot_wireframe(np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v), color='gray', alpha=0.1, linewidth=0.5)
    ax.plot(np.cos(np.linspace(0,2*np.pi,100)), np.sin(np.linspace(0,2*np.pi,100)), 0, 'k--', alpha=0.3)
    ax.set_xlabel('S1 (H/V)'); ax.set_ylabel('S2 (45/-45)'); ax.set_zlabel('S3 (RCP/LCP)')
    ax.view_init(elev=20, azim=45)

setup_poincare(ax_p)
point_p, = ax_p.plot([], [], [], 'ro', markersize=10)
vector_p = ax_p.quiver(0, 0, 0, 0, 0, 0, color='r', arrow_length_ratio=0.1)

# ==========================================
# 3. RUTINA DE ACTUALIZACIÓN MAESTRA
# ==========================================
def update(val):
    t, p = s_theta.val, s_phi.val
    e_r, e_i = s_eps.val, s_epsi.val 
    desf = s_desfase.val
    
    # Antenas
    ax_link.clear()
    ax_link.set_xlim(-1, 5); ax_link.set_ylim(-1.5, 1.5); ax_link.axis('off')
    ax_link.set_title("Desalineación Mecánica")
    dibujar_antena(ax_link, 0, 0, "Tx", 'blue')
    dibujar_antena(ax_link, 4, t, f"Rx ({t:.1f}°)", 'red')
    
    # Análisis
    p_plf.set_data([t], [calcular_plf_db(t)])
    p_xpd.set_data([t], [calcular_xpd_db(t)])
    r_te_arr, r_tm_arr = fresnel(angles, e_r, e_i)
    line_te.set_ydata(np.abs(r_te_arr)); line_tm.set_ydata(np.abs(r_tm_arr))
    r_te_op, r_tm_op = fresnel(p, e_r, e_i)
    p_te.set_data([p], [np.abs(r_te_op)]); p_tm.set_data([p], [np.abs(r_tm_op)])
    
    # Esfera
    s1, s2, s3 = stokes_completo(p, e_r, e_i, desf)
    point_p.set_data([s1], [s2]); point_p.set_3d_properties([s3])
    global vector_p
    vector_p.remove()
    vector_p = ax_p.quiver(0, 0, 0, s1, s2, s3, color='r', linewidth=2)
    ax_p.set_title(f"Esfera de Poincaré\nS1={s1:.2f}, S2={s2:.2f}, S3={s3:.2f}")
    
    # Actualización unificada
    fig.canvas.draw_idle()

s_theta.on_changed(update); s_phi.on_changed(update)
s_eps.on_changed(update); s_epsi.on_changed(update); s_desfase.on_changed(update)
update(None)

plt.show()