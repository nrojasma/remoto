# fraunhofer_app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simulador de Difracción de Fraunhofer", layout="wide")

st.title("🔭 Simulador de Difracción de Fraunhofer - Múltiples Rendijas")

st.markdown("""
Ajusta los parámetros del experimento para observar cómo cambia el patrón de difracción de Fraunhofer.  
Se muestra la interferencia total (rojo) y el patrón de difracción de una sola rendija como envolvente (azul punteado).
""")

# Sliders de parámetros
col1, col2 = st.columns(2)

with col1:
    N = st.slider("Número de rendijas (N)", min_value=1, max_value=10, value=4, step=1)
    a_um = st.slider("Separación entre rendijas **a** (µm)", min_value=5.0, max_value=50.0, value=10.0, step=1.0)

with col2:
    b_um = st.slider("Ancho de rendijas **b** (µm)", min_value=1.0, max_value=20.0, value=2.0, step=1.0)
    wavelength_nm = st.slider("Longitud de onda **λ** (nm)", min_value=400, max_value=700, value=633, step=10)

# Conversión de unidades
wavelength = wavelength_nm * 1e-9
a = a_um * 1e-6
b = b_um * 1e-6

# Eje angular
alpha = np.linspace(-6*np.pi, 6*np.pi, 5000)
theta = (alpha / np.pi) * (wavelength / a)

# Variables beta y gamma
beta = (np.pi * b * theta) / wavelength
gamma = (np.pi * a * theta) / wavelength

# Evitar divisiones por cero
beta[beta == 0] = 1e-20
gamma[gamma == 0] = 1e-20

# Intensidad
I = (np.sin(beta) / beta)**2 * (np.sin(N * gamma) / np.sin(gamma))**2
I = I / np.max(I)

envelope = (np.sin(beta) / beta)**2
envelope = envelope / np.max(envelope)

# Gráfica
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(alpha/np.pi, I * 16, 'r', label='Interferencia total')
ax.plot(alpha/np.pi, envelope * 16, 'b--', label='Envolvente (una rendija)')
ax.set_xlabel(r'$\alpha$ (rad) / $\pi$')
ax.set_ylabel(r'$I / I_0$ (W/cm$^2$)')
ax.set_title(f"N = {N} rendijas | a = {a_um:.1f} µm | b = {b_um:.1f} µm | λ = {wavelength_nm} nm")
ax.grid(True)
ax.legend()

st.pyplot(fig)

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("🔲 Difracción de Fraunhofer 2D - Apertura Rectangular")

# Parámetros
wavelength = st.slider("Longitud de onda (nm)", 400, 700, 633) * 1e-9
L = st.slider("Distancia a la pantalla (cm)", 10, 200, 100) / 100  # metros
apertura_x = st.slider("Ancho de apertura en X (µm)", 10, 200, 100) * 1e-6
apertura_y = st.slider("Ancho de apertura en Y (µm)", 10, 200, 100) * 1e-6

# Dimensiones de la apertura
N = 1024  # resolución
dx = 2e-6  # resolución espacial en plano de la rendija
x = np.linspace(-N/2, N/2, N) * dx
X, Y = np.meshgrid(x, x)

# Apertura rectangular
apertura = np.where((np.abs(X) < apertura_x/2) & (np.abs(Y) < apertura_y/2), 1, 0)

# Campo difractado: Fraunhofer ≈ FT de la apertura
campo = np.fft.fftshift(np.fft.fft2(apertura))
intensidad = np.abs(campo)**2
intensidad /= np.max(intensidad)  # Normalización

# Tamaño del plano de observación
fx = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
x_observ = fx * wavelength * L  # conversión a coordenadas físicas

# Mostrar imagen
fig, ax = plt.subplots(figsize=(6, 6))
extent = [x_observ[0]*1e3, x_observ[-1]*1e3, x_observ[0]*1e3, x_observ[-1]*1e3]
ax.imshow(intensidad, cmap='gray', extent=extent)
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_title("Patrón de difracción 2D (Fraunhofer)")
st.pyplot(fig)
