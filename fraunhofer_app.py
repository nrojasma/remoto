import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import fresnel

st.set_page_config(page_title="Simulador de Difracción y Fresnel", layout="wide")
st.title("🔬 Calculadora de Fresnel y Simulador de Difracción 2D")

# ------------------ MODO PRINCIPAL ------------------
modo_principal = st.radio("¿Qué deseas realizar?", ["Calculadora de Coeficientes de Fresnel", "Simulación de Difracción"], horizontal=True)

# ------------------ CALCULADORA DE COEFICIENTES ------------------
if modo_principal == "Calculadora de Coeficientes de Fresnel":
    st.subheader("Calculadora de Coeficientes de Fresnel")

    n1 = st.number_input("Índice de refracción del medio 1 (n₁)", min_value=0.0, value=1.0, step=0.01)
    n2 = st.number_input("Índice de refracción del medio 2 (n₂)", min_value=0.0, value=1.5, step=0.01)
    theta_i_deg = st.slider("Ángulo de incidencia (°)", 0.0, 90.0, 45.0)
    theta_i = np.radians(theta_i_deg)

    sin_theta_t = n1 / n2 * np.sin(theta_i)
    if abs(sin_theta_t) <= 1:
        theta_t = np.arcsin(sin_theta_t)
        rs = (n1 * np.cos(theta_i) - n2 * np.cos(theta_t)) / (n1 * np.cos(theta_i) + n2 * np.cos(theta_t))
        rp = (n2 * np.cos(theta_i) - n1 * np.cos(theta_t)) / (n2 * np.cos(theta_i) + n1 * np.cos(theta_t))
        Rs = rs**2
        Rp = rp**2
        Ts = 1 - Rs
        Tp = 1 - Rp

        st.markdown(f"**Ángulo de transmisión:** {np.degrees(theta_t):.2f}°")
        st.markdown(f"**Rs (⊥):** {Rs:.4f} | **Ts (⊥):** {Ts:.4f}")
        st.markdown(f"**Rp (∥):** {Rp:.4f} | **Tp (∥):** {Tp:.4f}")

        fig, ax = plt.subplots()
        ax.bar(["Rs", "Rp"], [Rs, Rp], color='red', label="Reflexión")
        ax.bar(["Ts", "Tp"], [Ts, Tp], bottom=[Rs, Rp], color='green', label="Transmisión")
        ax.set_ylabel("Coeficiente")
        ax.set_title("Coeficientes de Fresnel")
        ax.legend()
        st.pyplot(fig)
    else:
        st.error("Incidencia total interna: no hay transmisión.")

# ------------------ SIMULADOR DE DIFRACCIÓN ------------------
elif modo_principal == "Simulación de Difracción":
    modo_dif = st.radio("Tipo de difracción:", ["Fraunhofer", "Fresnel"], horizontal=True)
    tipo_apertura = st.radio("Tipo de apertura:", ["Rectangular", "Circular"], horizontal=True)
    wavelength = st.slider("Longitud de onda (nm)", 400, 700, 633) * 1e-9
    L = st.slider("Distancia a la pantalla (mm)", 100, 7000, 2000, step=10) / 1000

    # Parámetros espaciales
    N = 2048
    dx = 5e-6
    x = np.linspace(-N/2, N/2, N) * dx
    X, Y = np.meshgrid(x, x)
    R2 = X**2 + Y**2

    if tipo_apertura == "Rectangular":
        lado_x_mm = st.slider("Lado en X (mm)", 0.1, 10.0, 1.0, step=0.1)
        lado_y_mm = st.slider("Lado en Y (mm)", 0.1, 10.0, 4.0, step=0.1)
        lado_x = lado_x_mm * 1e-3
        lado_y = lado_y_mm * 1e-3
        apertura = np.where((np.abs(X) <= lado_x/2) & (np.abs(Y) <= lado_y/2), 1, 0)
    else:
        radio = st.slider("Radio de apertura (mm)", 0.1, 5.0, 0.5, step=0.1) * 1e-3
        apertura = np.where(R2 < radio**2, 1, 0)

    # ------------------ DIFRACCIÓN DE FRAUNHOFER ------------------
    if modo_dif == "Fraunhofer":
        st.subheader("Difracción de Fraunhofer 2D")

        campo = np.fft.fftshift(np.fft.fft2(apertura))
        intensidad = np.abs(campo)**2
        intensidad /= np.max(intensidad)
        intensidad = np.sqrt(intensidad)

        fx = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
        zoom_factor = 0.1  # acercar el patrón
        extent = [-zoom_factor*N*dx*1e3/2, zoom_factor*N*dx*1e3/2, -zoom_factor*N*dx*1e3/2, zoom_factor*N*dx*1e3/2]

        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(intensidad, cmap='inferno', extent=extent, vmin=0, vmax=0.1)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title("Patrón de difracción 2D (Fraunhofer)")
        st.pyplot(fig)

    # ------------------ DIFRACCIÓN DE FRESNEL ------------------
    elif modo_dif == "Fresnel":
        st.subheader("Difracción de Fresnel 2D")

        fase_cuadratica = np.exp(1j * (np.pi / (wavelength * L)) * (X**2 + Y**2))
        campo = apertura * fase_cuadratica
        U = np.fft.fftshift(np.fft.fft2(campo))
        intensidad = np.abs(U)**2
        intensidad /= np.max(intensidad)
        intensidad = np.sqrt(intensidad)

        fx = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
        zoom_factor = 0.05  # acercar patrón rectangular en Fresnel para mostrar pequeñas interferencias
        extent = [-zoom_factor*N*dx*1e3/2, zoom_factor*N*dx*1e3/2, -zoom_factor*N*dx*1e3/2, zoom_factor*N*dx*1e3/2]

        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(intensidad, cmap='gray', extent=extent, vmin=0, vmax=0.1)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title("Patrón de difracción 2D (Fresnel)")
        st.pyplot(fig)



