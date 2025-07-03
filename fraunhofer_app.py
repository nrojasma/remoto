import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import fresnel

st.set_page_config(page_title="Simulador de Difracción", layout="wide")
st.title("🔬 Simulador de Difracción y Calculadora de Fresnel")

# ------------------------ MODO PRINCIPAL ------------------------
modo_principal = st.radio("¿Qué deseas hacer?", ["Calculadora de Coeficientes de Fresnel", "Simulación de Difracción"], horizontal=True)

if modo_principal == "Calculadora de Coeficientes de Fresnel":
    st.subheader("🔍 Calculadora de Coeficientes de Fresnel")

    n1 = st.number_input("Índice de refracción del medio 1 (n₁)", min_value=0.0, value=1.0, step=0.01)
    n2 = st.number_input("Índice de refracción del medio 2 (n₂)", min_value=0.0, value=1.5, step=0.01)
    theta_i_deg = st.slider("Ángulo de incidencia (°)", 0.0, 90.0, 45.0)
    theta_i = np.radians(theta_i_deg)

    # Ley de Snell
    sin_theta_t = n1 / n2 * np.sin(theta_i)
    if abs(sin_theta_t) <= 1:
        theta_t = np.arcsin(sin_theta_t)

        # Coeficientes de reflexión y transmisión
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
        st.error("Condición de incidencia total interna: no hay transmisión.")

elif modo_principal == "Simulación de Difracción":
    # ------------------------ SELECCIONES PRINCIPALES ------------------------
    modo_dif = st.radio("Selecciona el tipo de difracción:", ["Fraunhofer", "Fresnel"], horizontal=True)
    modo_dim = st.radio("Selecciona la visualización:", ["1D", "2D"], horizontal=True)

    if modo_dim == "2D":
        tipo_apertura = st.radio("Tipo de apertura para difracción 2D:", ["Rectangular", "Circular"], horizontal=True)

    # ------------------------ PARÁMETROS COMUNES ------------------------
    wavelength = st.slider("Longitud de onda (nm)", 400, 700, 633) * 1e-9
    L = st.slider("Distancia a la pantalla (cm)", 10, 200, 100) / 100

    # ------------------------ DIFRACCIÓN DE FRAUNHOFER ------------------------
    if modo_dif == "Fraunhofer":
        st.subheader("Difracción de Fraunhofer")

        if modo_dim == "1D":
            N = st.slider("Número de rendijas", 1, 10, 4)
            a_um = st.slider("Separación entre rendijas a (µm)", 5, 50, 10)
            b_um = st.slider("Ancho de rendija b (µm)", 1, 20, 2)

            a = a_um * 1e-6
            b = b_um * 1e-6

            alpha = np.linspace(-6*np.pi, 6*np.pi, 5000)
            theta = (alpha / np.pi) * (wavelength / a)
            beta = (np.pi * b * theta) / wavelength
            gamma = (np.pi * a * theta) / wavelength

            beta[beta == 0] = 1e-20
            gamma[gamma == 0] = 1e-20

            I = (np.sin(beta) / beta)**2 * (np.sin(N * gamma) / np.sin(gamma))**2
            I /= np.max(I)
            envelope = (np.sin(beta) / beta)**2
            envelope /= np.max(envelope)

            fig, ax = plt.subplots()
            ax.plot(alpha / np.pi, I * 16, 'r', label='Interferencia total')
            ax.plot(alpha / np.pi, envelope * 16, 'b--', label='Envolvente')
            ax.set_xlabel(r'$\alpha$ (rad) / $\pi$')
            ax.set_ylabel(r'$I/I_0$')
            ax.set_title("Patrón de difracción 1D")
            ax.legend()
            st.pyplot(fig)

        elif modo_dim == "2D":
            N = 2048
            dx = 5e-6
            x = np.linspace(-N/2, N/2, N) * dx
            X, Y = np.meshgrid(x, x)

            if tipo_apertura == "Rectangular":
                apertura_x = st.slider("Ancho en X (µm)", 10, 200, 100) * 1e-6
                apertura_y = st.slider("Ancho en Y (µm)", 10, 200, 100) * 1e-6
                apertura = np.where((np.abs(X) < apertura_x/2) & (np.abs(Y) < apertura_y/2), 1, 0)
            else:
                radio = st.slider("Radio de apertura (mm)", 0.1, 1.0, 0.5, step=0.01) * 1e-3
                apertura = np.where(X**2 + Y**2 < radio**2, 1, 0)

            campo = np.fft.fftshift(np.fft.fft2(apertura))
            intensidad = np.abs(campo)**2
            intensidad /= np.max(intensidad)

            fx = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
            x_obs = fx * wavelength * L
            mask = (np.abs(x_obs) <= 10e-3)
            recorte = np.ix_(mask, mask)
            extent = [x_obs[mask][0]*1e3, x_obs[mask][-1]*1e3, x_obs[mask][0]*1e3, x_obs[mask][-1]*1e3]

            fig, ax = plt.subplots(figsize=(6,6))
            ax.imshow(intensidad[recorte], cmap='gray', extent=extent)
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
            ax.set_title("Patrón de difracción 2D (Fraunhofer)")
            st.pyplot(fig)

    # ------------------------ DIFRACCIÓN DE FRESNEL ------------------------
    elif modo_dif == "Fresnel":
        st.subheader("Difracción de Fresnel")

        if modo_dim == "1D":
            b_um = st.slider("Ancho de rendija b (µm)", 10, 200, 50)
            b = b_um * 1e-6

            x = np.linspace(-5e-3, 5e-3, 1000)
            s = np.sqrt(2 / (wavelength * L)) * x
            S, C = fresnel(s)
            I = (C**2 + S**2)
            I /= np.max(I)

            fig, ax = plt.subplots()
            ax.plot(x * 1e3, I, 'k')
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("Intensidad (normalizada)")
            ax.set_title("Difracción de Fresnel (ranura simple)")
            st.pyplot(fig)

        elif modo_dim == "2D":
            N = 1024
            dx = 2e-6
            x = np.linspace(-N/2, N/2, N) * dx
            X, Y = np.meshgrid(x, x)
            R2 = X**2 + Y**2

            if tipo_apertura == "Rectangular":
                apertura_x = st.slider("Ancho en X (µm)", 10, 200, 100) * 1e-6
                apertura_y = st.slider("Ancho en Y (µm)", 10, 200, 100) * 1e-6
                apertura = np.where((np.abs(X) < apertura_x/2) & (np.abs(Y) < apertura_y/2), 1, 0)
            else:
                radio = st.slider("Radio de apertura (m)", 0.001, 1.0, 0.01, step=0.001)
                apertura = np.where(R2 < radio**2, 1, 0)

            k = 2 * np.pi / wavelength
            r = np.sqrt(R2 + L**2)
            kernel = np.exp(1j * k * r) / r

            campo = apertura * kernel
            U = np.fft.fftshift(np.fft.fft2(campo))
            intensidad = np.abs(U)**2
            intensidad /= np.max(intensidad)

            fx = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
            x_obs = fx * wavelength * L
            extent = [x_obs[0]*1e3, x_obs[-1]*1e3, x_obs[0]*1e3, x_obs[-1]*1e3]

            fig, ax = plt.subplots(figsize=(6,6))
            ax.imshow(intensidad, cmap='gray', extent=extent)
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
            ax.set_title("Patrón de difracción 2D (Fresnel)")
            st.pyplot(fig)
