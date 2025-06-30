# fraunhofer_app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import fresnel

st.set_page_config(page_title="Simulador de Difracción", layout="wide")
st.title("🔬 Simulador de Difracción de Fraunhofer y Fresnel")

# ------------------------ SELECCIONES PRINCIPALES ------------------------
modo_dif = st.radio("Selecciona el tipo de difracción:", ["Fraunhofer", "Fresnel"], horizontal=True)
modo_dim = st.radio("Selecciona la visualización:", ["1D", "2D"], horizontal=True)

if modo_dim == "2D":
    tipo_apertura = st.radio("Tipo de apertura para difracción 2D:", ["Rectangular", "Circular"], horizontal=True)

# ------------------------ PARÁMETROS COMUNES ------------------------
wavelength = st.slider("Longitud de onda (nm)", 400, 700, 633) * 1e-9

# ------------------------ DIFRACCIÓN DE FRAUNHOFER ------------------------
if modo_dif == "Fraunhofer":
    st.subheader("Difracción de Fraunhofer")

    if modo_dim == "1D":
        L = st.slider("Distancia a la pantalla (m)", 1.0, 10.0, 1.0, step=0.1)
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
        L = st.slider("Distancia a la pantalla (mm)", 3000, 5000, 3951, step=1) / 1000  # convierte a metros
        dx = 5e-6  # 10 micras
        N = 2048   # resolución
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

        intensidad = np.abs(campo)**2
        intensidad /= np.max(intensidad)

        fig, ax = plt.subplots(figsize=(6,6))
        extent = [x_obs[mask][0]*1e3, x_obs[mask][-1]*1e3, x_obs[mask][0]*1e3, x_obs[mask][-1]*1e3]
        ax.imshow(intensidad[recorte], cmap='gray', extent=extent)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title("Patrón de difracción 2D (Fraunhofer)")
        st.pyplot(fig)

# ------------------------ DIFRACCIÓN DE FRESNEL ------------------------
elif modo_dif == "Fresnel":
    st.subheader("Difracción de Fresnel")

    L = st.slider("Distancia a la pantalla (m)", 0.01, 1.0, 0.1, step=0.01)

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