import numpy as np
from math import factorial

def radial_polynomial(n, m, rho):
    R = np.zeros_like(rho)
    for k in range((n - abs(m)) // 2 + 1):
        num=(-1)**k * factorial(n-k)
        dem = factorial(k) * factorial(0.5*n + 0.5*m -k) * factorial(0.5*n - 0.5*m -k)
        R += (num/dem) * rho ** (n - 2 * k)
    return R

# Función para calcular el polinomio de Zernike
def zernike(n, m, field):
    N, M = np.shape(field)
    rho_vals = np.linspace(0, 1, N)
    theta_vals = np.linspace(0, 2 * np.pi, N)
    rho, theta = np.meshgrid(rho_vals, theta_vals)


    R = radial_polynomial(n, abs(m), rho)
    if m >= 0:
        return R * np.cos(m * theta), rho, theta
    else:
        return R * np.sin(abs(m) * theta), rho, theta

def aberrated_wavefront(aberration, wavefront):
    aberrated_fourier=np.fft.fftshift(np.fft.fft2(aberration))
    wavefront_fourier=np.fft.fftshift(np.fft.fft2(wavefront))
    # out = aberrated_fourier*wavefront_fourier
    out = aberration + wavefront
    return out
