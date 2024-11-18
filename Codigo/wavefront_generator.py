import numpy as np 

def plane_wavefront(shape):
    plane=np.ones(shape)
    return plane

def spherical_wavefront(shape, wavelength):
    N = shape[0]
    M = shape[1]
    k = 2 *np.pi / wavelength

    #Se genera el tamaño del frente de onda en coordenadas cartesianas
    x = np.linspace(-N/2, N/2, N)
    y = np.linspace(-M/2, M/2, M)
    X,Y = np.meshgrid(x, y)

    #Se convierte a coordenadas esfericas para generar el frente de onda
    r=np.sqrt(X**2 + Y**2)
    wavefront=np.exp(1j * k * r) / r

    return  wavefront

def cilindrical_wavefront(shape, wavelength):
    N = shape[0]
    M = shape[1]
    k = 2 *np.pi / wavelength

    #Se genera el tamaño del frente de onda en coordenadas cartesianas
    x = np.linspace(-N/2, N/2, N)
    y = np.linspace(-M/2, M/2, M)
    X,Y = np.meshgrid(x, y)

    #Se convierte a coordenadas esfericas para generar el frente de onda
    r=np.sqrt(X**2)
    wavefront=np.exp(1j * k * r) / np.sqrt(r)

    return  wavefront
