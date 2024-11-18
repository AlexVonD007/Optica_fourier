from propagadores import angular_espectrum, fresnel, z_c
from pros_imagenes import read, mag, phase, intensity, show
from aberrations import zernike, aberrated_wavefront
from shack_hartman_sensor import generate_grid, simulate_shack_hartmann, thick_lens, cost, displacement
from wavefront_generator import plane_wavefront, spherical_wavefront, cilindrical_wavefront

import pygad
from pyswarm import pso
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



##Parametros para los frents de onda
shape=(500,500) # Tamaño de la imgen en pixeles asegurarse de que el tamaño sea de NxN 
wavelength=650e-6 # Longitud de onda en metros

## Parametros para el sensor Shack-Hartman
focal_length = 4.1e-3  # Distancia focal en metros
lens_array_shape = (5,5)  # Tamaño del arreglo de NxM lentes 
lens_size = 140e-6  # Tamaño de cada lente
pixel_size = 5.8e-6  # Tamaño del pixel en metros
size=shape[0]*pixel_size #Tamaño de la lente gruesa para corregir la aberración


plane_wave=plane_wavefront(shape)
shperical_wave=spherical_wavefront(shape, wavelength)
cilindrical_wave=cilindrical_wavefront(shape, wavelength)

## Generacion de las aberraciones
aberration=zernike(40,0, cilindrical_wave)

Z=aberrated_wavefront(aberration[0], plane_wave)

# Conversión a coordenadas cartesianas
X = aberration[1] * np.cos(aberration[2])
Y = aberration[1] * np.sin(aberration[2])

output_image, centroid = simulate_shack_hartmann(aberration[0], wavelength, focal_length, lens_array_shape, lens_size, pixel_size)
output_image2, centroid2 = simulate_shack_hartmann(plane_wave, wavelength, focal_length, lens_array_shape, lens_size, pixel_size)
grid=generate_grid(lens_array_shape, lens_size, pixel_size, Z)
# Gráfico del frente de onda aberrado y el sensor shack hartman
non_zero_values = centroid[centroid != 0]
non_zero_values_2=centroid2[centroid2 != 0]

# # print(np.shape(centroid))
# print("Valores distintos de cero imagen no aberrada:", len(non_zero_values_2)/2)
# print("Valores distintos de cero imagen aberrada:", len(non_zero_values)/2)

# print("Desplazamiento de los centroides", displacement(centroid2, centroid))




lb = [0.9, 1e-3, -52]  # Límite inferior para [n, thicknes, focal]
ub = [5, 10, 52]  # Límite superior para [n, thicknes, focal]

# Ejecutar el algoritmo PSO
xopt, fopt = pso(cost, lb, ub, args=(plane_wave, aberration[0], wavelength, focal_length, lens_array_shape, lens_size, pixel_size, size)
                 , swarmsize=30, maxiter=100)

print(f"Mejor solución: {xopt}")
print(f"Valor de la función de costo en la mejor solución: {fopt}")

n, thickness, focal= xopt
new_field=thick_lens(thickness, focal, wavelength, aberration[0], size, pixel_size, n)

output_image3, centroid3= simulate_shack_hartmann(new_field, wavelength, focal_length, lens_array_shape, lens_size, pixel_size)


plt.figure(figsize=(60, 60))

plt.subplot(2, 2, 1)
show(intensity(output_image2) + grid, 'Frente de onda de entrada')
for i in range(centroid2.shape[0]):
    for j in range(centroid2.shape[1]):
        plt.plot(centroid2[i, j, 1], centroid2[i, j, 0], 'ro')

plt.subplot(2, 2, 2)
show(intensity(output_image) + grid, 'Frente de onda aberrado')
for i in range(centroid.shape[0]):
    for j in range(centroid.shape[1]):
        plt.plot(centroid[i, j, 1], centroid[i, j, 0], 'ro')

plt.subplot(2,2,3)
show(intensity(output_image3) + grid, 'Frente de onda con elemento corrector')
for i in range(centroid3.shape[0]):
    for j in range(centroid3.shape[1]):
        plt.plot(centroid3[i, j, 1], centroid3[i, j, 0], 'ro')

plt.subplot(2,2,4)
plt.pcolormesh(X, Y, intensity(Z), shading='auto', cmap='turbo_r')
plt.colorbar(label='Intensidad')
plt.title('Aberración')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()