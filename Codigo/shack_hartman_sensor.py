import numpy as np
from scipy.ndimage import center_of_mass

def lens(field, l, focal, size, dx): 
    k = 2 * np.pi / l  # Número de onda
    N = field.shape[0]
    L = size  # Tamaño del plano de entrada (tamaño de la lente)
    x = np.linspace(0, L, int(L / dx))
    y = np.linspace(0, L, int(L / dx))
    X, Y = np.meshgrid(x, y)
    
    # Transferencia de fase de la lente
    lens_phase = np.exp(-1j * k * (X**2 + Y**2) / (2 * focal))
    
    # Campo de salida después de la lente
    U_out = field * lens_phase
    
    # Aplicar Transformada de Fourier para obtener el campo en el plano de imagen
    U_out_fft = np.fft.fftshift(np.fft.fft2(U_out))
    
    return U_out

def simulate_shack_hartmann(input_plane, wavelength, focal_length, lens_array_shape, lens_size, pixel_size):
    M, N = lens_array_shape
    a, b =np.shape(input_plane)
    lens_width = lens_height = lens_size
    pixel_width = pixel_height = pixel_size
    centroid_position= np.zeros((M,N,2))
    # Calculate the number of pixels per lens
    pixels_per_lens_x = int(lens_width / pixel_width)
    pixels_per_lens_y = int(lens_height / pixel_height)
    
    # Initialize the output image
    output_image = np.zeros((M * pixels_per_lens_y, N * pixels_per_lens_x))
    
    # Iterate over each lens in the array
    for i in range(M):
            for j in range(N):
                # Extract the subarray corresponding to the current lens
                subarray = input_plane[i * pixels_per_lens_y:(i + 1) * pixels_per_lens_y ,
                                    j * pixels_per_lens_x:(j + 1) * pixels_per_lens_x ]
                
                # Resize the subarray to match the size of the lens phase array
                subarray_resized = np.resize(subarray, (int(lens_height / pixel_height), int(lens_width / pixel_width)))
                
                # Propagate through the lens
                propagated_subarray = lens(subarray_resized, wavelength, focal_length, lens_size, pixel_size)
                
                # Calculate the centroid of the propagated subarray
                centroid = center_of_mass(propagated_subarray)

                
                # Check for NaN values in centroid and skip if found
                if not np.isnan(centroid).any():
                    # Mark the centroid on the output image
                    if 0 <= int(i * pixels_per_lens_y + centroid[0]) < output_image.shape[0] and 0 <= int(j * pixels_per_lens_x + centroid[1]) < output_image.shape[1]:
                        output_image[int(i * pixels_per_lens_y + centroid[0]), int(j * pixels_per_lens_x + centroid[1])] = 1
                        centroid_position[i,j]=[int(i * pixels_per_lens_y + centroid[0]), int(j * pixels_per_lens_x + centroid[1])]
    
    return output_image, centroid_position

def generate_grid(lens_array_shape, lens_size, pixel_size, field):
    M, N = lens_array_shape
    a, b =np.shape(field)
    lens_width = lens_height = lens_size
    pixel_width = pixel_height = pixel_size
    
    # Calculate the number of pixels per lens
    pixels_per_lens_x = int(lens_width / pixel_width)
    pixels_per_lens_y = int(lens_height / pixel_height)
    
    # Initialize the grid
    grid = np.zeros((M * pixels_per_lens_y, N * pixels_per_lens_x))
    
    # Draw horizontal lines
    for i in range(1, M):
        grid[i * pixels_per_lens_y - 1:i * pixels_per_lens_y + 1, :] = 0.5
    
    # Draw vertical lines
    for j in range(1, N):
        grid[:, j * pixels_per_lens_x - 1:j * pixels_per_lens_x + 1] = 0.5
    
    return grid

def displacement(original_centroid, aberrated_centroid):
    M, N, c = np.shape(original_centroid)
    d=[]
    d1=0
    for i in range(M):
        for j in range(N):
            d1=aberrated_centroid[i, j] - original_centroid[i, j]
            d.append(np.sqrt(d1[0]**2 + d1[1]**2))
    return d

def cost(individual, original_field, aberrated_field,wavelength, focal_length, lens_array_shape, lens_size, pixel_size, size):
    '''
    individual debe de contener la infomación en este orden indice de refraccion, tamaña de la lente, grososr de la lente y focal de la lente
    '''
    
    wave, original_centroid= simulate_shack_hartmann(original_field,wavelength, focal_length, lens_array_shape, lens_size, pixel_size)

    n, thicknes, focal=individual

    #se propaga por la lente gruesa para eliminar la aberración
    thick_len=thick_lens(thicknes, focal, wavelength, aberrated_field, size, pixel_size, n)

    wave, new_centroid =simulate_shack_hartmann(thick_len, wavelength, focal_length, lens_array_shape, lens_size, pixel_size)
    #Se calcula el desplazamiento de cada centtroide
    displacement_radius= displacement(original_centroid, new_centroid)
    return np.sum(displacement_radius)

def thick_lens(thicknes, focal, wavelenght, field, size, dx, n):
    k = 2 * np.pi / wavelenght  # Número de onda
    N = field.shape[0]
    L = size  # Tamaño del plano de entrada (tamaño de la lente)
    x = np.linspace(0, L, int(L / dx))
    y = np.linspace(0, L, int(L / dx))
    X, Y = np.meshgrid(x, y)
    
    # Transferencia de fase de la lente
    lens_phase = np.exp(-1j * k * (X**2 + Y**2) / (2 * focal)) * np.exp(1j *n* k * thicknes)
    
    # Campo de salida después de la lente
    U_out = field * lens_phase
    
    # Aplicar Transformada de Fourier para obtener el campo en el plano de imagen
    U_out_fft = np.fft.fftshift(np.fft.fft2(U_out))
    U_out=np.fft.fftshift(np.fft.ifft2(U_out_fft)) #Se calcula la inversa para asi simular un sistema 4f
    return U_out