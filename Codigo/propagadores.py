import numpy as np

def angular_espectrum(field, z, l, dx, dy):
    '''
	field es el camnpo de entrada;
    z es la distancia de propagación en metros;
    l es la longitud de onda en metros;
    dx y dy es el tamaño de la malla en metros;
	'''
    [N,M]=np.shape(field)
    p=np.arange(0,N,1)
    q=np.arange(0,M,1)
    n,m=np.meshgrid(p - N/2, q - M/2)
    dfx=1/(dx*N)
    dfy=1/(dy*M)

    ##Factor de fase
    t1=1/l
    fx=n*dfx
    fy=m*dfy
    phase=np.exp(2j*np.pi*z*np.sqrt(t1**2 -fx**2 - fy**2))

    ##Transformada de fourier
    Tf = np.fft.fftshift(np.fft.fft2(field))

    a_s=Tf*phase

    output= np.fft.ifft2(a_s)
    return output

def fresnel(field, z , l, dx, dy):
    '''
	field es el campo de entrada;
z es la distancia de propagación en metros;
l es la longitud de onda en metros;
dx y dy es el tamaño de la malla en metros;
	'''
    N, M =np.shape(field)
    p=np.arange(0,N,1)
    q=np.arange(0,M,1)
    n,m=np.meshgrid(p - (N/2), q - (M/2), indexing='xy')

    dxout=(l*z)/(N*dx)
    dyout=(l*z)/(M*dy)

    t1=1/l
    t2=1/(l*z)

    z_phase=np.exp2((1j * 2 * np.pi * z * t1))/(1j*l*z)

    out_phase=np.exp2((1j*np.pi*t2)*(np.power(n*dxout, 2) + np.power(m*dyout, 2)))
    in_phase=np.exp2((1j*np.pi*t2)*(np.power(n*dx, 2) + np.power(m*dy, 2)))

    U=field*in_phase
    U=np.fft.fftshift(U)
    U=np.fft.fft2(U)
    U=np.fft.fftshift(U)

    out=z_phase*out_phase*dx*dy*U

    return out

def z_c(M, dx, l):
    '''
    Con esta funcion se calcula la distancia minima para el propagador a usar
    M es el tamaño de la imagen en pixeles ej: 500x500, 1024x1024, etc;
    dx es el tamaño del pixel ;
    l es la longitud de onda ;
    '''
    zc = (M * dx**2) / l
    # Retornar el valor en notación científica
    # print(f'{zc:.2e}')
    return zc