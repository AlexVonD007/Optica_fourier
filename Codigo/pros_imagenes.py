import cv2
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def read_image(filename, height, width):
    '''
    En filename solo poner el nombre de la imagen y el formato 
    En este formato filename='ejemplo.png', no filename=ejemplo.png 
    height es la altura de la imagen en pixeles
    width es el ancho de la imagen en pixeles
	'''
    img1=cv2.imread(filename) 
    
    ##Convertir a escala de grises

    img2=cv2.cvtColor(cv2.resize(img1,(height,width)),cv2.COLOR_BGR2GRAY)

    img3=np.array(img2)
    return np.array(img3)

def read(filename):
    img = Image.open(filename)
    img = ImageOps.grayscale(img)
    return np.array(img)

def mag(field):
    field_mag = np.abs(field)  
    return field_mag

def phase(field):
    phi =np.angle(field)
    return phi

def intensity(field):
    I=np.abs(field)
    I=I*I
    return I
def show(field, title, cmap='gray'):
    plt.imshow(field, cmap), plt.title(title)  # image in gray scale

    if title == None:
        plt.imshow(field, cmap)
    # plt.show()  # show image
    return 
