import random as rn

from matplotlib import pyplot
from matplotlib.patches import Rectangle, Circle

from Configuracion import Configuracion

def imprimir_Anchors(anchors = None, img = None, config = Configuracion):
    # Forma de imprimir
    mx,my = img.shape[0]//config.S, img.shape[1]//config.S
    color_malla=[0,0,0]
    img[:,::my,:] = color_malla
    img[::mx,:,:] = color_malla

    _, ax = pyplot.subplots(1, figsize=(16,16))
    height, width = img.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title("Anchors")

    for anchor in anchors:
        
        y1, x1, y2, x2 = anchor
        r,g,b = rn.random(), rn.random(), rn.random()
        amr = Rectangle((x1,y1), x2-x1, y2-y1, linewidth=1.5, alpha=0.7, linestyle='solid', edgecolor=(r,g,b), facecolor='none')
        ax.add_patch(amr)
        amr2 = Circle((x1+(0.5*(x2-x1)) , y1+(0.5*(y2-y1))), radius=0.5, linewidth=1.5, alpha=0.7, linestyle='solid', edgecolor=(r,g,b), facecolor='none')
        ax.add_patch(amr2)
            
    pyplot.imshow(img)
    pyplot.show()