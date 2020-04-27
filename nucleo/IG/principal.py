import os
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk

canvas = None
x = 0
y = 0
start_x = 0
start_y = 0
rect = None

eRSelect1 = None
eRSelect2 = None

# Función para abrir labelImg-master
def iniciar_Etiquetador():
    dirr = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'labelImg-master'))
    os.system('/usr/bin/python3 '+ dirr+'/labelImg.py')

#Funciones agregar elementos al dataSet
def obtenerImagen():
    oI = Toplevel()
    oI.title("Obtener Imagen")
    oIFrame = Frame(oI)
    oIFrame.pack()
    labelOi = Label(oIFrame, text ="Origen de la imagen")
    labelOi.grid(row=0,column=0, pady=5, padx=5)
    oIElejir = Button(oIFrame, text="Elegir Imagen", command = obtenerImagen2)
    oIElejir.grid(row=1, column=0,pady=5, padx=5)
    oITomar = Button(oIFrame, text="Tomar fotografia")
    oITomar.grid(row=1, column=3,pady=5, padx=5)
    oI.mainloop()

def obtenerImagen2():
    global canvas
    eI = Toplevel()
    pic =  filedialog.askopenfilename(initialdir = "/",title = "Seleccionar Imagen",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    img = Image.open(pic)
    
    tOri = img.size
    tObj = (600,400)
    factor = min(float(tObj[1])/tOri[1], float(tObj[0])/tOri[0])
    width = int(tOri[0] * factor)
    height = int(tOri[1] * factor)

    rImg= img.resize((width, height), Image.ANTIALIAS)
    rImg = ImageTk.PhotoImage(rImg)

    canvas = Canvas(eI, width=tObj[0], height= tObj[1], cursor="cross")
    canvas.create_image(tObj[0]/2, tObj[1]/2, anchor=CENTER, image=rImg, tags="img")
    canvas.pack(fill=None, expand=False)
    canvas.bind("<ButtonPress-1>", on_button_press)
    canvas.bind("<B1-Motion>", on_move_press)
    canvas.bind("<ButtonRelease-1>", on_button_release)

    eIFrame = Frame(eI)
    eIFrame.pack(side = BOTTOM)
    eIAceptar = Button(eIFrame, text = "Guardar")
    eIAceptar.grid(row=0, column=2, pady=3, padx=10)
    eICancelar = Button(eIFrame, text = "Cancelar")
    eICancelar.grid(row=0, column=1, pady=3, padx=10)

    eI.mainloop()

#Eventos para craer rectangulo en canvas
def on_button_press(event):
    # save mouse drag start position
    global canvas, start_x, start_y, rect
    start_x = event.x
    start_y = event.y
    rect = canvas.create_rectangle(event.x, event.y, 1, 1)

def on_move_press(event):
    global canvas, start_x, start_y, rect
    curX, curY = (event.x, event.y)
    # expand rectangle as you drag the mouse
    canvas.coords(rect, start_x, start_y, curX, curY)

def on_button_release(event):
    eT = Toplevel()
    eT.title("Clase del objeto")
    eTFrame = Frame(eT)
    eTFrame.pack()
    eTLabel = Label(eTFrame, text ="Origen de la imagen")
    eTLabel.grid(row=0,column=0, pady=5, padx=5)
    eTSelect = Listbox(eTFrame)
    eTSelect.insert(0,"Persona")
    eTSelect.insert(1,"Perro")
    eTSelect.grid(row=1, column=0, columnspan =3, pady=5, padx=5)
    eTAceptar = Button(eTFrame, text = "Aceptar")
    eTAceptar.grid(row=2, column=2,pady=5, padx=5)
    eT.mainloop()


#Funciones para las opciones de directorio
#TODO: Borrar esta sección
def directorios():
    dir = Toplevel()
    dir.title("Administrar directorios")
    dirFrame = Frame(dir)
    dirFrame.pack()
    dirSelDat = Button(dirFrame, text="Seleccionar directorio de DataSet", command=elegir_Dataset)
    dirSelDat.grid(row=0, column=0, padx=15, pady=10)
    dirSelRed = Button(dirFrame, text="Seleccionar directorio de Redes", command=elegir_Modelo)
    dirSelRed.grid(row=1, column=0, padx=15, pady=10)
    dir.mainloop()
    
# Elegir directorio de dataset
def elegir_Dataset():
    directorio = filedialog.askdirectory(initialdir = "datasets", title="Seleccionar Directorio")
    dirrs = directorio.split('/')
    if dirrs[-1]=="":
        dirrs[-1]="Ninguno"
    etiqueta_21.config(text=dirrs[-1])
    
# Elegir archivo de modelo
def elegir_Modelo():
    archivo = filedialog.askopenfilename(initialdir = "modelos", title="Seleccionar Modelo", filetypes=(("Modelos y Pesos","*.h5"),))
    dirrs = archivo.split('/')
    if dirrs[-1]=="":
        dirrs[-1]="Ninguno"
    etiqueta_31.config(text=dirrs[-1])


#Funciones de REDES

def seleccionarRed():
    sR = Toplevel()
    sR.title("Seleccionar Red")
    sRFrame = Frame(sR)
    sRFrame.pack()
    sRLabel = Label (sRFrame, text="Seleccione la red a utilizar")
    sRLabel.grid(row=0, column=0, padx=5, pady=5)
    sRSelect = Listbox(sRFrame)
    sRSelect.insert(0, "Detector de personas")
    sRSelect.insert(1, "Detector de perros")
    sRSelect.grid(row=1, column=0, columnspan=3, pady=5, padx=5)
    sRAceptar = Button(sRFrame, text="Aceptar")
    sRAceptar.grid(row=2, column=1, pady=5, padx=5)
    sRCancelar = Button(sRFrame, text="Cancelar")
    sRCancelar.grid(row=2, column=0, pady=5, padx=5)
    sR.mainloop()

def entrenarRed():
    global eRSelect1, eRSelect2
    eR = Toplevel()
    eR.title("Entrenar Red Neuronal")
    eRFrame = Frame(eR)
    eRFrame.pack()
    eRLabel1 = Label(eRFrame, text="Clases en el DataSet")
    eRLabel1.grid(row=0, column=0, columnspan=3, padx=10, pady=5)
    eRLabel2 = Label(eRFrame, text="Clases para entrenar")
    eRLabel2.grid(row=0, column=3, columnspan=3, padx=10, pady=5)
    eRSelect1 = Listbox(eRFrame)
    eRSelect1.insert(0,"Persona")
    eRSelect1.insert(1,"Perro")
    eRSelect1.bind("<Double-Button-1>", pasarAB)
    eRSelect1.grid(row=1, column=0, columnspan=3, padx=10, pady=5)
    eRSelect2 = Listbox(eRFrame)
    eRSelect2.bind("<Double-Button-1>", pasarBA)
    eRSelect2.grid(row=1, column=3, columnspan=3, padx=10, pady=5)
    eRAceptar = Button(eRFrame, text= "Aceptar", command=nombrarNRed)
    eRAceptar.grid(row=2, column=3, columnspan=3, padx=10, pady=5)
    eRCancelar = Button(eRFrame, text= "Cancelar")
    eRCancelar.grid(row=2, column=0, columnspan=3, padx=10, pady=5)
    eR.mainloop()

def pasarAB(event):
    global eRSelect1, eRSelect2
    sle = eRSelect1.curselection()
    slr = eRSelect1.get(sle)
    eRSelect2.insert(sle,slr)
    eRSelect1.delete(sle)
    print(sle)
def pasarBA(event):
    global eRSelect1, eRSelect2
    sle = eRSelect2.curselection()
    slr = eRSelect2.get(sle)
    eRSelect1.insert(sle,slr)
    eRSelect2.delete(sle)
    print(sle)
def nombrarNRed():
    nNRed = Toplevel()
    nNRed.title("Nombre")
    nNRedFrame = Frame(nNRed)
    nNRedFrame.pack()
    nNRedLabel = Label(nNRedFrame, text="Nombre de la red neuronal:")
    nNRedLabel.grid(row=0, column=0, columnspan=2)
    nNRedInput = Entry(nNRedFrame, width=25)
    nNRedInput.grid(row=1, column=0, columnspan=3, padx=20, pady=5)
    nNRedAceptar = Button(nNRedFrame, text= "Aceptar", command=entrenar)
    nNRedAceptar.grid(row=2, column=1, padx=10, pady=5)
    nNRedCancelar = Button(nNRedFrame, text= "Cancelar")
    nNRedCancelar.grid(row=2, column=0, padx=10, pady=5)
    nNRed.mainloop()

def entrenar():
    train = Toplevel()
    train.title("Entrenamiento")
    trainFrame = Frame(train)
    trainFrame.pack()
    tFLabel0 = Label(trainFrame, text="Datos del Entrenamiento")
    tFLabel1 = Label(trainFrame, text="Epoca:")
    tFLabel2 = Label(trainFrame, text="1")
    tFLabel3 = Label(trainFrame, text="Precisión:")
    tFLabel4 = Label(trainFrame, text="0.26")
    tFLabel5 = Label(trainFrame, text="Error:")
    tFLabel6 = Label(trainFrame, text="0.74")
    tfDetener = Button(trainFrame,text="Detener")
    tFLabel0.grid(row=0,column=1, padx=10, pady=5)
    tFLabel1.grid(row=1,column=0,  padx=10, pady=5)
    tFLabel2.grid(row=1,column=2,  padx=10, pady=5)
    tFLabel3.grid(row=2,column=0,  padx=10, pady=5)
    tFLabel4.grid(row=2,column=2,  padx=10, pady=5)
    tFLabel5.grid(row=3,column=0,  padx=10, pady=5)
    tFLabel6.grid(row=3,column=2,  padx=10, pady=5)
    tfDetener.grid(row=4,column=1, padx=10, pady=5)
    train.mainloop()

def predecir1():
    pred = Toplevel()
    pred.title("Identificar Objetos")
    predFrame = Frame(pred)
    predFrame.pack()
    predLabel = Label(predFrame, text ="Origen de la imagen")
    predLabel.grid(row=0,column=0, pady=5, padx=5)
    predElejir = Button(predFrame, text="Elegir Imagen", command = obtenerImagen3)
    predElejir.grid(row=1, column=0,pady=5, padx=5)
    predTomar = Button(predFrame, text="Tomar fotografia")
    predTomar.grid(row=1, column=3,pady=5, padx=5)
    pred.mainloop()

def obtenerImagen3():
    eI = Toplevel()
    pic =  filedialog.askopenfilename(initialdir = "/",title = "Seleccionar Imagen",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    img = Image.open(pic)
    
    tOri = img.size
    tObj = (600,400)
    factor = min(float(tObj[1])/tOri[1], float(tObj[0])/tOri[0])
    width = int(tOri[0] * factor)
    height = int(tOri[1] * factor)

    rImg= img.resize((width, height), Image.ANTIALIAS)
    rImg = ImageTk.PhotoImage(rImg)

    canvax = Canvas(eI, width=tObj[0], height= tObj[1], cursor="cross")
    canvax.create_image(tObj[0]/2, tObj[1]/2, anchor=CENTER, image=rImg, tags="img")
    canvax.pack(fill=None, expand=False)

    eIFrame = Frame(eI)
    eIFrame.pack(side = BOTTOM)
    eIAceptar = Button(eIFrame, text = "Aceptar")
    eIAceptar.grid(row=0, column=2, pady=3, padx=10)
    eICancelar = Button(eIFrame, text = "Cancelar")
    eICancelar.grid(row=0, column=1, pady=3, padx=10)

    eI.mainloop()



# Ventana Principal
root = Tk()
root.title("Visión Artificial Basada en Redes Neuronales")

# Barra de Menú
barra_menu = Menu(root)
# Menu Dataset
menu_Dataset = Menu(barra_menu, tearoff=0)
menu_Dataset.add_command(label='Lanzar "LabelImg"',command=iniciar_Etiquetador)
menu_Dataset.add_command(label='Seleccionar DataSet',command=elegir_Dataset)
barra_menu.add_cascade(label="Dataset", menu=menu_Dataset)
# Menu Dataset
menu_Modelo = Menu(barra_menu, tearoff=0)
menu_Modelo.add_command(label='Seleccionar Modelo',command=elegir_Modelo)
barra_menu.add_cascade(label="Modelo", menu=menu_Modelo)

# Panel principal
contenedor = Frame(root)
contenedor.pack()


etiqueta_1 = Label(contenedor, text="Información:",
                   relief=FLAT, height=1,
                   font=("Helvetica","12","bold"))
etiqueta_1.grid(row=0, column=0, padx=10, sticky=W)

etiqueta_2 = Label (contenedor, text="Dataset seleccionado:",
                   relief=FLAT, height=1)
etiqueta_2.grid(row=1, column=0, padx=10, sticky=W)

etiqueta_21 = Label (contenedor, text="Ninguno",
                   relief=RIDGE, height=1, width=50)
etiqueta_21.grid(row=1, column=1, padx=10, sticky=W)

etiqueta_3 = Label (contenedor, text="Modelo seleccionado:",
                   relief=FLAT, height=1)
etiqueta_3.grid(row=2, column=0, padx=10, sticky=W)

etiqueta_31 = Label (contenedor, text="Ninguno",
                   relief=RIDGE, height=1, width=50)
etiqueta_31.grid(row=2, column=1, padx=10, sticky=W)

#Frame de administación del DataSet
# etiqueta_1 = Label(contenedor, text="Administrar DataSet",
#                    relief=FLAT, height=1,width=25)
# etiqueta_1.grid(row=0, column=0,padx =10)

# marco_1 = Frame(contenedor)
# marco_1.grid(row=1, column=0, rowspan=4)

# btn_Admin_Etiquetas = Button(marco_1, text="Administrar Etiquetas",
#                                      width=20, command=iniciar_Etiquetador)
# btn_Admin_Etiquetas.pack()

# btn_Elegir_Dataset = Button(marco_1, text="Elegir DataSet",
#                             width=20, command=iniciar_Etiquetador)
# btn_Elegir_Dataset.pack()


#Frame de uso de Redes Neuronales
# label2 = Label (contenedor, text = "Red Neuronal", relief = RAISED, height = 1, width =25 )
# label2.grid(row=0,column=1, padx = 10)

# f2 = Frame(contenedor)
# f2.grid(row=1,column=1, rowspan=4, pady=10)

# botonSelect = Button(f2,text ="Seleccionar", width = 20, command=seleccionarRed)
# botonSelect.pack()

# botonTrain = Button(f2,text ="Entrenar", width = 20, command=entrenarRed)
# botonTrain.pack()

# botonPredict = Button(f2,text ="Predecir", width = 20, command=predecir1)
# botonPredict.pack()

#Frame de Configuración
label3 = Label (contenedor, text = "Configuración", relief = RAISED, height = 1, width =25 )
label3.grid(row=0,column=2, padx = 10)

f3 = Frame(contenedor)
f3.grid(row=1,column=2, rowspan=4)

botonCDir = Button(f3,text ="Directorios", width = 20, command=directorios)
botonCDir.pack()

botonCAvan = Button(f3,text ="Avanzado", width = 20)
botonCAvan.pack()

root.config(menu=barra_menu)
root.mainloop()