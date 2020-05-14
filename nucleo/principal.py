import os
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk

from Interface import Analizador, Entrenador


class Interfaz_Grafica:
    
    def __init__(self):
        # Atributos a usar
        self.archivo_P = "" # Archivo de pesos de red
        self.archivo_I = "" # Archivo de Imagen
        self.archivo_C = "" # Archivo de configuracion
        self.directorio_d ="" # Directorio del dataset
        # Ventana Principal
        self.root = Tk()
        self.root.title("Visión Artificial Basada en Redes Neuronales")

        # Barra de Menú
        self.barra_menu = Menu(self.root)
        # Menu Dataset
        self.menu_Dataset = Menu(self.barra_menu, tearoff=0)
        self.menu_Dataset.add_command(label='Lanzar "LabelImg"',command=self.iniciar_Etiquetador)
        self.menu_Dataset.add_command(label='Seleccionar DataSet',command=self.elegir_Dataset)
        self.menu_Dataset.add_command(label='Seleccionar Lista de clases',command=self.elegir_ListaClases)
        self.barra_menu.add_cascade(label="Dataset", menu=self.menu_Dataset)
        # Menu Modelo
        self.menu_Red = Menu(self.barra_menu, tearoff=0)
        self.menu_Red.add_command(label='Seleccionar Pesos',command=self.elegir_Pesos)
        self.menu_Red.add_separator()
        self.menu_Red.add_command(label='Entrenar Modelo',command=self.entrenar)
        self.barra_menu.add_cascade(label="Red", menu=self.menu_Red)
        # Menu Configuración
        self.menu_Configuracion = Menu(self.barra_menu, tearoff=0)
        self.menu_Configuracion.add_command(label="Seleccionar Configuracion", command=self.elegir_Configuracion)
        self.barra_menu.add_cascade(label="Configuración", menu=self.menu_Configuracion)

        # Panel principal
        self.contenedor = Frame(self.root)
        self.contenedor.pack()

        self.etiqueta_1 = Label(self.contenedor, text="Información:",
                        relief=FLAT, height=1,
                        font=("Helvetica","12","bold"))
        self.etiqueta_1.grid(row=0, column=0, padx=10, sticky=W)

        self.etiqueta_2 = Label (self.contenedor, text="Dataset seleccionado:",
                        relief=FLAT, height=1)
        self.etiqueta_2.grid(row=1, column=0, padx=10, sticky=W)

        self.etiqueta_21 = Label (self.contenedor, text="Ninguno",
                        relief=RIDGE, height=1, width=50)
        self.etiqueta_21.grid(row=1, column=1, padx=10, sticky=W)

        self.etiqueta_3 = Label (self.contenedor, text="Modelo seleccionado:",
                        relief=FLAT, height=1)
        self.etiqueta_3.grid(row=2, column=0, padx=10, sticky=W)

        self.etiqueta_31 = Label (self.contenedor, text="Ninguno",
                        relief=RIDGE, height=1, width=50)
        self.etiqueta_31.grid(row=2, column=1, padx=10, sticky=W)

        self.etiqueta_4 = Label (self.contenedor, text="Configuración seleccionada:",
                        relief=FLAT, height=1)
        self.etiqueta_4.grid(row=3, column=0, padx=10, sticky=W)

        self.etiqueta_41 = Label (self.contenedor, text="Ninguna",
                        relief=RIDGE, height=1, width=50)
        self.etiqueta_41.grid(row=3, column=1, padx=10, sticky=W)

        self.etiqueta_5 = Label (self.contenedor, text="Lista de clases:",
                        relief=FLAT, height=1)
        self.etiqueta_5.grid(row=0, column=2, padx=10, sticky=W)

        self.etiqueta_51 = Label (self.contenedor, text="Ninguna",
                        relief=RIDGE, height=1, width=16)
        self.etiqueta_51.grid(row=0, column=3, padx=10, sticky=W)

        self.lista_1 = Listbox(self.contenedor, height=7, width=30)
        self.lista_1.grid(row=1, column=2, columnspan=2, rowspan=3, pady=5)

        self.contenedor2 = Frame(self.contenedor)
        self.contenedor2.grid(row=4, column=1)

        self.boton_Elegir = Button(self.contenedor2, text="Elegir Imagen", command=self.elegir_Imagen)
        self.boton_Elegir.grid(row=0, column=0)
        
        self.boton_Analizar = Button(self.contenedor2, text="Analizar Imagen", command=self.analizar_Imagen, state=DISABLED)
        self.boton_Analizar.grid(row=0, column=1)

        self.etiqueta_6 = Label (self.contenedor, text="Imagen seleccionada:",
                        relief=FLAT, height=1)
        self.etiqueta_6.grid(row=5, column=0, padx=10, sticky=W)

        self.etiqueta_61 = Label (self.contenedor, text="Ninguna",
                        relief=RIDGE, height=1, width=50)
        self.etiqueta_61.grid(row=5, column=1, padx=10, sticky=W)

        self.etiqueta_7 = Label (self.contenedor, text="Epocas para entrenamiento:",
                        relief=FLAT, height=1)
        self.etiqueta_7.grid(row=4, column=2, padx=10, columnspan=2)
        
        def only_numbers(char):
            return char.isdigit()
        validacion = self.root.register(only_numbers)
        self.entrada_1 = Entry(self.contenedor,  validate="key", validatecommand=(validacion, '%P'))
        self.entrada_1.grid(row=5, column=2, padx=10, columnspan=2)
        
        self.root.config(menu=self.barra_menu)
        self.root.mainloop()
    
    
    # Función para abrir labelImg-master
    def iniciar_Etiquetador(self):
        dirr = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'labelImg-master'))
        os.system('/usr/bin/python3 '+ dirr+'/labelImg.py')

    # Elegir directorio de dataset
    def elegir_Dataset(self):
        self.directorio_d = filedialog.askdirectory(initialdir = "datasets", title="Seleccionar Directorio")
        dirrs = self.directorio_d.split('/')
        if dirrs[-1]=="":
            dirrs[-1]="Ninguno"
        self.etiqueta_21.config(text=dirrs[-1])

    # Elegir lista de clases
    def elegir_ListaClases(self):
        archivo = filedialog.askopenfilename(initialdir = "datasets", title="Seleccionar Lista de clases", filetypes=(("Archivo de texto","*.txt"),))
        dirrs = archivo.split('/')
        if dirrs[-1]=="":
            dirrs[-1]="Ninguna"
            for _i in range(self.lista_1.size()):
                self.lista_1.delete(0)
        else:
            arc = open(archivo,"r")
            lista = arc.read()
            arc.close()
            lista = lista.splitlines()
            for linea in lista:
                self.lista_1.insert(END, linea)
        self.etiqueta_51.config(text=dirrs[-1])

    # Elegir archivo de modelo
    def elegir_Pesos(self):
        self.archivo_P = filedialog.askopenfilename(initialdir = "pesos", title="Seleccionar Pesos de Red", filetypes=(("Modelos y Pesos","*.h5"),))
        dirrs = self.archivo_P.split('/')
        if dirrs[-1]=="":
            dirrs[-1]="Ninguno"
        self.etiqueta_31.config(text=dirrs[-1])

    # Elegir archivo de configuracion
    def elegir_Configuracion(self):
        self.archivo_C = filedialog.askopenfilename(initialdir = "config", title="Seleccionar configuración", filetypes=(("Configuración","*.conf"),))
        dirrs = self.archivo_C.split('/')
        if dirrs[-1]=="":
            dirrs[-1]="Ninguna"
        self.etiqueta_41.config(text=dirrs[-1])

    # Elegir imagen
    def elegir_Imagen(self):
        self.archivo_I = filedialog.askopenfilename(initialdir = "pruebas", title="Seleccionar Imagen", filetypes=(("Imagenes",("*.jpg", "*.jpeg", "*png",
                                                                                                                        "*.bmp", "*.tif", "*.tiff")),))
        dirrs = self.archivo_I.split('/')
        if dirrs[-1]=="":
            dirrs[-1]="Ninguna"
            self.boton_Analizar.config(state=DISABLED)
        else:
            self.boton_Analizar.config(state=NORMAL)
        self.etiqueta_61.config(text=dirrs[-1])

    # Iniciar Analisis
    def analizar_Imagen(self):
        buscador = Analizador()
        buscador.cargar_Clases(self.lista_1.get(first=0, last=self.lista_1.size()))
        buscador.cargar_Configuracion(self.archivo_C)
        buscador.cargar_Imagen(self.archivo_I)
        buscador.cargar_Red(self.archivo_P)
        buscador.analizar_Imagen()
    
    def entrenar(self):
        entrenador = Entrenador()
        entrenador.cargar_Configuracion(self.archivo_C)
        entrenador.cargar_Red(self.archivo_P)
        entrenador.cargar_Dataset(ruta=self.directorio_d, nombre=self.etiqueta_21.cget("text"), clases=self.lista_1.get(first=0, last=self.lista_1.size()))
        entrenador.establecer_Epocas_Entrenamiento(cantidad=self.entrada_1.get())
        entrenador.entrenar_Red()
        
ventana = Interfaz_Grafica()
