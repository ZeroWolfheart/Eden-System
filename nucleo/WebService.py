from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
import base64
import socket
import netifaces as ni

from Interface import Analizador

# Clase que contiene el manejador de solicitudes
class __ManejadorRespuestas__(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/nucleo',)
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    # Añadir cabeceras a todas las respuestas
    def end_headers(self):
        self.send_header("Access-Control-Allow-Headers", 
                         "Origin, X-Requested-With, Content-Type, Accept")
        self.send_header("Access-Control-Allow-Origin", "*")
        SimpleXMLRPCRequestHandler.end_headers(self)

# Clase que contiene el servicio web
class servidorXMLRCP:
    
    def __init__(self):
        #Atributos de  la clase:
        self.__buscador__ = Analizador()
        self.__clases__ = []
        self.__configuracion__ = None
        self.__imagen__ = None
        self.__red__ = None
    
    def inicializarBuscador(self, listaClases, archivoConfig, archivoRed):
        self.__buscador__.cargar_Clases(listaClases)
        self.__buscador__.cargar_Configuracion(archivoConfig)
        self.__buscador__.cargar_Red(archivoRed)
    
    def publicarIp(self):
        # Conocer interfaz conectada a la red y utilizar dicha ip
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    
    def iniciarServidor(self, socketS=8001):
        ip = self.publicarIp()
            
        # Crear Servidor
        with SimpleXMLRPCServer((ip,socketS),
                                requestHandler = __ManejadorRespuestas__) as servidor:
            
            servidor.register_introspection_functions()

        #Registrar una funcion bajo un nombre diferente    
            def funcion_ejemplo(text):
                # Guardar cadena base64 de respaldo
                gg = open('lacad.txt', 'wt')
                gg.write(text)
                gg.close()
                # Guardar imagen recibida
                imagen_bin = base64.urlsafe_b64decode(text)
                imagen_ent = open('recibido.jpeg', 'wb')
                imagen_ent.write(imagen_bin)
                imagen_ent.close()
                # Analizar imagen
                self.__buscador__.cargar_Imagen('recibido.jpeg')
                self.__buscador__.analizar_Imagen(imprimir=True)
                # Leer y enviar una imagen
                imagen_salr = open('analizado.png','rb')
                imagen_sal = imagen_salr.read()
                imagen_pro = base64.encodebytes(imagen_sal)
                imagen_pro = imagen_pro.decode()
                return imagen_pro
            servidor.register_function(funcion_ejemplo,'ejemplo')

        # Ejecutar el servidor en el main loop
            print("Dirección del servidor: "+servidor.server_address[0]+":"+str(servidor.server_address[1]))
            servidor.serve_forever()