# main.py
# Función para obtener posibles candidatos a morosidad
# ARCHIVO PRINCIPAL
# main.py
# Función para obtener posibles candidatos a morosidad
# ARCHIVO PRINCIPAL

import pandas as pd 
import numpy as np
import math
import os

from funciones_apoyo import FuncionesParaPrediccion  


path = r'./datasets'
test = FuncionesParaPrediccion(path)  

def predicciones():
    test.BitacoraDelTodo() # Creación de archivo de apoyo para entrenamiento y predicciones
    test.hacer_prediccion(0.5) # Definir el umbral para clasificación de predicciones según su probabilidad [0,1]
    return "Predicciones realizadas con éxito"

predicciones() 