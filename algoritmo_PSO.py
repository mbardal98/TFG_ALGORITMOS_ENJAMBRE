

# Algoritmo de Optimización por Enjambre de Partículas (PSO).
# Autora: Marta Bardal Castrillo.

# Librerías utilizadas.
import numpy as np
import random
import numpy.matlib


# FUNCIÓN OBJETIVO 
"""
Función sencilla que define el problema de minimización a resolver.
"""
funObj = lambda x: (x[0]+6) * (x[1]+20)


# 1. INICIALIZACIÓN
####################

## Configuración de parámetros.
d = 2 # Dimensión del espacio de búsqueda 
lb = np.repeat(0,d) # Límite inferior del espacio de búsqueda
ub = np.repeat(9,d) # Límite superior del espacio de búsqueda

N = 300 # Número de partículas
it = 0 # Iteración inicial
it_max = 30 # Número máximo de iteraciones (criterio de paro: k = it_max)
c1 = 2 # Constante cognitiva
c2 = 5 # Constante social

Evolucion = np.zeros(it_max + 1) # Vector que recoge el mejor fitness global de cada iteración.


## Definición de la clase Particula.
class Particula():
    """
    Representa a una partícula del algoritmo PSO.
    """
    def __init__(self):
        """
        Atributos de instancia:
            posicion (numpy.ndarray) : posición de la partícula de dimensión d.
            
            velocidad (numpy.ndarray) : velocidad de la partícula de dimensión d.
            
            fit (numpy.ndarray) : valor de fitness de la partícula.
            
            mejor_fitness_local (float) : valor del mejor fitness de la partícula.
            
            mejor_posicion_local (numpy.ndarray) : posición de la partícula cuando alcanza el mejor_fitness_local.
        """
        self.posicion = np.repeat(None, d)
        self.velocidad = np.zeros(d) # Velocidad inicial = 0
        self.fit = np.array(None)
        self.mejor_fitness_local = None
        self.mejor_posicion_local = None
   
## Definición del enjambre, conjunto de partículas.
Particulas = np.array([])
for i in range(N):
    particula = Particula()
    Particulas = np.append(Particulas,particula)


## Inicialización del enjambre.
for k in range(N):
    Particulas[k].posicion = np.array([random.uniform(lb[0],ub[0]), random.uniform(lb[1],ub[1])], dtype=object) 
    Particulas[k].fit = funObj(Particulas[k].posicion)
    # Mejor fitness local de cada partícula
    Particulas[k].mejor_fitness_local = Particulas[k].fit
    # Mejor posición local de cada partícula
    Particulas[k].mejor_posicion_local = Particulas[k].posicion
    if k == 0:
        mejor_fitness_global = Particulas[k].fit
    else:
        if Particulas[k].mejor_fitness_local < mejor_fitness_global:
            mejor_fitness_global = Particulas[k].mejor_fitness_local
            mejor_posicion_global = Particulas[k].mejor_posicion_local

Evolucion[0] = mejor_fitness_global
print("Iteracion ", str(0), ": Costo(Fitness) = ",str(Evolucion[0]))


# 2. PROCESO ITERATIVO
#######################

while it < it_max: # Criterio de paro
    it += 1
    # Cálculo de la nueva velocidad y posición de cada partícula:
    for i in range(N):
        parte1 = c1*random.uniform(0,1)*(Particulas[i].mejor_posicion_local - Particulas[i].posicion)
        parte2 = c2*random.uniform(0,1)*(mejor_posicion_global - Particulas[i].posicion)
        nueva_velocidad = Particulas[i].velocidad + parte1 + parte2
        Particulas[i].velocidad = nueva_velocidad
        Particulas[i].posicion = Particulas[i].posicion + Particulas[i].velocidad
        # Verificar que las partículas no se salgan de los límites lb y ub:
        for i in range(N):
            for j in range(d):
                if  Particulas[i].posicion[j] < lb[j]:
                    Particulas[i].posicion[j] = lb[j];
                elif Particulas[i].posicion[j] > ub[j]:
                    Particulas[i].posicion[j] = ub[j];
        Particulas[i].fit = funObj(Particulas[i].posicion)
        # Mejor fitness local de cada partícula
        if Particulas[i].fit < Particulas[i].mejor_fitness_local:
            Particulas[i].mejor_fitness_local = Particulas[i].fit
            Particulas[i].mejor_posicion_local = Particulas[i].posicion
        # Fitness global  
        if  Particulas[i].mejor_fitness_local < mejor_fitness_global:
            mejor_fitness_global = Particulas[i].mejor_fitness_local
            mejor_posicion_global = Particulas[i].mejor_posicion_local
    
    Evolucion[it] = mejor_fitness_global
    # Mostrar iteración actual y mejor fitness actual
    print("Iteracion ", str(it), ": Costo(Fitness) = ",str(Evolucion[it]))
 