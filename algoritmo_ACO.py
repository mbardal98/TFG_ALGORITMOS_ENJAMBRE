# -*- coding: utf-8 -*-

# Algoritmo de Optimización por Colonia de Hormigas (ACO).
# Autora: Marta Bardal Castrillo.

# Librerías utilizadas.
import numpy as np
import random
import math


# 1. INICIALIZACIÓN
####################

## Configuración de parámetros:
    
nodos = 20 # Número de nodos del grafo.
d = 2 # Dimensión del espacio de búsqueda.
lb = 0 # Límite inferior del espacio de búsqueda para cada coordenada.
ub = 100 # Límite superior del espacio de búsqueda para cada coordenada.

## Creación del grafo: se genera la posición de los nodos de forma random.
posicion_nodos = np.zeros((nodos,d))
for nodo in range(nodos):
    posicion_nodos[nodo] = np.array([random.uniform(lb,ub), random.uniform(lb,ub)], dtype=object)
"""
posicion_nodos = np.array([[27.80217941, 10.60330955],
       [91.55088034,  7.18023882],
       [37.4142924 , 56.06764855],
       [85.65670579,  5.07311231],
       [10.78474335, 32.44747315],
       [29.01085912, 93.25796105],
       [77.12459501,  7.36786913],
       [26.0863445 , 31.07970239],
       [23.94814938, 43.98853112],
       [72.41439776, 73.6736767 ],
       [69.25793365, 36.65853016],
       [71.94928243, 49.45611459],
       [54.27366425, 46.26659801],
       [37.50265528, 88.34395811],
       [73.92254568, 95.39165434],
       [62.19846885, 22.15041868],
       [34.14425706,  8.79532419],
       [ 3.52308977,  3.68666872],
       [63.83667221,  5.97010104],
       [59.91507208, 86.34757655]])
"""

N = 200 # Número de agentes de búsqueda (hormigas).
Q = 1 # Cantidad máxima de feromonas que puede desprender una hormiga en su ruta.
alpha = 1 # Peso exponencial asociado a la cantidad de feromonas.
beta = 2 # Peso exponencial asociado a la facilidad para transitar un camino.
rho = 0.05 # Coeficiente de evaporación de feromonas.
it_max = 100 # Número máximo de iteraciones (criterio de paro).
# Falta definir tau_0, que es la cantidad inicial de feromonas y depende de la dist media entre los puertos.


## Distancia entre nodos:
# Matriz simétrica de distancias aleatorias.
def distancia_nodos(posicion_nodos):
    """
    A partir de la posición de los nodos, devuelve la matriz de distancias entre cada uno de los nodos.

    Parameters
    ----------
    posicion_nodos : numpy.ndarray.

    Returns
    -------
    distancia_nodos : numpy.ndarray.

    """
    P = posicion_nodos.shape[0]
    distancia_nodos = np.zeros((P,P)) 
    for i in range(P):
        for j in range(i+1):
            distancia_nodos[i,j] = math.sqrt((posicion_nodos[j,0]- posicion_nodos[i,0])**2 + (posicion_nodos[j,1]- posicion_nodos[i,1])**2)
            distancia_nodos[j,i] = distancia_nodos[i,j]
    return distancia_nodos

distancia_nodos = distancia_nodos(posicion_nodos) # Matriz con las distancias entre los nodos.

# Parámetro tau_0 (en función de la media de las distancias entre nodos):
tau_0= (10*Q) / (distancia_nodos.mean()*nodos) # Cantidad inicial de feromonas.


## Definición de la función objetivo:
def funObj(tour, distancia_nodos):
    """
    A partir de la ruta por los nodos seguida por una hormiga, y de la matriz de distancias entr nodos,
    devuelve el valor de esta ruta en la función objetivo a minimizar, es decir, la suma de distancias 
    recorridas por la hormiga.

    Parameters
    ----------
    tour : list.
    distancia_nodos : numpy.ndarray.

    Returns
    -------
    L : float.

    """
    n = len(tour)
    tour.append(tour[0]) # Las hormigas regresan a su punto de partida. 
    L = 0
    for i in range(n):
        L = L + distancia_nodos[int(tour[i]),int(tour[i+1])] # Suma de las distancias recorridas en el tour
    return L


## Inicialización de las matrices de información:
facilidad_transito =  np.zeros((nodos,nodos)) # Matriz de facilidad para el tránsito.
for i in range(nodos):
    for j in range(nodos):
        if distancia_nodos[i,j] == 0:
            facilidad_transito[i,j] = 0
        else:
            facilidad_transito[i,j] = 1/distancia_nodos[i,j]
tau = tau_0*np.ones((nodos,nodos)) # Matriz de concentración de feromonas.


## Definición de la clase Hormiga.
class Hormiga():
    """
    Representa a una hormiga, un agente de búsqueda del algoritmo ACO.
    """
    def __init__(self):
        """
        Atributos de instancia:
            tour (list) : recorrido de la instancia hormiga por los nodos.
            
            fit (float) : valor de fitness del tour de la instancia hormiga.

        """
        self.tour = []
        self.fit = None
        
    def añadir_nodo(self, nodo):
        """
        # Añade el nodo dado al tour de la hormiga.

        Parameters
        ----------
        nodo : int.

        """
        self.tour.append(nodo)

## Definición del enjambre, conjunto de hormigas.
Hormigas = np.array([])
for i in range(N):
    hormiga = Hormiga()
    Hormigas = np.append(Hormigas,hormiga)

Evolucion = np.zeros(it_max) # Registro de los mejores fitness.


# 2. PROCESO ITERATIVO
#######################

for it in range(it_max): # Criterio de paro.
    
    ## 1. Transición de las hormigas entre estados.
    for k in range(N):
        Hormigas[k].tour = []
        r = random.randrange(0, nodos) # Cada hormiga comienza en un nodo aleatorio.
        Hormigas[k].añadir_nodo(r)  
        rango = np.array(range(nodos))
        nodos_restantes = np.delete(rango,r)
        
        while len(nodos_restantes) > 0:
            i = Hormigas[k].tour[-1] # Último nodo visitado por la hormiga k
            P = np.zeros(nodos) # Vector de probabilidades de ir desde el nodo i al resto de los nodos.
            for j in range(nodos):
                if j in nodos_restantes:
                    P[j] = tau[i,j]**alpha * facilidad_transito[i,j]**beta # Probabilidad de ir del nodo i al j
            P = P/sum(P)
            
            # Método de selección por ruleta:
            C = np.cumsum(P)
            siguiente = np.where(np.less_equal(random.random(), C))[0][0] #Índice del primer elemento que cumple que es mayor que el número random.
            
            Hormigas[k].añadir_nodo(siguiente)
            indice = np.where(nodos_restantes == siguiente)
            nodos_restantes = np.delete(nodos_restantes,indice)
        
        Hormigas[k].fit = funObj(Hormigas[k].tour,distancia_nodos) # Valor de fitness de la ruta de la hormiga k.
        if it == 0 and k == 0:
            mejor_fitness_global = Hormigas[k].fit
            mejor_tour_global = Hormigas[k].tour
        else:
            if Hormigas[k].fit < mejor_fitness_global:
                mejor_fitness_global = Hormigas[k].fit
                mejor_tour_global = Hormigas[k].tour
        
    ## 2. Actualización (incremento) de feromonas.
    for k in range(N):
        tour = Hormigas[k].tour
        for l in range(nodos):
            i = tour[l]
            j = tour[l+1]
            tau[i,j] = tau[i,j] + Q / Hormigas[k].fit 
    
    ## 3. Evaporación de feromonas.
    tau = (1-rho) * tau

    # Registrar la mejor solución para la iteración actual
    Evolucion[it] = mejor_fitness_global
    # Mostrar iteración actual y mejor fitness hasta el momento
    print("Iteracion ", str(it), ": Costo(Fitness) = ",str(Evolucion[it]))
print(mejor_tour_global)
