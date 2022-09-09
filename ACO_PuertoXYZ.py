# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 11:25:00 2022

@author: UX325
"""

# Algoritmo de Optimización Colonia de Hormigas Artificiales aplicado al problema del Puerto XYZ.
# Cuenta con restricciones de capacidad, de horario de tareas, de turno y de realizar cada tarea una única vez.

# Autora: Marta Bardal Castrillo.


# Librerí­as utilizadas.
import numpy as np
import random
import math


# 1. INICIALIZACIÓN
####################

## Configuración inicial de parámetros:
P = 5 # Número de puertos incluyendo el depósito.
d = 2 # Dimensión del espacio de búsqueda.
lb = 0 # Lí­mite inferior del espacio de búsqueda.
ub = 100 # Lí­mite superior del espacio de búsqueda.
N = 12 # Número de hormigas (camiones).
M = 10 # Número de tareas.
Q = 1 # Cantidad máxima de feromonas que puede desprender una hormiga en su ruta.
alpha = 1 # Peso exponencial asociado a la cantidad de feromonas.
beta = 1 # Peso exponencial asociado a la facilidad para transitar un camino.
rho = 0.05 # Coeficiente de evaporación de feromonas.
it_max = 500 # Número máximo de iteraciones (criterio de paro).
# Falta definir tau_0, que es la cantidad inicial de feromonas y depende de la dist media entre los puertos.


# PROBLEMA PUERTO XYZ
#######################

# POSICIÓN DE LOS PUERTOS
# Coordenadas [x,y] de cada puerto, incluyendo el depósito.
posicion_puertos = np.zeros((P,d))
for p in range(P):
    posicion_puertos[p] = np.array([random.uniform(lb,ub), random.uniform(lb,ub)], dtype=object)


# DISTANCIA ENTRE PUERTOS
def distancia_puertos(posicion_puertos):
    """
    A partir de la posición de los puertos, devuelve la matriz de distancias en km entre cada uno de los puertos.
    Esta matriz es simétrica: la distancia del puerto i al j es la misma que la distancia del puerto j al i.

    Parameters
    ----------
    posicion_puertos : numpy.ndarray.

    Returns
    -------
    distancia_puertos : numpy.ndarray.

    """
    P = posicion_puertos.shape[0]
    distancia_puertos = np.zeros((P,P)) # El 0 es el depósito
    for i in range(P):
        for j in range(i+1):
            distancia_puertos[i,j] = math.sqrt((posicion_puertos[j,0]- posicion_puertos[i,0])**2 + (posicion_puertos[j,1]- posicion_puertos[i,1])**2)
            distancia_puertos[j,i] = distancia_puertos[i,j]
    return distancia_puertos
D = distancia_puertos(posicion_puertos)

tau_0 = (10*Q) / (D.mean()*P) # Cantidad inicial de feromonas.
    

# TIEMPO ENTRE PUERTOS
def tiempo_entre_puertos(posicion_puertos):
    """
    A partir de la posición de los puertos, devuelve la matriz con los tiempos en horas
    que un camión tarda en hacer el trayecto entre cada uno de los puertos.
    Se asume que la velocidad a la que se desplazan los camiones es 100 km/h.

    Parameters
    ----------
    posicion_puertos : numpy.ndarray.

    Returns
    -------
    tiempo_entre_puertos : numpy.ndarray.

    """
    P = posicion_puertos.shape[0]
    tiempo_entre_puertos = np.zeros((P,P)) # El 0 es el depósito
    for i in range(P):
        for j in range(i+1):
            tiempo_entre_puertos[i,j] = distancia_puertos(posicion_puertos)[i,j] / 100 
            tiempo_entre_puertos[j,i] = tiempo_entre_puertos[i,j]
    return tiempo_entre_puertos
T = tiempo_entre_puertos(posicion_puertos)


# FUEL ENTRE PUERTOS
def fuel_entre_puertos(posicion_puertos):
    """
    A partir de la posición de los puertos, devuelve la matriz del fuel en 
    litros consumido por un camión en el trayecto entre cada uno de los puertos.
    Se considera que un camión consume 40 litros de fuel cada 100 km recorridos.

    Parameters
    ----------
    posicion_puertos : numpy.ndarray.

    Returns
    -------
    fuel_entre_puertos : numpy.ndarray.

    """
    P = posicion_puertos.shape[0]
    fuel_entre_puertos = np.zeros((P,P))
    for i in range(P):
        for j in range(i+1):
            fuel_entre_puertos[i,j] = (40 * distancia_puertos(posicion_puertos)[i,j]) / 100 
            fuel_entre_puertos[j,i] = fuel_entre_puertos[i,j]
    return fuel_entre_puertos
F = fuel_entre_puertos(posicion_puertos) 
     
   
# IMPACTO MEDIOAMBIENTAL ENTRE PUERTOS
def impacto_entre_puertos(posicion_puertos):
    """
    A partir de la posición de los puertos, devuelve la matriz del impacto medioambiental 
    causado por el desplazamiento de un camión entre cada uno de los puertos. 

    Parameters
    ----------
    posicion_puertos : numpy.ndarray.

    Returns
    -------
    impacto_entre_puertos : numpy.ndarray.

    """
    P = posicion_puertos.shape[0]
    impacto_entre_puertos = np.zeros((P,P))
    for i in range(P):
        for j in range(i+1):
            impacto_entre_puertos[i,j] = distancia_puertos(posicion_puertos)[i,j]/10
            impacto_entre_puertos[j,i] = impacto_entre_puertos[i,j]
    return impacto_entre_puertos
I = impacto_entre_puertos(posicion_puertos)


# TAREAS
# Array con los orígenes y array con los destinos de las M tareas.
origen = np.repeat(0,M)
destino = np.repeat(0,M)
for i in range(M):
    puertos = np.array(range(1,P))
    origen[i] = random.choice(puertos)
    j = np.delete(puertos, origen[i]-1)
    destino[i] = random.choice(j) # El destino tiene que ser diferente al origen.

# Juntamos en un array el puerto origen con el destino.
puertos_parejas = np.zeros((M,2))
for i in range(M):
    puertos_parejas[i][0] = origen[i]
    puertos_parejas[i][1] = destino[i]
puertos_parejas = puertos_parejas.astype(int)
 
# Peso de cada tarea en kg
peso_tarea = np.random.uniform(100, 1200, size = M)

# Periodo para cada realizar cada tarea:
inicio_tarea = np.random.uniform(0, 10, size = M) # El último inicio posible es a las 10 para que de tiempo a realizar la tarea.
fin_tarea = np.zeros(M) # Se suma la hora de inicio, el tiempo entre puertos y un número random entre 0 y 2h para que sea aleatoria.
for i in range(M):
    fin_tarea[i] = inicio_tarea[i] + tiempo_entre_puertos(posicion_puertos)[origen[i], destino[i]] + random.uniform(0,2)



# FUNCIÓN OBJETIVO.
def funObj(dist_vac,dist_tot,fuel,impacto_med,tiempo_espera, penalizaciones):
    """
    Método de combinación de objetivos: 5 funciones objetivo que se suman ya que todas ellas son de minimización.
    Ponderaciones: se quiere dar más peso a la distancia recorrida tanto en vacío como total y a los tiempos de 
    espera, frente al fuel consumido y al impacto medioambiental provocado.
    Penalizacion: se penaliza las soluciones que no cumplen alguna de las restricciones del problema. Como el problema
                es de minimización, se suma la penalización al valor_fitness final.
    
    funObj1: suma de distancia de cada camión en vacío entre suma de distancia de cada camión total.
    funObj2 = suma de la distancia recorrida por cada camión.
    funObj3 = suma de fuel consumido por cada camión.
    funObj4 = suma del impacto medioambiental provocado por cada camión.
    funObj5 = suma del tiempo de espera de cada camión.
    penalizacion = suma de la penalización de cada camión.
   
    Parameters
    ----------
    dist_vac : numpy.ndarray.
    dist_tot : numpy.ndarray.
    fuel : numpy.ndarray.
    impacto_med : numpy.ndarray.
    tiempo_espera : numpy.ndarray.
    penalizaciones : numpy.ndarray.

    Returns
    -------
    valor_fitness : float.

    """
    funObj1 = np.sum(dist_vac)/np.sum(dist_tot)
    funObj2 = np.sum(dist_tot)
    funObj3 = np.sum(fuel)
    funObj4 = np.sum(impacto_med)
    funObj5 = np.sum(tiempo_espera)
    penalizacion = np.sum(penalizaciones)
    valor_fitness = (1/4)*funObj1 + (1/4)*funObj2 + (1/8)*funObj3 + (1/8)*funObj4 +  (1/4)*funObj5 + penalizacion
    return valor_fitness


def distancia_fuel_impacto_distvacia(R):
    """
    A partir de las rutas generadas por la función decoding, devuelve los vectores de
    distancia total recorrida por cada camión, distancia en vacío recorrida por cada camión,
    fuel total consumido por cada camión e impacto medioambiental provocado por cada camión.
    Estos vectores son de longitud N: 
        dist_tot[i] = distancia total recorrida por el camión i, para todo 0<=i<=N

    Parameters
    ----------
    R : numpy.ndarray.

    Returns
    -------
    dist_vac : numpy.ndarray.
    dist_tot : numpy.ndarray.
    fuel_tot : numpy.ndarray.
    impacto_tot : numpy.ndarray.

    """
    dist_tot = np.repeat(None,N)
    fuel_tot = np.repeat(None,N)
    impacto_tot = np.repeat(None,N)
    dist_vac = np.repeat(None,N)
    for i in range(N): # Para cada vehículo, sumamos las distancias de su ruta
        if (R[i] is None):
            dist_tot[i] = 0
            fuel_tot[i] = 0
            impacto_tot[i] = 0
            dist_vac[i] = 0
        else:
            dist_tot[i] = D[0, R[i][0]] + D[R[i][R[i].size-1], 0] ################################################3 Sumamos las distancias del depósito al puerto inicial y del puerto final al depósito.
            fuel_tot[i] = F[0, R[i][0]] + F[R[i][R[i].size-1], 0]
            impacto_tot[i] = I[0, R[i][0]] + I[R[i][R[i].size-1], 0]
            dist_vac[i] = D[0, R[i][0]] + D[R[i][R[i].size-1], 0]
            for j in range(R[i].size-1):
                dist_tot[i] = dist_tot[i] + D[R[i][j], R[i][j+1]]
                fuel_tot[i] = fuel_tot[i] + F[R[i][j], R[i][j+1]]
                impacto_tot[i] = impacto_tot[i] + I[R[i][j], R[i][j+1]]
            for j in range(1,R[i].size-1,2):
                dist_vac[i] = dist_vac[i] + D[R[i][j], R[i][j+1]]
    return dist_vac,dist_tot,fuel_tot,impacto_tot


def decoding2(ruta_hormiga,tareas_hormiga):
    """
    A partir de la ruta por los nodos de una hormiga, y los í­ndices indicando
    el orden en que realiza las M tareas, devuelve un array que contiene las
    rutas de cada uno de los camiones para completar las M tareas sin sobrepasar 
    las capacidades de carga de cada camión, un array indicando cuántas veces se ha
    realizado cada tarea, y un array con el orden de tareas realizadas por cada camión.

    Parameters
    ----------
    ruta_hormiga : numpy.ndarray.
    
    tareas_hormiga : numpy.ndarray.

    Returns
    -------
    rutas_totales : numpy.ndarray.
    
    tareas : numpy.ndarray.
    
    orden_tareas : numpy.ndarray.

    """
    rutas_totales = np.repeat(None,N)
    camion=0
    tareas = np.zeros(M)
    orden_tareas = np.repeat(None, N) # Tareas realizadas por cada vehículo, por orden de realización.
    
    while camion in range(N):
        capacidad = capacidades[camion]
        i=0
        
        while (len(tareas_hormiga)>0) and (i<len(tareas_hormiga)):
            
            tarea_a_realizar = tareas_hormiga[i]
            pareja = puertos_parejas[tarea_a_realizar]
            
            if (capacidad - peso_tarea[tarea_a_realizar]) >= 0: # El camión puede realizar la tarea
                if (rutas_totales[camion] is None):
                    rutas_totales[camion] = np.array(ruta_hormiga[i])
                    orden_tareas[camion] = np.array([tarea_a_realizar])
                else:
                    rutas_totales[camion], orden_tareas[camion] = mejor_secuencia(rutas_totales[camion], orden_tareas[camion],tarea_a_realizar,pareja)
                  
                tareas_hormiga = np.delete(tareas_hormiga,i)
                tareas[tarea_a_realizar] += 1
                eliminado = ruta_hormiga.pop(i) # Se elimina la pareja ya adjudicada.
                i=M
            else:
                i=i+1
        camion = camion +1
        if (camion == N) and (len(tareas_hormiga)>0):
            camion = 0
    tareas = tareas.astype(int)
    return rutas_totales, tareas, orden_tareas


def mejor_secuencia(ruta_v, orden_tareas_v, c, pareja):
    """
    A partir de la ruta de un vehículo, las tareas asociadas a esa ruta, la nueva tarea 
    a realizar y el [origen, destino] de dicha tarea, devuelve la ruta que recorre menor distancia
    entre todas las posibles inserciones del [origen, destino] en la ruta dada del vehículo, y el 
    orden en que realiza las tareas.

    Parameters
    ----------
    ruta_v : numpy.ndarray.
    orden_tareas_v : numpy.ndarray.
    c : int.
    pareja : numpy.ndarray.

    Returns
    -------
    ruta_v : numpy.ndarray.
    orden_tareas_v : numpy.ndarray.

    """
    l = ruta_v.size
    ruta_pruebas = np.repeat(None, l/2 + 1) # l intentos de insertar c en los diferentes sitios posibles de ruta_v. 
    orden_tareas_pruebas = np.repeat(None,l/2 + 1)
    dist_tot_pruebas = np.repeat(None, l/2 + 1) # Distancia total que se recorre en cada una de las inserciones de c.
    
    for s in range(0,l+1,2): # No se pueden romper las parejas, solo se puede insertar en las posiciones pares.
        ruta_pruebas[int(s/2)] = ruta_v
        orden_tareas_pruebas[int(s/2)] = orden_tareas_v
        ruta_pruebas[int(s/2)] = np.insert(ruta_pruebas[int(s/2)], s, pareja)
        orden_tareas_pruebas[int(s/2)] = np.insert(orden_tareas_pruebas[int(s/2)], int(s/2), c)
        dist_tot_pruebas[int(s/2)] = D[0, ruta_pruebas[0][0]] + D[ruta_pruebas[int(s/2)][ruta_pruebas[int(s/2)].size-1], 0]
        for i in range(ruta_pruebas[int(s/2)].size-1):
            dist_tot_pruebas[int(s/2)] = dist_tot_pruebas[int(s/2)] + D[ruta_pruebas[int(s/2)][i], ruta_pruebas[int(s/2)][i+1]]
    # Ordenamos las diferentes inserciones según la distancia recorrida de menor a mayor.
    pruebas_ordenadas = np.argsort(dist_tot_pruebas)
    ruta_v = ruta_pruebas[pruebas_ordenadas[0]]
    orden_tareas_v =  orden_tareas_pruebas[pruebas_ordenadas[0]] 
    return ruta_v, orden_tareas_v


def horas(orden_tareas):
    """
    A partir de la secuencia de tareas que realiza cada camión, ordenadas por orden de
    realización, calcula las horas de llegada a los puertos origen de cada camión para
    cada tarea que realiza, las horas de llegada a los puertos destino de cada camión para
    cada tarea que realiza, la hora de vuelta al depósito de cada camión y  devuelve el tiempo
    total de espera de cada camión y las penalizaciones de cada uno por no cumplir las restricciones.

    Parameters
    ----------
    orden_tareas : numpy.ndarray.

    Returns
    -------
    tiempo_espera : numpy.ndarray.

    penalizaciones : numpy.ndarray.
    
    """
    hora_llegada_origen = np.repeat(None,N)
    hora_llegada_destino = np.repeat(None,N)
    tiempo_espera = np.repeat(0., N)
    penalizaciones = np.zeros(N)
        
    vehiculo = 0
    while vehiculo in range(N):
        
        if orden_tareas[vehiculo] is None:
            vehiculo += 1
        else:
            hora_llegada_origen[vehiculo] = np.repeat(0.,orden_tareas[vehiculo].size)
            hora_llegada_destino[vehiculo] = np.repeat(0.,orden_tareas[vehiculo].size)
            hora_vuelta_deposito = np.repeat(None,N)
    
            for i in range(orden_tareas[vehiculo].size):
                diferencia = 0
                if i == 0:
                    hora_llegada_origen[vehiculo][i] = T[0, origen[orden_tareas[vehiculo][i]]] # Trayecto del depósito al primer origen.
                    if hora_llegada_origen[vehiculo][i] < inicio_tarea[orden_tareas[vehiculo][i]]:
                        diferencia = inicio_tarea[orden_tareas[vehiculo][i]] - hora_llegada_origen[vehiculo][i]
                        tiempo_espera[vehiculo] = tiempo_espera[vehiculo] + diferencia
                        hora_llegada_origen[vehiculo][i] = inicio_tarea[orden_tareas[vehiculo][i]] # Si el camión llega antes, espera.
                    hora_llegada_destino[vehiculo][i] = hora_llegada_origen[vehiculo][i] + T[origen[orden_tareas[vehiculo][i]],destino[orden_tareas[vehiculo][i]]]
                    
                else:
                    hora_llegada_origen[vehiculo][i] = hora_llegada_destino[vehiculo][i-1] + T[destino[orden_tareas[vehiculo][i-1]],origen[orden_tareas[vehiculo][i]]]
                    if hora_llegada_origen[vehiculo][i] < inicio_tarea[orden_tareas[vehiculo][i]]:
                        diferencia = inicio_tarea[orden_tareas[vehiculo][i]] - hora_llegada_origen[vehiculo][i]
                        hora_llegada_origen[vehiculo][i] = inicio_tarea[orden_tareas[vehiculo][i]] 
                    hora_llegada_destino[vehiculo][i] = hora_llegada_origen[vehiculo][i] + tiempo_entre_puertos(posicion_puertos)[origen[orden_tareas[vehiculo][i]],destino[orden_tareas[vehiculo][i]]]
                
                contador = restriccion_horario_tareas(orden_tareas[vehiculo][i], hora_llegada_origen[vehiculo][i], hora_llegada_destino[vehiculo][i], diferencia)
                penalizaciones[vehiculo] = penalizaciones[vehiculo] + contador
            
            hora_vuelta_deposito[vehiculo] = hora_llegada_destino[vehiculo][-1] + T[destino[orden_tareas[vehiculo][-1]],0]
            penalizaciones[vehiculo] = penalizaciones[vehiculo] + restriccion_turnos(hora_vuelta_deposito[vehiculo])
            
            vehiculo += 1
        
    return tiempo_espera, penalizaciones
   

def restriccion_horario_tareas(orden_tareas_v_i, hora_llegada_origen_v_i, hora_llegada_destino_v_i, diferencia):
    """
    A partir de una tarea dada, la hora a la que el camión que la realiza llega al nodo origen, la hora a la que 
    llega al nodo destino, y el tiempo de espera del camión a realizar la tarea en el puerto origen, si es que espera,
    devuelve un contador que suma la penalización correspondiente.
    Se considera un error leve que el camión no llegue al puerto origen antes de la hora de inicio de la tarea,
    por eso se penaliza con 0.5 unidades.
    Se considera un error leve que el tiempo que espera un camión en el origen hasta el inicio de la tarea sea
    superior a 2 horas, por lo que se penaliza con 0.5 unidades.
    Se considera un error grave que el camión llegue al puerto destino más tarde de la hora de finalización de la tarea, por eso
    se penaliza con 1 unidad.

    Parameters
    ----------
    orden_tareas_v_i : int.
    hora_llegada_origen_v_i : float.
    hora_llegada_destino_v_i : float.
    diferencia : float.

    Returns
    -------
    contador : float.

    """
    contador = 0
    penalizacion_1 = hora_llegada_origen_v_i >= inicio_tarea[orden_tareas_v_i]
    penalizacion_2 = diferencia > 2 # Si llega más de 2 h antes al origen.
    penalizacion_3 = hora_llegada_destino_v_i >= fin_tarea[orden_tareas_v_i] # Si llega tarde al destino.
    if penalizacion_1:
        contador += 0.5
    if penalizacion_2:
        contador += 0.5  
    if penalizacion_3:
        contador += 1
    return contador
           

def restriccion_turnos(hora_vuelta_deposito_v):
    """
    A partir de la hora de regreso al depósito de un vehículo, se devuelve un contador que vale 0 si ésta es menor
    o igual que 12, o 1 si es mayor. Por tanto, se considera un error grave llegar tarde al depósito.
    Esta función tiene como objetivo el cumplimiento de que los turnos de los camiones duran 12 horas.

    Parameters
    ----------
    hora_vuelta_deposito_v : float.

    Returns
    -------
    contador : int.

    """
    contador = 0
    penalizacion_4 = hora_vuelta_deposito_v > 12
    if penalizacion_4:
            contador = 1
    return contador
        

def restriccion_tareas_unica_vez(tareas):
    """
    Las tareas deben realizarse una única vez. A partir del array de tareas, si alguna
    tarea se ha realizado más de una vez, se impone una penalización en la variable
    contador, que se sumará al valor en la función objetivo.

    Parameters
    ----------
    tareas : numpy.ndarray.

    Returns
    -------
    contador : int.

    """
    contador = 0
    for i in range(M):
        if tareas[i]!=0 and tareas[i]!=1:
            contador += 1
    return contador


## Inicializar matrices de información:
eta =  np.zeros((P,P)) # Matriz de facilidad para el tránsito
for i in range(P):
    for j in range(P):
        if D[i,j] == 0:
            eta[i,j] = 0
        else:
            eta[i,j] = 1/D[i,j]
tau = tau_0*np.ones((P,P)) # Matriz de concentración de feromonas


## Definición de la clase Hormiga:
class Hormiga():
    """
    Representa a una hormiga, un agente de búsqueda del algoritmo ACO.
    """
    def __init__(self):
        """
        Atributos de instancia:
            tour (list) : recorrido de la hormiga (camión) por los nodos.
            
            fit (float) : valor de fitness del tour de la hormiga (camión).
            
            cap (float) : valor de la capacidad en kg de la hormiga (camión).

        """
        self.tour = []
        self.fit = None
        self.cap = np.random.uniform(700, 1300)
    def añadir_nodo(self, nodo):
        """
        # Añade el nodo dado al tour de la hormiga.

        Parameters
        ----------
        nodo : int.

        """
        self.tour.append(nodo)


## Definición del enjambre, conjunto de hormigas:
capacidades = np.repeat(None,N) # Capacidad de cada camión.
Hormigas = np.array([])
for i in range(N):
    hormiga = Hormiga()
    Hormigas = np.append(Hormigas,hormiga)
    capacidades[i] = Hormigas[-1].cap

Evolucion = np.zeros(it_max) # Registro de los mejores fitness globales.


# 2. PROCESO ITERATIVO
#######################

for it in range(it_max):
    
    ## 1. Transición entre estados.
    for k in range(N):
        tareas_hormiga = np.array([])
        Hormigas[k].tour = []
        r = np.random.choice(origen) # Cada hormiga comienza en un nodo aleatorio.
        ind = np.where(origen == r)[0]
        indice = np.random.choice(ind) # indice es el número de la tarea que se realiza.
        pareja = puertos_parejas[indice]
        Hormigas[k].añadir_nodo(pareja) # Se añade la pareja a la ruta de la hormiga.
        tareas_hormiga = np.append(tareas_hormiga,indice) # Registro de las tareas ya realizadas.
        
        while len(tareas_hormiga) < M: # Mientras queden tareas por hacer:
            i = Hormigas[k].tour[-1][1] # Último nodo visitado por la hormiga k (nodo destino).
            Prob = np.zeros(M) # Vector de probabilidades de ir desde el nodo i al resto de los nodos.
            for j in range(M):
                if j not in tareas_hormiga: # Las tareas se realizan una única vez.
                    Prob[j] = tau[i,origen[j]]**alpha * eta[i,origen[j]]**beta # Probabilidad de realizar la tarea número j estando en el nodo i.
            if sum(Prob)!=0:
                Prob = Prob/sum(Prob)
            
            # Método de selección por ruleta:
            C = np.cumsum(Prob)
            if all(C)==0: # Entonces no se puede aplicar la selección por ruleta.
                for l in range(M):
                    if l not in tareas_hormiga:
                        siguiente = l
            else:
                siguiente = np.where(np.less_equal(random.random(), C))[0][0] # Índice del primer elemento que cumple que es mayor que el número random.
            # siguiente es el indice de la tarea elegida para ser realizada por la hormiga k.
            pareja = puertos_parejas[siguiente]
            Hormigas[k].añadir_nodo(pareja)
            tareas_hormiga = np.append(tareas_hormiga,siguiente) # Registro de las tareas realizadas.
        tareas_hormiga = tareas_hormiga.astype(int)
        
        ruta = Hormigas[k].tour.copy() # Hacemos una copia porque en la actualización de feromonas necesitaremos Hormigas[k].tour.
        rutas_totales, tareas, orden_tareas = decoding2(ruta,tareas_hormiga) # Construimos la ruta de cada camión.
        dist_vac,dist_tot,fuel,impacto_med = distancia_fuel_impacto_distvacia(rutas_totales)
        tiempo_espera, penalizaciones = horas(orden_tareas)
        penalizacion_tarea_unica = restriccion_tareas_unica_vez(tareas)
        penalizaciones += penalizacion_tarea_unica # Se sumaría penalizacion_tarea_unica a cada valor del array penalizaciones, asegurando así que se cumple esa restricción.
        Hormigas[k].fit = funObj(dist_vac,dist_tot,fuel,impacto_med,tiempo_espera, penalizaciones) # Valor de fitness.
     
        if it == 0 and k == 0:
            mejor_fitness_global = Hormigas[k].fit
            mejor_tour_global = Hormigas[k].tour
        else:
            if Hormigas[k].fit < mejor_fitness_global:
                mejor_fitness_global = Hormigas[k].fit
                mejor_tour_global = Hormigas[k].tour
        
    ## 2. Actualización de feromonas.
    for k in range(N):
        tour = Hormigas[k].tour
        for l in range(P):
            i = tour[l]
            j = tour[l+1]
            tau[i,j] = tau[i,j] + Q / Hormigas[k].fit 
    
    ## 3. Evaporación de feromonas.
    tau = (1-rho) * tau

    # Registrar la mejor solución para la iteración actual.
    Evolucion[it] = mejor_fitness_global
    # Mostrar iteración actual y mejor fitness hasta el momento
    print("Iteracion ", str(it), ": Costo(Fitness) = ",str(Evolucion[it]))
