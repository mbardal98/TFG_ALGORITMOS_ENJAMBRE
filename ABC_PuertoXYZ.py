# -*- coding: utf-8 -*-


# Algoritmo de Optimización por Colonia Artificial de Abejas aplicado al problema del puerto XYZ.
# Cuenta con restricciones de capacidad, horario de tareas, turno y de realizar cada tarea una única vez.

# Autora: Marta Bardal Castrillo.


# Librerías utilizadas.
import numpy as np
import random
import math
import time


for ejecucion in range(20): # Bucle para realizar las 20 ejecuciones para el estudio de parámetros.
    
    # 1. INICIALIZACIÓN
    ####################
    
    ## Configuración de parámetros
    
    N = 5 # Número de fuentes de alimento (camiones).
    M = 15 # Número de tareas.
    P = 7 # Número de puertos incluyendo el depósito.
    d = 2 # Dimensión del espacio de soluciones
    lb = 400 # Límite inferior del espacio de búsqueda para las dos dimensiones.
    ub = 1200 # Límite superior del espacio de búsqueda para las dos dimensiones.
    it_max = 70 # Número máximo de iteraciones (criterio de paro).
    N_Obs = N # Número de abejas observadoras
    a = 1.0 # Coeficiente de aceleración (tamaño de paso)
    limit = N*2 # Número máximo de intentos (criterio de abandono de fuente de alimento)
    
    trial = np.zeros(N) # Contador de intentos
    Evolucion = np.zeros(it_max + 1)
    # Registro del fitness en cada iteración de cada fuente de alimento para poder tomar el mejor
    # y el peor fitness con el objetivo de calcular la calidad de cada fuente en cada iteración:
    registro_fitness = np.repeat(None, N) 
    
    
    # PROBLEMA PUERTO XYZ
    #######################
    
    # POSICIÓN DE LOS PUERTOS
    # Coordenadas [x,y] de cada puerto, incluyendo el depósito.
    posicion_puertos = np.zeros((P,d))
    for p in range(P):
        posicion_puertos[p] = np.array([random.uniform(lb,ub), random.uniform(lb,ub)], dtype=object)
    
    """
    # Posición de los puertos sobre la que se ha realizado el estudio de parámetros.
    
    posicion_puertos = np.array([[19.29761503, 21.02175458],
                                 [ 2.31502326, 49.66344282],
                                 [ 9.39603107, 89.69459349],
                                 [34.91953419, 59.81239703],
                                 [54.11077709, 11.3642149 ]])
    """
    
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
    
    
    # FUEL
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
    
    
    # IMPACTO MEDIOAMBIENTAL
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
    
    
    """
    # Datos acerca de las tareas con los que se han realizado el estudio de parámetros.
    
    origen = np.array([4, 1, 2, 2, 1, 1, 3, 2, 4, 1, 2, 3, 4, 1, 3])
    
    destino = np.array([2, 3, 3, 3, 2, 2, 1, 1, 3, 1, 3, 1, 2, 3, 2])
    
    puertos_parejas = np.array([[4, 2],
           [1, 3],
           [2, 3],
           [2, 3],
           [1, 2],
           [1, 2],
           [3, 1],
           [2, 1],
           [4, 3],
           [1, 1],
           [2, 3],
           [3, 1],
           [4, 2],
           [1, 3],
           [3, 2]])
    
    peso_tarea = np.array([ 941.91001973,  749.50018493,  839.50612148,  568.56028023,
            707.16715006,  961.19766525,  590.49882243,  705.49907541,
            761.25136794,  620.4785829 ,  594.19284897,  980.38689866,
            414.81202143,  272.76752845,  656.53774574])
    
    inicio_tarea = np.array([9.70382134, 5.71913036, 7.22875723, 7.92266632, 7.87832384,
           6.66814187, 1.20222571, 7.56193899, 8.45804572, 8.00912841,
           2.84067953, 1.8128774 , 8.13589976, 8.05878682, 5.71024439])
    
    fin_tarea = np.array([11.13981036,  7.30185358,  7.81895575, 10.52491863,  9.20375729,
            7.70194501,  2.06874698,  8.7938783 ,  9.69874703,  9.17753254,
            3.1852078 ,  4.22194045,  9.95183546, 10.85007814,  6.75627841])
    """
    
    
    # FUNCIÓN OBJETIVO.
    def funObj(dist_vac,dist_tot,fuel,impacto_med,tiempo_espera, penalizaciones):
        """
        Método de combinación de objetivos: 5 funciones objetivo que se suman ya que todas ellas son de minimización.
        Ponderaciones: se quiere dar más peso a la distancia recorrida tanto en vacío como total y a los tiempos de 
        espera, frente al fuel consumido y al impacto medioambiental provocado.
        Penalizacion: se penaliza las soluciones que no cumplen alguna de las restricciones del problema. Como el problema
                    es de minimización, se suma la penalización al valor_fitness final.
        
        funObj1: suma de distancia de cada camión en vacío entre suma de distancia de cada camión total.
        funObj2: suma de la distancia recorrida por cada camión.
        funObj3: suma de fuel consumido por cada camión.
        funObj4: suma del impacto medioambiental provocado por cada camión.
        funObj5: suma del tiempo de espera de cada camión.
        penalizacion: suma de la penalización de cada camión.
       
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
    
    
    def mejor_secuencia(ruta_v, orden_tareas_v, c, pareja):
        """
        A partir de la ruta de un vehículo, las tareas asociadas a esa ruta, la nueva tarea 
        a realizar y el [origen, destino] de dicha tarea, devuelve la ruta que recorre menor distancia
        entre todas las posibles inserciones del [origen, destino] en la ruta dada del vehículo y el
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
    
    
    # DECODING
    def decoding(i,posiciones):
        """
        A partir de la posición de una partícula, generada a través del algoritmo PSO,
        devuelve la ruta a seguir por cada uno de los camiones, de forma que se realice el
        mayor número de tareas posible, respetando la restricción de capacidad de cada camión.
    
        Parameters
        ----------
        i : int.
        posiciones : numpy.ndarray.
    
        Returns
        -------
        R : numpy.ndarray.
        tareas : numpy.ndarray.
        orden_tareas : numpy.ndarray.
    
        """
        tareas = np.zeros(M)
        xref = np.zeros(N)
        yref = np.zeros(N)
    
        # 1. Lista de prioridad de puertos.
        prioridad_tareas = np.argsort(posiciones[i][puertos_parejas[:,0]]) # Ordenamos de forma ascendente las primeras 200 coords de la partícula.
        
        # 2. Matriz de prioridades de vehículos.
        for j in range(N):
            # Puntos de referencia de los vehículos.
            xref[j] = posiciones[i][M+2*j-1]
            yref[j] = posiciones[i][M+2*j]
        dist_euclidea = np.zeros((M,N))
        W = np.zeros((M,N))
        for m in range(M):
            for j in range(N):
                # Distancia Euclídea entre cada puerto  (tarea) y los puntos de orientación de las rutas.
                dist_euclidea[m,j] = math.sqrt((posicion_puertos[puertos_parejas[:,0][m]][0] - xref[j])**2 + (posicion_puertos[puertos_parejas[:,0][m]][1] - yref[j])**2)
            
            # Ordenar los puntos de referencia según su distancia al puerto m.
            W[m] = np.argsort(dist_euclidea[m]) 
        W = W.astype(int) # Matriz de prioridades de vehículos.
         
        R = np.repeat(None, N) # Array de rutas.
        orden_tareas = np.repeat(None, N) # Tareas realizadas por cada vehículo, por orden de realización.
        
        for k in range(M):
            c = prioridad_tareas[k] 
            pareja = puertos_parejas[c] 
            
            b = 0
            while b in range(N):
                vehiculo = W[c,b] # Vehículo más cercano al puerto c
                if peso_tarea[c] <= capacidades[vehiculo]: # Restricción de capacidad:
                    
                    if (R[vehiculo] is None):
                        R[vehiculo] = np.array(pareja)
                        orden_tareas[vehiculo] = np.array([c])
                        
                    # Si ya hay nodos en la ruta, tenemos que insertar la pareja [origen,destino] en la posición más óptima.
                    else:
                        R[vehiculo], orden_tareas[vehiculo] = mejor_secuencia(R[vehiculo], orden_tareas[vehiculo], c, pareja)
                        
                    tareas[c] += 1 # Marcamos que la tarea ha sido realizada.
                    b = N
                    
                else: # Si no se cumple la restricción de capacidad:
                    b += 1
        tareas = tareas.astype(int)
        return R, tareas, orden_tareas 
    
    
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
    
    
    ## Definición de la clase Fuente_alimento.
    class Fuente_alimento():
        """
        Representa a una fuente de alimento del algoritmo ABC.
        """
        def __init__(self):
            """
            Atributos de instancia:
                posicion (numpy.ndarray) : posición de la fuente de alimento (camión) de dimensión M + 2*N.
                
                fit (numpy.ndarray) : valor de fitness de la fuente de alimento (camión).
                
                calidad (float) : valor de calidad de la fuente de alimento (camión), que depende de su valor fit.
                
                cap (float): capacidad de la fuente de alimento (camión).
    
            """
            self.posicion = np.repeat(None, M + 2*N)
            self.fit = np.array(None)
            self.calidad = None
            self.cap = None
    
    ## Definición del enjambre, conjunto de fuentes de alimento.
    Fuentes_alimento = np.array([])
    for i in range(N):
        fuente_alimento = Fuente_alimento()
        Fuentes_alimento = np.append(Fuentes_alimento,fuente_alimento)
    # Posición de cada camión
    posiciones = np.repeat(None,N)
    # Capacidad de cada camión
    capacidades = np.repeat(None,N)
    
    
    ## Inicialización del enjambre.
    # Posición y capacidad de cada fuente de alimento:
    for k in range(N):
        Fuentes_alimento[k].posicion = np.random.uniform(lb, ub, size = M + 2*N)
        posiciones[k] = Fuentes_alimento[k].posicion
        Fuentes_alimento[k].cap = np.random.uniform(700, 1300)
        capacidades[k] = Fuentes_alimento[k].cap
      
    # Valor de fitness de cada fuente de alimento:
    for i in range(N):
        # Decoding:
        rutas_dec,tareas,orden_tareas = decoding(i,posiciones)
        dist_vac,dist_tot,fuel,impacto_med = distancia_fuel_impacto_distvacia(rutas_dec)
        tiempo_espera, penalizaciones = horas(orden_tareas)
        penalizacion_tarea_unica = restriccion_tareas_unica_vez(tareas)
        penalizaciones += penalizacion_tarea_unica
        Fuentes_alimento[i].fit = funObj(dist_vac,dist_tot,fuel,impacto_med,tiempo_espera,penalizaciones)
        registro_fitness[i] = Fuentes_alimento[i].fit 
        
        if i == 0:
            mejor_fitness_global = Fuentes_alimento[i].fit
            mejor_posicion_global = Fuentes_alimento[i].posicion
            mejor_fitness_actual = Fuentes_alimento[i].fit
            peor_fitness_actual = Fuentes_alimento[i].fit
            
        else:
            if Fuentes_alimento[i].fit < mejor_fitness_global:
                mejor_fitness_global = Fuentes_alimento[i].fit
                mejor_posicion_global = Fuentes_alimento[i].posicion
                mejor_fitness_actual = Fuentes_alimento[i].fit
            elif Fuentes_alimento[i].fit > peor_fitness_actual:
                peor_fitness_actual = Fuentes_alimento[i].fit
    
    # Calidad de cada fuente de alimento:
    for k in range(N):
        Fuentes_alimento[k].calidad = (Fuentes_alimento[k].fit - peor_fitness_actual)/(mejor_fitness_actual - peor_fitness_actual)
    
    iteracion_primera = None
    Evolucion[0] = mejor_fitness_global
    
    """
    # Iteración en la que se alcanza por primera vez el valor óptimo con los datos utilizados.
    if round(Evolucion[0], 4) == 296.2148:
        iteracion_primera = 1
    """
    
    print("Iteracion ", str(0), ": Costo(Fitness) = ",str(Evolucion[0]))
    
    
    # 2. PROCESO ITERATIVO.
    ########################
    
    tiempoInicio = time.time()
    
    for it in range(it_max): # Criterio de paro
        
        ## 1. Abejas Empleadas
        for i in range(N):
           # Seleccionar aleatoriamente una solución diferente a 'i':
            rango = np.array(range(N))
            j = np.delete(rango,  i)
            r = random.choice(j)
            # Definir Coeficiente de Aceleración (tamaño de paso):
            phi = a*np.random.uniform(-a,a,M + 2*N)
            # Definir la posición de la nueva fuente de alimento:
            fuente_alimento_i_posicion =  Fuentes_alimento[i].posicion + phi*(Fuentes_alimento[i].posicion - Fuentes_alimento[r].posicion)
            posiciones[i] = fuente_alimento_i_posicion
            
            # Evaluar el fitness de la nueva solución candidata:
            rutas_dec,tareas,orden_tareas = decoding(i,posiciones)  # Decoding
            dist_vac,dist_tot,fuel,impacto_med = distancia_fuel_impacto_distvacia(rutas_dec)
            tiempo_espera, penalizaciones = horas(orden_tareas)
            penalizacion_tarea_unica = restriccion_tareas_unica_vez(tareas)
            penalizaciones += penalizacion_tarea_unica
            fuente_alimento_i_fit = funObj(dist_vac,dist_tot,fuel,impacto_med,tiempo_espera,penalizaciones)
            registro_fitness[i] = fuente_alimento_i_fit
            # Mejor y peor fitness actual:
            mejor_fitness_actual = min(registro_fitness)
            peor_fitness_actual= max(registro_fitness)
            
            # Evaluar la calidad de la nueva fuente de alimento:
            fuente_alimento_i_calidad = (fuente_alimento_i_fit - peor_fitness_actual)/(mejor_fitness_actual - peor_fitness_actual)
            # Comparar fuentes de alimento:
            if fuente_alimento_i_calidad > Fuentes_alimento[i].calidad:
                Fuentes_alimento[i].posicion = fuente_alimento_i_posicion
                Fuentes_alimento[i].fit = fuente_alimento_i_fit
                Fuentes_alimento[i].calidad = fuente_alimento_i_calidad
                trial[i] = 0
            else:
                posiciones[i] = Fuentes_alimento[i].posicion
                registro_fitness[i] = Fuentes_alimento[i].fit
                trial[i] = trial[i] + 1
                
        ## 2. Abejas Observadoras
        # Calcular probabilidades de selección:
        calidades = np.zeros(N)
        for i in range(N):
            calidades[i] = Fuentes_alimento[i].calidad
        P = calidades/sum(calidades)
        
        for k in range(N_Obs):
            # Seleccionar fuente de alimento (SELECCIÓN POR RULETA):
            C = np.cumsum(P)
            i = np.where(np.less_equal(random.random(), C))[0][0] #Índice del primer elemento que cumple que es mayor que el número random.
            
            # Se repite el proceso de las abejas empleadas:
            # Seleccionar aleatoriamente una solución diferente a 'i':
            rango = np.array(range(N))
            j = np.delete(rango,  i)
            r = random.choice(j)
            # Definir Coeficiente de Aceleración (tamaño de paso):
            phi = a*np.random.uniform(-a,a,M + 2*N)
            # Definir la posición de la nueva fuente de alimento:
            fuente_alimento_k_posicion =  Fuentes_alimento[k].posicion + phi*(Fuentes_alimento[k].posicion - Fuentes_alimento[r].posicion)
            posiciones[k] = fuente_alimento_k_posicion
            # Evaluar el fitness de la nueva solución candidata:
            rutas_dec,tareas,orden_tareas = decoding(k,posiciones) # Decoding
            dist_vac,dist_tot,fuel,impacto_med = distancia_fuel_impacto_distvacia(rutas_dec)
            tiempo_espera, penalizaciones = horas(orden_tareas)
            penalizacion_tarea_unica = restriccion_tareas_unica_vez(tareas)
            penalizaciones += penalizacion_tarea_unica
            fuente_alimento_k_fit = funObj(dist_vac,dist_tot,fuel,impacto_med,tiempo_espera,penalizaciones)
            registro_fitness[k] = fuente_alimento_k_fit
            # Mejor y peor fitness actual:
            mejor_fitness_actual = min(registro_fitness)
            peor_fitness_actual= max(registro_fitness)
            # Evaluar la calidad de la nueva fuente de alimento:
            fuente_alimento_k_calidad = (fuente_alimento_k_fit - peor_fitness_actual)/(mejor_fitness_actual - peor_fitness_actual)
            # Comparar fuentes de alimento:
            if fuente_alimento_k_calidad > Fuentes_alimento[k].calidad:
                Fuentes_alimento[k].posicion = fuente_alimento_k_posicion
                Fuentes_alimento[k].fit = fuente_alimento_k_fit
                Fuentes_alimento[k].calidad = fuente_alimento_k_calidad
                trial[k] = 0
            else:
                posiciones[k] = Fuentes_alimento[k].posicion
                registro_fitness[k] = Fuentes_alimento[k].fit
                trial[k] = trial[k] + 1
        
        ## 3. Abejas exploradoras
        for i in range(N):
            if trial[i] >= limit:
                Fuentes_alimento[i].posicion = np.random.uniform(lb, ub, size = M + 2*N)
                posiciones[i] = Fuentes_alimento[i].posicion
                # Evaluar el fitness de la nueva solución:
                rutas_dec,tareas,orden_tareas = decoding(i,posiciones) # Decoding
                dist_vac,dist_tot,fuel,impacto_med = distancia_fuel_impacto_distvacia(rutas_dec)
                tiempo_espera, penalizaciones = horas(orden_tareas)
                penalizacion_tarea_unica = restriccion_tareas_unica_vez(tareas)
                penalizaciones += penalizacion_tarea_unica
                Fuentes_alimento[i].fit = funObj(dist_vac,dist_tot,fuel,impacto_med,tiempo_espera,penalizaciones)
                registro_fitness[i] = Fuentes_alimento[i].fit
                mejor_fitness_actual = min(registro_fitness)
                peor_fitness_actual= max(registro_fitness)
                # Evaluar la calidad de la nueva fuente de alimento:
                Fuentes_alimento[i].calidad = (Fuentes_alimento[i].fit - peor_fitness_actual)/(mejor_fitness_actual - peor_fitness_actual)
                # Reiniciar contador de intentos:
                trial[i] = 0
        
        
        ## Verificar que las partículas no se salgan de los límites lb y ub:
        for i in range(N):
            for j in range(M + 2*N):
                if  Fuentes_alimento[i].posicion[j] < lb:
                    Fuentes_alimento[i].posicion[j] = lb
                    posiciones[i] = Fuentes_alimento[i].posicion
                    # Decoding:
                    rutas_dec,tareas,orden_tareas = decoding(i,posiciones) # Decoding
                    dist_vac,dist_tot,fuel,impacto_med = distancia_fuel_impacto_distvacia(rutas_dec)
                    tiempo_espera, penalizaciones = horas(orden_tareas)
                    penalizacion_tarea_unica = restriccion_tareas_unica_vez(tareas)
                    penalizaciones += penalizacion_tarea_unica
                    Fuentes_alimento[i].fit = funObj(dist_vac,dist_tot,fuel,impacto_med,tiempo_espera,penalizaciones)
                    registro_fitness[i] = Fuentes_alimento[i].fit
                    
                elif Fuentes_alimento[i].posicion[j] > ub:
                    Fuentes_alimento[i].posicion[j] = ub
                    posiciones[i] = Fuentes_alimento[i].posicion
                    # Decoding:
                    rutas_dec,tareas,orden_tareas = decoding(i,posiciones) # Decoding
                    dist_vac,dist_tot,fuel,impacto_med = distancia_fuel_impacto_distvacia(rutas_dec)
                    tiempo_espera, penalizaciones = horas(orden_tareas)
                    penalizacion_tarea_unica = restriccion_tareas_unica_vez(tareas)
                    penalizaciones += penalizacion_tarea_unica
                    Fuentes_alimento[i].fit = funObj(dist_vac,dist_tot,fuel,impacto_med,tiempo_espera,penalizaciones)
                    registro_fitness[i] = Fuentes_alimento[i].fit
                    
            mejor_fitness_actual = min(registro_fitness)
            peor_fitness_actual= max(registro_fitness)
            # Evaluar la calidad de la nueva fuente de alimento
            Fuentes_alimento[i].calidad = (Fuentes_alimento[i].fit - peor_fitness_actual)/(mejor_fitness_actual - peor_fitness_actual)
            
        ## Registro de la mejor solución global
        if mejor_fitness_actual < mejor_fitness_global:
            mejor_fitness_global = mejor_fitness_actual
            ind = np.where(registro_fitness==mejor_fitness_actual)[0][0]
            mejor_posicion_global = Fuentes_alimento[ind].posicion
        
        
        Evolucion[it+1] = mejor_fitness_global
        
        """
        # Iteración en la que se alcanza por primera vez el valor óptimo con los datos utilizados.
        if (iteracion_primera is None) and (round(Evolucion[it+1], 4) == 296.2148):
            iteracion_primera = it
        """
        
        # Mostrar iteración actual y mejor fitness actual
        print("Iteracion ", str(it+1), ": Costo(Fitness) = ",str(Evolucion[it+1]))
   
    tiempoFin = time.time()
    tiempo = tiempoFin - tiempoInicio
    nombreFichero = "ABC_PuertoXYZ_num_particulas_= " + str(N) + ".txt"
    with open(nombreFichero, 'a') as archivo:
        archivo.write(str(Evolucion) + '\n')
        archivo.write("iteracion = " + str(iteracion_primera) + '\n')
        archivo.write(str(tiempo) + '\n')
        archivo.write('FIN\n' + '\n')
        