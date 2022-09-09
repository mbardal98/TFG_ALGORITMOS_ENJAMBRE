
# Algoritmo de Colonia Artificial de Abejas  (ABC).
# Autora: Marta Bardal Castrillo.

# Librerías utilizadas.
import numpy as np
import random


# FUNCIÓN OBJETIVO
"""
Función sencilla que define el problema de minimización a resolver.
"""
funObj = lambda x: (x[0]+6) * (x[1]+20)


# 1. INICIALIZACIÓN
####################

## Configuración de parámetros
N = 200 # Número de fuentes de alimento.
d = 2 # Dimensión del espacio de soluciones
lb = [0, 0] # Límites inferiores del espacio de búsqueda
ub = [9, 9] # Límites superiores del espacio de búsqueda
it_max = 30 # Número máximo de iteraciones (criterio de paro).
N_Obs = N # Número de abejas observadoras
a = 1.0 # Coeficiente de aceleración (tamaño de paso)
limit = 30 # Número máximo de intentos (criterio de abandono de fuente de alimento)

trial = np.zeros(N) # Contador de intentos
Evolucion = np.zeros(it_max + 1)
# Registro del fitness en cada iteración de cada fuente de alimento para poder tomar el mejor
# y el peor fitness con el objetivo de calcular la calidad de cada fuente en cada iteración:
registro_fitness = np.repeat(None, N) 

## Definición de la clase Fuente_alimento.
class Fuente_alimento():
    """
    Representa a una fuente de alimento del algoritmo ABC.
    """
    def __init__(self):
        """
        Atributos de instancia:
            posicion (numpy.ndarray) : posición de la fuente de alimento de dimensión d.
            
            fit (numpy.ndarray) : valor de fitness de la fuente de alimento.
            
            calidad (float) : valor de calidad de la fuente de alimento, que depende de su valor fit.
        """
        self.posicion = np.repeat(None, d)
        self.fit = np.array(None)
        self.calidad = None

## Definición del enjambre.
Fuentes_alimento = np.array([])
for i in range(N):
    fuente_alimento = Fuente_alimento()
    Fuentes_alimento = np.append(Fuentes_alimento,fuente_alimento)

## Inicialización del enjambre.
for k in range(N):
    Fuentes_alimento[k].posicion = np.array([random.uniform(lb[0],ub[0]), random.uniform(lb[1],ub[1])], dtype=object) 
    Fuentes_alimento[k].fit = funObj(Fuentes_alimento[k].posicion)
    registro_fitness[k] = Fuentes_alimento[k].fit 
    if k == 0:
        mejor_fitness_global = Fuentes_alimento[k].fit
        mejor_posicion_global = Fuentes_alimento[k].posicion
        peor_fitness_actual = Fuentes_alimento[k].fit
    else:
        if Fuentes_alimento[k].fit < mejor_fitness_global:
            mejor_fitness_global = Fuentes_alimento[k].fit
            mejor_posicion_global = Fuentes_alimento[k].posicion
        if Fuentes_alimento[k].fit > peor_fitness_actual: 
            peor_fitness_actual = Fuentes_alimento[k].fit # Peor fitness de esta iteración (inicialización). 
    mejor_fitness_actual = mejor_fitness_global

## Evaluar la calidad de las fuentes de alimento
for k in range(N):
    Fuentes_alimento[k].calidad = (Fuentes_alimento[k].fit - peor_fitness_actual)/(mejor_fitness_actual - peor_fitness_actual)

Evolucion[0] = mejor_fitness_global
print("Iteracion ", str(0), ": Costo(Fitness) = ",str(Evolucion[0]))


# 2. PROCESO ITERATIVO.
########################

for it in range(it_max): # Criterio de paro
    
    ## 1. Abejas Empleadas
    for i in range(N):
        # Seleccionar aleatoriamente una solución diferente a 'i':
        rango = np.array(range(N))
        j = np.delete(rango,  i)
        r = random.choice(j)
        # Definir Coeficiente de Aceleración (tamaño de paso):
        phi = a*np.random.uniform(-a,a,d)
        # Definir la posición de la nueva fuente de alimento:
        fuente_alimento_i_posicion =  Fuentes_alimento[i].posicion + phi*(Fuentes_alimento[i].posicion - Fuentes_alimento[r].posicion)
        # Evaluar el fitness de la nueva solución candidata:
        fuente_alimento_i_fit = funObj(fuente_alimento_i_posicion)
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
        phi = a*np.random.uniform(-a,a,d)
        # Definir la posición de la nueva fuente de alimento:
        fuente_alimento_k_posicion =  Fuentes_alimento[k].posicion + phi*(Fuentes_alimento[k].posicion - Fuentes_alimento[r].posicion)
        # Evaluar el fitness de la nueva solución candidata:
        fuente_alimento_k_fit = funObj(fuente_alimento_k_posicion)
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
            registro_fitness[k] = Fuentes_alimento[k].fit
            trial[k] = trial[k] + 1
    
    ## 3. Abejas exploradoras
    for i in range(N):
        if trial[i] >= limit:
            Fuentes_alimento[i].posicion = np.array([random.uniform(lb[0],ub[0]), random.uniform(lb[1],ub[1])], dtype=object) 
            # Evaluar el fitness de la nueva solución candidata:
            Fuentes_alimento[i].fit = funObj(Fuentes_alimento[i].posicion)
            registro_fitness[i] = Fuentes_alimento[i].fit
            mejor_fitness_actual = min(registro_fitness)
            peor_fitness_actual= max(registro_fitness)
            # Evaluar la calidad de la nueva fuente de alimento:
            Fuentes_alimento[i].calidad = (Fuentes_alimento[i].fit - peor_fitness_actual)/(mejor_fitness_actual - peor_fitness_actual)
            # Reiniciar contador de intentos:
            trial[i] = 0
    
    
    ## Verificar que las partículas no se salgan de los límites lb y ub:
    for i in range(N):
        for j in range(d):
            if  Fuentes_alimento[i].posicion[j] < lb[j]:
                Fuentes_alimento[i].posicion[j] = lb[j]
                Fuentes_alimento[i].fit = funObj(Fuentes_alimento[i].posicion)
                registro_fitness[i] = Fuentes_alimento[i].fit
                
            elif Fuentes_alimento[i].posicion[j] > ub[j]:
                Fuentes_alimento[i].posicion[j] = ub[j]
                Fuentes_alimento[i].fit = funObj(Fuentes_alimento[i].posicion)
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
    
    # Registro histórico de las mejores soluciones
    Evolucion[it+1] = mejor_fitness_global
    
    # Mostrar iteración actual y mejor fitness actual
    print("Iteracion ", str(it+1), ": Costo(Fitness) = ",str(Evolucion[it+1]))
  