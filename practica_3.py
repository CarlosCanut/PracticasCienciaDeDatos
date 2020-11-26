import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import requests
import io
import math
from scipy import stats
import numpy as np
import researchpy as rp
import statsmodels.api as sm
import pprint

# -----------------------------------------------------------------------------
# ---------------------------- EXTRACCIÓN DE DATOS ----------------------------
# -----------------------------------------------------------------------------
string_csv = requests.get("https://gitlab.com/drvicsana/ciencia-datos/-/raw/master/datasets/fifa/players_20.csv").content
string_stream = io.StringIO(string_csv.decode("utf-8"))
player_df = pd.read_csv(string_stream)


# -----------------------------------------------------------------------------
# ---------------------------- EXPLORACIÓN DE DATOS ---------------------------
# -----------------------------------------------------------------------------

# numero de filas y columnas del df
pprint.pprint(player_df.shape)

# descripcion de los datos
pprint.pprint(player_df.describe())

# encabezado de los datos
pprint.pprint(player_df.head(5))

# descripcion categorica
print(player_df.describe(include=["object"]))


# -----------------------------------------------------------------------------
# ---------------------------- FILTRADO DE DATOS ------------------------------
# -----------------------------------------------------------------------------


# filtrado para equipos españoles
spanish_teams = ["Real Madrid", "FC Barcelona", "Atlético Madrid", "Valencia CF", "Villarreal CF", "Real Valladolid CF", 
                 "Athletic Club de Bilbao", "Cádiz CF", "Granada CF", "Real Betis", "RCD Espanyol", "Real Sociedad", "Deportivo Alavés", "Elche CF", "Sevilla FC", 
                 "CA Osasuna", "SD Eibar", "RC Celta", "Levante UD", "SD Huesca" ]

# posiciones posibles de cada jugador
defender_positions = [ "SW", "RWB", "LWB", "RB", "LB", "CB" ]
midfield_positions = [ "DM", "RW", "LW", "LM", "RM", "CM", "AM" ]
forward_positions = [ "CF", "RF", "LF", "ST" ]
goalkeeper_positions = ["GK"]


# crear nuevo df con jugadores españoles
spanish_players_df = player_df[ player_df.club.isin(spanish_teams) ]

# descripcion de los datos
pprint.pprint(spanish_players_df.describe())

# encabezado de los datos
pprint.pprint(spanish_players_df.head(5))

# descripcion categorica
print(spanish_players_df.describe(include=["object"]))


# -----------------------------------------------------------------------------
# ----------------------- CREACIÓN DE NUEVOS DATOS (COLUMNAS) -----------------
# -----------------------------------------------------------------------------

########### Funciones para definir la posición principal de un jugador ###########

# mete las posiciones de un jugador en una lista
def from_string_position_to_list(positions_string):
    return positions_string.replace(" ", "").split(",")

# dice si pertenece a la lista prospective
def is_position(positions_string, prospective_positions):
  list_positions = from_string_position_to_list(positions_string)
  found = False
  for position in list_positions :
    if position in prospective_positions :
      found = True
  return found

# decide a que posición pertenece
def determine_main_position(positions_string):
  count_defender = 0
  count_midfield = 0
  count_forward = 0
  count_goalkeeper = 0
  list_positions = from_string_position_to_list(positions_string)
  for position in list_positions :
    if position in defender_positions :
      count_defender = count_defender + 1
    elif position in midfield_positions :
      count_midfield = count_midfield + 1
    elif position in forward_positions :
      count_forward = count_forward + 1
    elif position in goalkeeper_positions :
      count_goalkeeper = count_goalkeeper + 1
  
  return max( [ (count_defender, "DF"), (count_midfield, "MF"), (count_forward, "FW"), (count_goalkeeper, "GK") ] )[1]


# funciones lambda que dice si un jugador es o no de una posición (esto testea todas las posiciones en las que juega el jugador indicado).
is_defender = lambda s : is_position(s, defender_positions)
is_midfielder = lambda s : is_position(s, midfield_positions)
is_forward = lambda s : is_position(s, forward_positions)
is_goalkeeper = lambda s: is_position(s, goalkeeper_positions)

# crea la columna "main_position" en la cual muestra la posición del player
spanish_players_df = spanish_players_df.assign( main_position = spanish_players_df.apply( lambda row: determine_main_position(row.player_positions), axis=1 ) )

# * determina si en algun caso el jugador es defensa * (utiliza la lista con todas las posiciones en las que juega el jugador).
spanish_players_df = spanish_players_df.assign( is_defender = spanish_players_df.apply( lambda row : is_position(row.player_positions, defender_positions), axis=1 ))

# determina si el jugador tiene como posicion principal defensa
spanish_players_df = spanish_players_df.assign( is_defender = spanish_players_df.apply( lambda row : True if ("DF" == row.main_position) else False, axis=1 ))
# determina si el jugador tiene como posicion principal delantero
spanish_players_df = spanish_players_df.assign( is_forward = spanish_players_df.apply( lambda row : True if ("FW" == row.main_position) else False, axis=1 ))



########### Definir de que continente es cada jugador en una nueva columna ###########

# definir cada continente
country_to_continent = { "Argentina": "America", "Slovenia": "Europe", "Belgium":"Europe", "Germany": "Europe", "Croatia":"Europe",
                        "Spain": "Europe", "Uruguay":"America", "France":"Europe", "Brazil":"America", "Costa Rica":"America",
                        "Netherlands":"Europe", "Wales": "Europe", "Colombia":"America", "Chile":"America", "Serbia":"Europe",
                        "Portugal":"Europe", "Ghana":"Africa", "Montenegro":"Europe", "Mexico":"America", "Denmark":"Europe",
                        "Central African Rep.":"Africa", "Czech Republic":"Europe", "Algeria":"Africa", "Israel":"Asia",
                        "England":"Europe", "Slovakia":"Europe", "Dominican Republic":"America", "Cameroon":"Africa",
                        "China PR":"Asia", "Japan":"Asia", "Russia":"Asia", "Norway":"Europe", "Turkey":"Asia",
                        "FYR Macedonia":"Europe", "Senegal":"Africa", "Morocco":"Africa", "Italy":"Europe",
                        "Ukraine":"Europe", "Korea Republic":"Asia", "Nigeria":"Africa", "Sweden":"Europe", "Venezuela":"America",
                        "Bosnia Herzegovina":"Europe", "Ivory Coast":"Africa", "Romania":"Europe", "Equatorial Guinea":"Africa",
                        "Ecuador":"America", "Albania":"Europe", "Mauritania":"Africa", "Benin":"Africa", "Switzerland":"Europe",
                        "DR Congo":"Africa"}


spanish_players_df.apply( lambda row: row.age*2, axis=1 )

# crea la columna "continent" en la cual muestra el continente del player
spanish_players_df = spanish_players_df.assign( continent = spanish_players_df.apply(lambda row: country_to_continent[row.nationality], axis=1) )



# -----------------------------------------------------------------------------
# ---------------- ESTADISTICA DESCRIPTIVA (DATOS CATEGÓRICOS) ----------------
# -----------------------------------------------------------------------------

# frecuencia absoluta por posición principal
pprint.pprint(spanish_players_df.main_position.value_counts())

# frecuencia relativa por posicion principal
pprint.pprint(spanish_players_df.main_position.value_counts(normalize=True))


# gráfico de queso de jugadores por posición
count_data = spanish_players_df.main_position.value_counts()
plt.figure(figsize=(10, 10))
fig = plt.pie( count_data, labels = count_data.index, autopct="%.2f", textprops={"fontsize":14} )

# gráfico de queso de jugadores por continente
count_data = spanish_players_df.continent.value_counts()
plt.figure(figsize=(10,10))
fig = plt.pie( count_data, labels = count_data.index, autopct="%.2f", textprops={"fontsize":14} )

# gráfico de barras donde se muestran la frecuencia de cada posición
count_data = spanish_players_df.main_position.value_counts(normalize=True)
axes = sns.barplot( x=count_data.index, y=count_data )
axes.set_xlabel("Position")
axes.set_ylabel("Percentage")
axes.set_title("Player positions")

# gráfico de barras donde se muestran la frecuencia del continente
count_data = spanish_players_df.continent.value_counts(normalize=True)
axes = sns.barplot( x=count_data.index, y=count_data )
axes.set_xlabel("Continent")
axes.set_ylabel("Percentage")
axes.set_title("Player continent")

#### TABLA DE CONTINGENCIAS ####

# tabla de contingencia del pie que usan los jugadores por posición principal
pd.crosstab( spanish_players_df.preferred_foot, spanish_players_df.main_position )

# lo mismo pero normalizado
pd.crosstab( spanish_players_df.preferred_foot, spanish_players_df.main_position, normalize=True )

# frecuencias condicionada a la preferencia del pie
pd.crosstab( spanish_players_df.preferred_foot, spanish_players_df.main_position, normalize = "index" )

# ahora las frecuencias condicionadas a la posición que juegan
pd.crosstab( spanish_players_df.preferred_foot, spanish_players_df.main_position, normalize = "columns" )

# ahora las frecuencias condicionadas al continente al que pertenecen
pd.crosstab( spanish_players_df.preferred_foot, spanish_players_df.continent, normalize = "columns" )


# -----------------------------------------------------------------------------
# ---------------- ESTADISTICA DESCRIPTIVA (DATOS NUMÉRICOS) ------------------
# -----------------------------------------------------------------------------

#### HISTOGRAMAS ####

# histograma del overall del conjunto de datos, bins: numero de cubetas en el histograma
sns.histplot(data=spanish_players_df, x="overall", bins=30)

# con probabilidad (la suma de las alturas de las cubetas será 1)
sns.histplot(data=spanish_players_df, x="overall", bins=30, stat="probability")

# con densidad (el área del histograma será 1)
sns.histplot(data=spanish_players_df, x="overall", bins=30, stat="density")

# comparación de distintos histogramas 
# - x: datos a analizar (variable aleatoria) , 
# - hue: crea un histograma para cada clase de esta columna (diferencia entre histogramas).
sns.histplot(data=spanish_players_df, x="overall", bins=30, hue="preferred_foot", stat="density", common_norm=False)

# histograma por edad (densidades)
sns.histplot(data=spanish_players_df, x="age", bins=30, stat="density")

# histograma por salarios (densidades)
sns.histplot(data=spanish_players_df, x="wage_eur", bins=40, stat="density")

# comparación de histogramas con variable aleatoria edad diferenciando por dependencia de pie dominante
sns.histplot(data=spanish_players_df, x="age", bins=30, hue="preferred_foot", stat="density", common_norm=False)

#### BOX WHISKERS ####

# Caja Box Whiskers con info de los jugadores
sns.boxplot( x=spanish_players_df.overall )

# Dos cajas Box Whiskers en el mismo plot
sns.boxplot( x=spanish_players_df.preferred_foot, y=spanish_players_df.overall )



# -----------------------------------------------------------------------------
# ----------------------- INFERENCIA SOBRE PROPORCIONES -----------------------
# -----------------------------------------------------------------------------


# cogemos una muestra
small_sample = player_df.sample(n=100)

# calculamos la proporcion en la muestra (contamos los jugadores zurdos)
p = small_sample[ small_sample.preferred_foot == "Left" ].shape[0]/100
print(p)

#### INTERVALO DE CONFIANZA ####

# Genera 10 datos aleatorios procedentes de una normal con media 20 y desviación típica 2
random_data = stats.norm.rvs(loc=20, scale=2, size= 10)
print(random_data)
# Obtiene el valor de la función de densidad para el punto 21.3 en una normal con media 20 y desviación típica 2
print( stats.norm.pdf( 21.3, loc=20, scale=2 ) )
# Obtiene la probabilidad de observar valores menores o iguales a cero en una distribución normal con media 0 y desviación típica 1
print( stats.norm.cdf( 0 ,loc=0, scale=1 ) )
# Obtiene el percentil 97.5 (mayor o igual que el 97.5% de las posibles observaciones) para una normal con media 0 y desviación típica 1
print( stats.norm.ppf( 0.975, loc = 0, scale = 1 ) )
# Obtiene los extremos del intervalo centrado en la media que contiene el 95% de las observaciones en una normal con media 0 y desviación típica 1
print( stats.norm.interval( 0.95, loc=0, scale=1 ) )
# construimos el intervalo de confianza 
lower_bound = 0
upper_bound = 0
Zalpha = stats.norm.ppf( 0.975, loc=0, scale=1 )
deviation = Zalpha * math.sqrt( p*(1-p)/100 ) 
interval = ( p - deviation, p + deviation )
print(interval)


# parametro poblacional de los jugadores de FIFA zurdos
true_p = player_df[ player_df.preferred_foot == "Left" ].shape[0]/player_df.shape[0]
print(true_p)

# intervalo de confianza de 95% con 100 muestras
small_sample = player_df.sample(n=100)
p = small_sample[ small_sample.preferred_foot == "Left" ].shape[0]/100
Zalpha = stats.norm.ppf( 0.975, loc=0, scale=1 )
deviation = Zalpha * math.sqrt( p*(1-p)/100 ) 
interval = ( p - deviation, p + deviation )
print("Intervalo de 95% (100 muestras) -> ",interval)
print("Tamaño -> ",((p + deviation)-(p - deviation)))


# intervalo de confianza de 95% con 1000 muestras
sample = player_df.sample(n=1000)
p1 = sample[ sample.preferred_foot == "Left" ].shape[0]/1000
Zalpha = stats.norm.ppf( 0.975, loc=0, scale=1 )
deviation_1 = Zalpha * math.sqrt( p1*(1-p1)/100 ) 
interval = ( p1 - deviation_1, p1 + deviation_1 )
print("Intervalo de 95% (1000 muestras) -> ",interval)
print("Tamaño -> ",((p1 + deviation_1)-(p1 - deviation_1)))



# se hacen 100 muestras en las que se testea si el intervalo de confianza contiene o no el valor, se representa una linea roja 
# donde, si cruza con el intervalo es que ha acertado, tiene un % de acierto del 95%.
number_success = 0
intervals = []
n_rep = 100
for i in range(n_rep):
  n = 100

  small_sample = player_df.sample(n=n)

  p = small_sample[ small_sample.preferred_foot == "Left" ].shape[0]/n

  lower_bound = 0
  upper_bound = 0
  Zalpha = stats.norm.ppf( 0.975, loc=0, scale=1 )
  deviation = Zalpha * math.sqrt( p*(1-p)/n ) 
  interval = ( p - deviation, p + deviation )

  intervals.append(interval)

  if true_p >= interval[0] and true_p <= interval[1]:
    number_success = number_success + 1


print(number_success/n_rep)

plt.figure(figsize=(30,10))
plt.errorbar( x=list(range(1, n_rep+1)), y=[ (x[1]+x[0])/2 for x in intervals ], yerr=[ (x[1]-x[0])/2 for x in intervals ], fmt='o' )
plt.hlines( xmin=1, xmax=100, y=true_p, colors="red" )


#### PRUEBAS DE NORMALIDAD ####

sample = player_df.sample(n=100)

# prueba de normalidad de asimetria
#   H0:  El coeficiente de asimetría (CA) de la población sigue una asimetria propia de una población normal (CA = 0)
#   H1:  El coeficiente de asimetría (CA) de la población sigue una asimetría diferente de una población normal (CA != 0)
# si pvalue<=0.05 -> H1
# si pvalue>0.05 -> H0
print( stats.skewtest( sample.overall ) )

# prueba de normalidad de curtosis
#   H0:  El coeficiente de curtosis (CC) de la población sigue una curtosis propia de una población normal (CC=0)
#   H1:  El coeficiente de curtosis (CC) de la población tiene una curtosis diferente a la de una población normal (CC!=0)
# si pvalue<=0.05 -> H1
# si pvalue>0.05 -> H0
print( stats.kurtosistest( sample.overall ) )

# prueba de Kolmogorov-Smirnov
#   H0:  La función de distribución acumulada se comporta como la función de distribución acumulada de una distribución normal.
#   H1: La función de distribución acumulada difiere de la función de distribución acumulada de una distribución normal.
# si pvalue<=0.05 -> H1
# si pvalue>0.05 -> H0
stats.kstest( (sample.overall-sample.overall.mean())/sample.overall.std(), 'norm'  )

# prueba de Shapiro-Wilk
#   H0:  La variable aleatoria estudiada procede de una población normal
#   H1: La variable aleatoria estudiada no procede de una población normal
# si pvalue<=0.05 -> H1
# si pvalue>0.05 -> H0
stats.shapiro( sample.overall )


#### PAPEL PROBABILISTICO ####

# gráfico probabilistico (papel probabilistico) de overall con respecto a una normal
plot = sm.ProbPlot( sample.overall, dist="norm" )
plot.probplot(line="r")


#### PRUEBAS DE NORMALIDAD Y MUESTRAS GRANDES ####

# vamos a ver que ocurre con las pruebas en las que no pudimos
# descartar normalidad cuando la muestra es proporcionada es grande
new_sample1 = player_df.sample(n=1000)
new_sample2 = player_df.sample(n=2000)
new_sample3 = player_df.sample(n=4000)

print( stats.shapiro( new_sample1.overall ) ) 
print( stats.shapiro( new_sample2.overall ) ) 
print( stats.shapiro( new_sample3.overall ) ) 

print(stats.kstest( (new_sample1.overall-new_sample1.overall.mean())/new_sample1.overall.std(), 'norm'  ))
print(stats.kstest( (new_sample2.overall-new_sample2.overall.mean())/new_sample2.overall.std(), 'norm'  ))
print(stats.kstest( (new_sample3.overall-new_sample3.overall.mean())/new_sample3.overall.std(), 'norm'  ))

# analisis de toda la población 
# (kde -> kernel density estimate: estimiacion de la funcion de densidad no parametrica)
axes = sns.kdeplot( (player_df.overall - player_df.overall.mean() )/(player_df.overall.std()), common_norm=False  )
# representación de una normal para comparar con el kde
sns.lineplot( y=[ stats.norm.pdf(x) for x in np.arange(-3, 3, 0.01) ], x= [x for x in np.arange(-3, 3, 0.01)] )
# histograma
sns.histplot(player_df.overall, bins=25)
# gráfico de probabilidad normal
plot = sm.ProbPlot( player_df.overall, dist="norm" )
plot.probplot(line="r")


# -----------------------------------------------------------------------------
# ------------ INFERENCIA SOBRE LA MEDIA DE POBLACIONES NORMALES --------------
# -----------------------------------------------------------------------------

#### INTERVALO DE CONFIANZA CON T DE STUDENT ####

# genera 10 datos aleatorios procedentes de una t de Student con 4 grados de libertad
# rvs -> random variates of given type
random_data = stats.t.rvs( 4, size= 10)
print(random_data)

# obtiene el valor de la función de densidad para el punto 1 en una t de Student con 4 grados de libertad
print( stats.t.pdf( 1, 4 ) )

# obtiene la probabilidad de observar valores menores o iguales a cero en una distribución t de student con 4 grados de libertad 
print( stats.t.cdf( 0, 4 ) )

# obtiene el percentil 97.5 (mayor o igual que el 97.5% de las posibles observaciones) para una t de student con 4 grados de libertad 
print( stats.t.ppf( 0.975, 4 ) )

# obtiene los extremos del intervalo centrado en 0 que contiene el 95% de las observaciones en una t de student con 4 grados de libertad 
print( stats.t.interval( 0.95, 4 ) )


# intervalo de confianza del 95%
lower_bound = 0
upper_bound = 0
x = sample.overall.mean()
n = sample.shape[0]
t = stats.t.ppf(0.975, n-1 )
s = sample.overall.std()
lower_bound = x - t * s/math.sqrt(n)
upper_bound = x + t * s/math.sqrt(n)
print(lower_bound, upper_bound)
print("media muestral -> ",x)


# analisis de normalidad para el peso en kg de los jugadores
sample = player_df.sample(n=100)
print(" alpha -> 0.05 \n")
print( stats.skewtest( sample.weight_kg ) , "\n -> H0:  El coeficiente de asimetría (CA) de la población sigue una asimetria propia de una población normal (CA = 0)\n" )
print( stats.kurtosistest( sample.weight_kg ) , "\n -> H0:  El coeficiente de curtosis (CC) de la población sigue una curtosis propia de una población normal (CC=0)\n")
print( stats.kstest( (sample.weight_kg-sample.weight_kg.mean())/sample.weight_kg.std(), 'norm'  ) , "\n -> H0:  La función de distribución acumulada se comporta como la función de distribución acumulada de una distribución normal.\n")
print( stats.shapiro( sample.weight_kg ) , "\n -> H0:  La variable aleatoria estudiada procede de una población normal\n")
plot = sm.ProbPlot( sample.weight_kg, dist="norm" )
plot.probplot(line="r")
print("Despues de hacer varios tipos de pruebas de normalidad y todas diciendo que la distribución sigue una normal, concluimos que sigue una normal.\n")
# intervalo de confianza para la media de la variable weight_kg
lower_bound = 0
upper_bound = 0
x = sample.weight_kg.mean()
n = sample.shape[0]
t = stats.t.ppf(0.975, n-1 )
s = sample.weight_kg.std()
lower_bound = x - t * s/math.sqrt(n)
upper_bound = x + t * s/math.sqrt(n)
print("(lower_bound: ",lower_bound,", upper_bound: ", upper_bound,")")
print("media muestral -> ",x)



# -----------------------------------------------------------------------------
# ----------- INFERENCIA SOBRE LA VARIANZA EN POBLACIONES NORMALES ------------
# -----------------------------------------------------------------------------

#### INTERVALO DE CONFIANZA DE CHI-CUADRADO ####

# genera 10 datos aleatorios procedentes de una Chi cuadrado con 4 grados de libertad
random_data = stats.chi2.rvs( 4, size= 10)
print(random_data)

# obtiene el valor de la función de densidad para el punto 2 en una chi cuadrado con 4 grados de libertad
print( stats.chi2.pdf( 2, 4 ) )

# obtiene la probabilidad de observar valores menores o iguales a 1 en una distribución chi cuadrado con 4 grados de libertad 
print( stats.chi2.cdf( 1, 4 ) )

# obtiene el percentil 97.5 (mayor o igual que el 97.5% de las posibles observaciones) para una chi cuadrado 4 grados de libertad 
print( stats.chi2.ppf( 0.975, 4 ) )

# obtiene los extremos del intervalo que contiene el 95% de las observaciones en una chi cuadrado con 4 grados de libertad 
print( stats.chi2.interval( 0.95, 4 ) )


# intervalo de confianza 95% para la varianza poblacional de overall
lower_bound = 0
upper_bound = 0
n = sample.shape[0]
s2 = sample.overall.std()
chi1 = stats.chi2.ppf(1-0.05/2, n - 1 )
chi2 = stats.chi2.ppf(0.05/2, n - 1 )
lower_bound = (n-1)*s2/chi1
upper_bound = (n-1)*s2/chi2
print(lower_bound, upper_bound)
print("varianza muestral -> ",s2)

# intervaol de confianza 99% para la varianza poblacional de weight_kg
lower_bound = 0
upper_bound = 0
n = sample.shape[0]
s2 = sample.weight_kg.std()
chi1 = stats.chi2.ppf(1-0.01/2, n - 1 )
chi2 = stats.chi2.ppf(0.01/2, n - 1 )
lower_bound = (n-1)*s2/chi1
upper_bound = (n-1)*s2/chi2
print(lower_bound, upper_bound)
print("varianza muestral -> ",s2)



# -----------------------------------------------------------------------------
# ---------------------- EXPLORACION DE DATOS (TAREA) -------------------------
# -----------------------------------------------------------------------------

# analizar normalidad y sacar los intervalos de confianza de media y varianza de 3 muestras del conjunto

sample = spanish_players_df.sample(n=100)

# histogramas
plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
sns.histplot(data=spanish_players_df, x="potential", bins=30)
plt.title('Potential')
plt.subplot(1,3,2)
sns.histplot(data=spanish_players_df, x="international_reputation", bins=30)
plt.title('International Reputation')
plt.subplot(1,3,3)
sns.histplot(data=spanish_players_df, x="power_stamina", bins=30)
plt.title('Power Stamina')

# Gráficos probabilisticas
plt.figure()
sm.ProbPlot( sample.potential, dist="norm" ).probplot(line="r")
plt.title('Potential')
sm.ProbPlot( sample.international_reputation, dist="norm" ).probplot(line="r")
plt.title('International Reputation')
sm.ProbPlot( sample.power_stamina, dist="norm" ).probplot(line="r")
plt.title('Power Stamina')


# Pruebas de normalidad
print(" alpha -> 0.05 \n")

print("       Potential:\n")
print( stats.skewtest( sample.potential ) , "\n    -> H0:  El coeficiente de asimetría (CA) de la población sigue una asimetria propia de una población normal (CA = 0)" )
print( stats.kurtosistest( sample.potential ) , "\n    -> H0:  El coeficiente de curtosis (CC) de la población sigue una curtosis propia de una población normal (CC=0)")
print( stats.kstest( (sample.potential-sample.potential.mean())/sample.potential.std(), 'norm'  ) , "\n    -> H0:  La función de distribución acumulada se comporta como la función de distribución acumulada de una distribución normal.")
print( stats.shapiro( sample.potential ) , "\n     -> H0:  La variable aleatoria estudiada procede de una población normal")
print("**Despues de hacer varios tipos de pruebas de normalidad y todas diciendo que la distribución sigue una normal, concluimos que sigue una normal.**\n")

print("\n\n       International Reputation:\n")
print( stats.skewtest( sample.international_reputation ) , "\n    -> H1:  El coeficiente de asimetría (CA) de la población sigue una asimetría diferente de una población normal (CA != 0)" )
print( stats.kurtosistest( sample.international_reputation ) , "\n    -> H1:  El coeficiente de curtosis (CC) de la población tiene una curtosis diferente a la de una población normal (CC!=0)")
print( stats.kstest( (sample.international_reputation-sample.international_reputation.mean())/sample.international_reputation.std(), 'norm'  ) , "\n    -> H1:  La función de distribución acumulada difiere de la función de distribución acumulada de una distribución normal.")
print( stats.shapiro( sample.international_reputation ) , "\n     -> H1:  La variable aleatoria estudiada no procede de una población normal")
print("**Despues de hacer varios tipos de pruebas de normalidad y todas diciendo que la distribución NO sigue una normal, concluimos que NO sigue una normal.**\n")

print("\n\n       Power Stamina:\n")
print( stats.skewtest( sample.power_stamina ) , "\n    -> H1:  El coeficiente de asimetría (CA) de la población sigue una asimetría diferente de una población normal (CA != 0)" )
print( stats.kurtosistest( sample.power_stamina ) , "\n    -> H0:  El coeficiente de curtosis (CC) de la población sigue una curtosis propia de una población normal (CC=0)")
print( stats.kstest( (sample.power_stamina-sample.power_stamina.mean())/sample.power_stamina.std(), 'norm'  ) , "\n    -> H0:  La función de distribución acumulada se comporta como la función de distribución acumulada de una distribución normal.")
print( stats.shapiro( sample.power_stamina ) , "\n     -> H1:  La variable aleatoria estudiada no procede de una población normal")
print("**Despues de hacer varios tipos de pruebas de normalidad, aunque el de asimetría y la de shapiro dicen que la distribución no sigue una normal, observando el gráfico probabilistico y el histograma podemos concluir que sigue una normal.**\n")


# Intervalos de confianza

# Media
print("\n       Media:")
print("Potential")
lower_bound = 0
upper_bound = 0
x = sample.potential.mean()
n = sample.shape[0]
t = stats.t.ppf(0.975, n-1 )
s = sample.potential.std()
lower_bound = x - t * s/math.sqrt(n)
upper_bound = x + t * s/math.sqrt(n)
print("   Hay un 95% de probabilidad de que las muestras tengan una media de entre: (",lower_bound,", ", upper_bound,").")

print("Power Stamina")
lower_bound_1 = 0
upper_bound_1 = 0
x_1 = sample.power_stamina.mean()
n_1 = sample.shape[0]
t_1 = stats.t.ppf(0.975, n_1-1 )
s_1 = sample.power_stamina.std()
lower_bound_1 = x_1 - t_1 * s_1/math.sqrt(n_1)
upper_bound_1 = x_1 + t_1 * s_1/math.sqrt(n_1)
print("   Hay un 95% de probabilidad de que las muestras tengan una media de entre: (",lower_bound_1,", ", upper_bound_1,").")

# Varianza
print("\n       Varianza:")
print("Potential")
lower_bound_2 = 0
upper_bound_2 = 0
n_2 = sample.shape[0]
s2_2 = sample.potential.std()
chi1_2 = stats.chi2.ppf(1-0.05/2, n_2 - 1 )
chi2_2 = stats.chi2.ppf(0.05/2, n_2 - 1 )
lower_bound_2 = (n_2-1)*s2_2/chi1_2
upper_bound_2 = (n_2-1)*s2_2/chi2_2
print("   Hay un 95% de probabilidad de que las muestras tengan una varianza de entre: (",lower_bound_2,", ", upper_bound_2,").")

print("Power Stamina")
lower_bound_3 = 0
upper_bound_3 = 0
n_3 = sample.shape[0]
s2_3 = sample.power_stamina.std()
chi1_3 = stats.chi2.ppf(1-0.05/2, n_3 - 1 )
chi2_3 = stats.chi2.ppf(0.05/2, n_3 - 1 )
lower_bound_3 = (n_3-1)*s2_3/chi1_3
upper_bound_3 = (n_3-1)*s2_3/chi2_3
print("   Hay un 95% de probabilidad de que las muestras tengan una varianza de entre: (",lower_bound_3,", ", upper_bound_3,").")


