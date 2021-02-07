# -*- coding: utf-8 -*-
"""Proyecto Ciencia de Datos Carlos Canut.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rtotsoeGlLywim8yRSmL4xpTXb7SlqQu

>![Google's logo](https://logodownload.org/wp-content/uploads/2014/09/lol-league-of-Legends-logo-2-1.png)

# Selección del conjunto de datos
"""

!pip install statsmodels
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import math
from statsmodels.formula.api import ols

"""Para este proyecto vamos a utilizar un conjunto de datos con todas las partidas de las principales ligas a nivel internacional de la escena de league of legends, para esto utilizaremos los datos que se ofrecen de manera pública y gratuita en la web de Oracle's Elixir (https://oracleselixir.com/):"""

all_games = pd.read_csv("https://oracleselixir-downloadable-match-data.s3-us-west-2.amazonaws.com/2020_LoL_esports_match_data_from_OraclesElixir_20201220.csv")
all_games.head(5)

"""Primero que nada vamos a eliminar todos los campos irrelevantes para el análisis y los que nos den información demasiado especifica de partida:"""

games = all_games.drop(columns=["gameid","datacompleteness","url","playerid","game","playoffs","ban1","ban2","ban3","ban4","ban5","split"])
games = games.drop(columns=["doublekills","triplekills","quadrakills","pentakills","firstblood","firstbloodkill","firstbloodassist","firstbloodvictim","team kpm",
                                    "ckpm","firstdragon","opp_dragons","elementaldrakes","opp_elementaldrakes","infernals","mountains","clouds","oceans","dragons (type unknown)",
                                    "elders","opp_elders","firstherald","opp_heralds","firstbaron","opp_barons","firsttower","opp_towers","firstmidtower","firsttothreetowers",
                                    "opp_inhibitors","controlwardsbought","earnedgold","goldspent","gspd","minionkills","monsterkills","monsterkillsownjungle",
                                    "monsterkillsenemyjungle","goldat10","xpat10","csat10","opp_goldat10","opp_xpat10","opp_csat10","golddiffat10","xpdiffat10","csdiffat10","goldat15",
                                    "xpat15","csat15","opp_goldat15","opp_xpat15","opp_csat15","golddiffat15","xpdiffat15","csdiffat15","damagemitigatedperminute",
                                    "wcpm","damagetakenperminute","dragons","heralds","barons","towers","inhibitors"])
games.head(5)

"""Despues de limpiar los campos con los que no vamos a trabajar, eliminaremos las partidas a las que les falte información y las que no formen parte de las ligas que vamos a analizar (ademas de añadir las partidas de las ligas no regulares a sus respectivas ligas):"""

games = games.dropna()
# filtrado
main_leagues = ['KeSPA', 'LPL', 'LEC', 'LCS', 'LCK']
games = games[ games['league'].isin(main_leagues)]
# agrupamos datos de ligas
games.loc[games['league'] == "KeSPA", 'league'] = "LCK"

games.head(5)

"""Por último, como el tamaño de los datos es demasiado grande, vamos a hacer un muestreo estratificado aleatorio para obtener un total de 1000 muestras con las que trabajar."""

games_sample = games.groupby('league', group_keys=False).apply(lambda x: x.sample(int(np.rint(1000*len(x)/len(games))))).sample(frac=1).reset_index(drop=True)
games_sample.shape

print("LEC games -> " , games_sample[(games_sample['league'] == "LEC")]['league'].count())
print("LCK games -> " , games_sample[(games_sample['league'] == "LCK")]['league'].count())
print("LPL games -> " , games_sample[(games_sample['league'] == "LPL")]['league'].count())
print("LCS games -> " , games_sample[(games_sample['league'] == "LCS")]['league'].count())

games_sample.head(5)

"""# Descripción de los datos

Para este proyecto hemos elegido un conjunto de datos del famoso videojuego del género multijugador de arena de batalla en línea League of Legends.
El juego se basa en que dos equipos de 5 jugadores se enfrentan en una batalla la cual termina cuando uno de los dos equipos destruye todas las defensas enemigas.

Nosotros vamos a centrarnos en las diferencias que podemos encontrar entre las diferentes grandes ligas a nivel internacional y las diferencias en estilo de juego de estos en cada uno de las versiones del juego (el juego recibe cambios cada cierto tiempo en lo que se conocen como parches/versiones).

Para realizar todo esto hemos seleccionado un conjunto de datos con todas las partidas jugadas en 2020 en las grandes ligas China (LPL), Coreana (LCK), Europea (LEC) y Norteamericana (LCS). Estos datos fueron extraidos de (https://oracleselixir.com/) y preparados para su uso.

Las variables aleatorias utilizadas para esto son:


*   league (liga) [categórica]
*   position (rol en partida) [categórica]
*   gamelength (duración de la partida) [continua]
*   vspm (puntuación de visión por minuto, calculada por la suma de los minutos que ha otorgado a su equipo o denegado visión al enemigo entre todos los minutos de partida) [continua]
*   dpm (daño inflingido por minuto) [continua]
*   total cs (subditos eliminados en partida) [discreta]
"""

games_sample

"""# Análisis descriptivo

Para el análisis descriptivo vamos a comenzar observando las variables continuas y discretas que hemos elegido:

## Duración de partidas (variable continua)

Comenzamos con la duración de las partidas, podemos observar que la media de duración está en 1972.5 segundos (unos 33 minutos), bastante coherente ya que gran parte de la temporada ha tenido un estilo de juego muy lento ya que la mayoria de personajes necesitaba de muchos recursos para poder llegar a tener un buen impacto en partida.

Encontramos que la mayoria de las muestras se encuentran entre 1727.5 (28.8 minutos) y 2157 (35.95 minutos) (primer y tercer percentil), cosa que tiene sentido como hemos comentado anteriormente.

Si nos fijamos en la asimetría, el coeficiente de asimetría estandarizado y la cercania de la mediana con la media que se trata de una distribución bastante simétrica.

Por otro lado, si observamos la curtosis estandarizada, vemos que se trata de una distribución un tanto leptocúrtica y que segun la curtosis se desvia un poco de la normal.

Mirando el histograma, parece que la distribución que se asemeja a una normal, cosa que con la asimetría y curtosis nos dice tambien.
"""

# gamelength
print("Asimetría -> ",games_sample['gamelength'].skew())
print("Curtosis -> ",games_sample['gamelength'].kurt())
print(games_sample['gamelength'].describe())
sns.histplot(data=games_sample, x="gamelength", bins=30)

"""Observando el diagrama Box-Whisker, parece ser que es una distribución simétrica y con una cierta normalidad, de todas formas se observan ciertos valores un tanto alejados.

De todas formas, estos datos que vemos simplemente deben tratarse de partidas con composiciones que escalaban muy lento con el tiempo, como por ejemplo pasa con el mayor valor que encontramos, partida de Inmortals que está catalogada como la partida más larga de la LCS Norteamericana en 2020 con 61.20 minutos, esto nos dice que todos estos valores más perifericos simplemente se deben partidas un tanto peculiares, de hecho en ningun caso estas muestras superan el 5% del total de muestras (menos en la LCS norteamericana, la cual llega a contar con un 10% de partidas más largas, una de tantas razones por las que se considera esta liga como inferior respecto a las demas).
"""

sns.boxplot( x=games_sample.gamelength )

print("muestras totales: ",games_sample['gamelength'].count())
for x in list(pd.unique(games_sample['league'])):
  print(x, ": ", games_sample[(games_sample['league'] == x)]['gamelength'].count())

print("")
print("muestras < 1200: ", games_sample[games_sample['gamelength'] < 1200]['gamelength'].count())
# print(pd.unique(games_sample[games_sample['gamelength'] < 1200]['league']))
for x in list(pd.unique(games_sample[games_sample['gamelength'] < 1200]['league'])):
  print(x, ": ", games_sample[(games_sample['league'] == x) & (games_sample['gamelength'] < 1200)]['gamelength'].count())

print("")
print("muestras > 2700: ",games_sample[games_sample['gamelength'] > 2700]['gamelength'].count())
# print(pd.unique(games_sample[games_sample['gamelength'] > 2700]['league']))
for x in list(pd.unique(games_sample[games_sample['gamelength'] > 2700]['league'])):
  print(x, ": ", games_sample[(games_sample['league'] == x) & (games_sample['gamelength'] > 2700)]['gamelength'].count())

print("")
print("muestras (>1200) & (<2700): ",games_sample[(games_sample['gamelength'] < 2700) & (games_sample['gamelength'] > 1200)]['gamelength'].count())
for x in list(pd.unique(games_sample[(games_sample['gamelength'] < 2700) & (games_sample['gamelength'] > 1200)]['league'])):
  print(x, ": ", games_sample[(games_sample['league'] == x) & (games_sample['gamelength'] < 2700) & (games_sample['gamelength'] > 1200)]['gamelength'].count())

games_sample[(games_sample['gamelength'] > 3400)]

"""## Puntuación de visión por minuto (variable continua)

Al observar la puntuación de visión por minuto podemos ver que la mayor parte de los valores se encuentran en [0.97 - 1.94], la media se encuentra en 1.5, bastante coherente ya que en partidas de este nivel todas las jugadas se basan en la visión sobre objetivos lo que hace que todos tengan un impacto de visión en partida.

Si analizamos el coeficiente de asimetría estandarizado, podemos ver que posiblemente se trate de una distribución similar a una normal segun este factor. También puede verse que aunque se trate de una distribución simétrica, con el histograma parece tener una ligera asimetría positiva.

En lo que a la Curtosis estandarizada respecta, parece que tambien se parece a una distribución normal, cercana a una mesocúrtica aunque tendiendo hacia una distribución platicúrtica.
"""

# vspm
print("Asimetría -> ",games_sample['vspm'].skew())
print("Curtosis -> ",games_sample['vspm'].kurt())
print(games_sample['vspm'].describe())
sns.histplot(data=games_sample, x="vspm", bins=30)

"""Con el diagrama de Box-Whisker parece que tiene una ligera asimetría positiva como comentabamos, ademas de que se observan unos outliers en valores superiores a 3.3 aproximadamente.
Analizando estos outliers podemos ver que se trata de supports principalmente, este rol comple una función principalmente de utilidad en partida, lo que es bastante coherente para algo como la visión.
De todas maneras, estos outliers son simplemente algo puntual de partidas concretas, ya que estos no suman ni el 5% de las muestras totales.
"""

sns.boxplot( x=games_sample.vspm )

print("muestras totales: ",games_sample['vspm'].count())
for x in list(pd.unique(games_sample['league'])):
  print(x, ": ", games_sample[(games_sample['league'] == x)]['vspm'].count())


print("")
print("muestras > 3.3: ",games_sample[games_sample['vspm'] > 3.3]['vspm'].count())
# print(pd.unique(games_sample[games_sample['vspm'] > 3.3]['league']))
for x in list(pd.unique(games_sample[games_sample['vspm'] > 3.3]['league'])):
  print(x, ": ", games_sample[(games_sample['league'] == x) & (games_sample['vspm'] > 3.3)]['vspm'].count())

print("")
print("muestras < 3.3: ",games_sample[(games_sample['vspm'] < 3.3)]['vspm'].count())
for x in list(pd.unique(games_sample[(games_sample['vspm'] < 3.3)]['league'])):
  print(x, ": ", games_sample[(games_sample['league'] == x) & (games_sample['vspm'] < 3.3)]['vspm'].count())

games_sample[(games_sample['vspm'] > 3.3)]

"""## Daño infligido por minuto (variable continua)

Para empezar observamos la media y el rango en el que se encuentran la mayoria de las muestras [197.6 - 501.87] con media 366.5, bastante coherente dado que la media de daño base ronda los 60 de daño de ataque, y que hasta el minuto 10/15 solo ocurren peleas esporadicas y los jugadores se centran en dar el último golpe a los subditos para obtener oro pero no están peleando. Teniendo un autoataque cada 5 segundos de media serian unos (371.4/(60/5)) = 30.95, bastante coherente pensando que para asegurar el último golpe a un subdito deben asestar un autoataque que el daño inflingido excede por bastante su daño de ataque.

En cuanto a el coeficiente de asimetria estandarizado y la curtosis estandarizada, parece que se trata de una distribución similar a una normal, aunque con una muy ligera asimetría positiva y un poco leptocúrtica.
"""

# dpm
print("Asimetría -> ",games_sample['dpm'].skew())
print("Curtosis -> ",games_sample['dpm'].kurt())
print(games_sample['dpm'].describe())
sns.histplot(data=games_sample, x="dpm", bins=30)

"""En el diagrama de Box-Whisker podemos ver lo que parece ser una distribución similar a una normal aunque tiene una leve asimetría positiva, ademas se observan unos outliers con valores superiores a 900.

Estos valores parecen ser unos pocos correspondientes a las lineas de top, mid y bot (refiriendose este último a la posición de ad carry), tiene sentido que sean estas lineas debido a que son los roles donde se focaliza tener campeones con gran daño, suelen ser las principales fuentes de ataque del equipo. Ademas, si nos fijamos en que campeones son, vemos que son personajes que al inicio de la partida son más debiles, pero a partir del minuto 10 de partida empiezan a tener una gran cantidad de daño, más factores para entender porque encontramos estos outliers.
"""

sns.boxplot( x=games_sample.dpm )

print("muestras totales: ",games_sample['dpm'].count())
for x in list(pd.unique(games_sample['league'])):
  print(x, ": ", games_sample[(games_sample['league'] == x)]['dpm'].count())

print("")
print("muestras > 900: ",games_sample[games_sample['dpm'] > 900]['dpm'].count())
# print(pd.unique(games_sample[games_sample['dpm'] > 900]['league']))
for x in list(pd.unique(games_sample[games_sample['dpm'] > 900]['league'])):
  print(x, ": ", games_sample[(games_sample['league'] == x) & (games_sample['dpm'] > 900)]['dpm'].count())
print("")

dpm_outliers = games_sample[(games_sample['dpm'] > 900)]
dpm_outliers_roles = list(pd.unique(dpm_outliers['position']))
for role in dpm_outliers_roles:
  print(role, " games (>900 dpm): ", (dpm_outliers[dpm_outliers['position'] == role])['position'].count())
  print(list(pd.unique(dpm_outliers[dpm_outliers['position'] == role]['champion'])),"\n")

print("")
print("muestras < 900: ",games_sample[(games_sample['dpm'] < 900)]['dpm'].count())
for x in list(pd.unique(games_sample[(games_sample['dpm'] < 900)]['league'])):
  print(x, ": ", games_sample[(games_sample['league'] == x) & (games_sample['dpm'] < 900)]['dpm'].count())
print("")

"""## Total de Subditos eliminados (variable discreta)

"""

# total cs
print("Asimetría -> ",games_sample['total cs'].skew())
print("Curtosis -> ",games_sample['total cs'].kurt())
print(games_sample['total cs'].describe())
sns.histplot(data=games_sample, x="total cs", bins=30)

sns.boxplot( x=games_sample["total cs"] )

print("muestras totales: ",games_sample['total cs'].count())
for x in list(pd.unique(games_sample['league'])):
  print(x, ": ", games_sample[(games_sample['league'] == x)]['total cs'].count())

print("")
print("muestras > 500: ",games_sample[games_sample['total cs'] > 500]['total cs'].count())
# print(pd.unique(games_sample[games_sample['total cs'] > 500]['league']))
for x in list(pd.unique(games_sample[games_sample['total cs'] > 500]['league'])):
  print(x, ": ", games_sample[(games_sample['league'] == x) & (games_sample['total cs'] > 500)]['total cs'].count())

print("")
print("muestras < 500: ",games_sample[(games_sample['total cs'] < 500)]['total cs'].count())
for x in list(pd.unique(games_sample[(games_sample['total cs'] < 500)]['league'])):
  print(x, ": ", games_sample[(games_sample['league'] == x) & (games_sample['total cs'] < 500)]['total cs'].count())

games_sample[(games_sample['total cs'] > 500)]

"""## Variables categóricas

Para las variables categóricas hemos elegido primero la de ligas, en esta observamos que la LPL (liga china) es la que más partidas tiene, seguida de la LCK (liga coreana). Este repunte de la LPL se debe a que esta liga cuenta con 17 equipos que juegan todas las semanas con un formato de 6 partidas por jornada, la LCK cuenta con el mismo tipo de formato de 6 partidas semanales, pero solo cuenta con 10 equipos (razón por la que tiene menos partidas). Por último la LEC (liga Europea) y la LCS (liga Norteamericana) cuentan con 10 equipos y 2 partidos semanales.

Por lo cual, debido a el numero de partidos que tiene cada liga semanalmente y el numero de equipos, es coherente el balance de muestras por liga.
"""

count_data = games_sample.league.value_counts()
plt.figure(figsize=(10, 10))
fig = plt.pie( count_data, labels = count_data.index, autopct="%.2f", textprops={"fontsize":14} )
print(games_sample.league.value_counts(normalize=True))

"""La segunda variable categórica que vamos a analizar es la posición en partida, vemos que están bastante repartidas las muestras, y esto es algo normal dado que en una partida juegan 2 personas de cada posición. La ligera diferencia que se encuentra más notoriamente entre bot y las demas posiciones se debe a que hay datos que se han perdido al limpiar y reducir los datos."""

count_data = games_sample.position.value_counts()
plt.figure(figsize=(10, 10))
fig = plt.pie( count_data, labels = count_data.index, autopct="%.2f", textprops={"fontsize":14} )
print(games_sample.position.value_counts(normalize=True))

"""# Normalidad de los datos e inferencia

## Duración de partidas

Vamos a ver si la variable de duración de cada partida sigue una distribución normal o parecida a una normal:
"""

sample = games_sample.sample(n=100)
print( stats.skewtest( sample['gamelength'] ) , "\n -> H1:  El coeficiente de asimetría (CA) de la población sigue una asimetría diferente de una población normal (CA != 0)\n" )
print( stats.kurtosistest( sample['gamelength'] ) , "\n -> H0:  El coeficiente de curtosis (CC) de la población sigue una curtosis propia de una población normal (CC=0)\n")
print( stats.kstest( (sample['gamelength']-sample['gamelength'].mean())/sample['gamelength'].std(), 'norm'  ) , "\n -> H0:  La función de distribución acumulada se comporta como la función de distribución acumulada de una distribución normal.\n")
print( stats.shapiro( sample['gamelength'] ) , "\n -> H1: La variable aleatoria estudiada no procede de una población normal\n")

sns.histplot(x=sample.gamelength, stat="density", common_norm=False, bins=30)

# papel probabilistico
plot = sm.ProbPlot( sample['gamelength'], dist="norm" )
plot.probplot(line="r")

"""Aunque no parece que se trate de una distribución normal, se aproxima a la normal, mientras que los test de normalidad nos dicen que no se trata de una normal, observando el histograma y el papel probabilistico, deducimos que se trata de una distribución aproximadamente normal aunque tiene una asimetria por la derecha.

## Puntuación de visión por minuto

Ahora vamos a analizar la normalidad en la distribución de vspm (puntuación de visión por minuto):
"""

print( stats.skewtest( sample['vspm'] ) , "\n -> H1:  El coeficiente de asimetría (CA) de la población sigue una asimetría diferente de una población normal (CA != 0)\n" )
print( stats.kurtosistest( sample['vspm'] ) , "\n -> H1:  El coeficiente de curtosis (CC) de la población tiene una curtosis diferente a la de una población normal (CC!=0)\n")
print( stats.kstest( (sample['vspm']-sample['vspm'].mean())/sample['vspm'].std(), 'norm'  ) , "\n -> H1: La función de distribución acumulada difiere de la función de distribución acumulada de una distribución normal.\n")
print( stats.shapiro( sample['vspm'] ) , "\n -> H1: La variable aleatoria estudiada no procede de una población normal\n")

sns.histplot(x=sample.vspm, stat="density", common_norm=False, bins=30)

# papel probabilistico
plot = sm.ProbPlot( sample['vspm'], dist="norm" )
plot.probplot(line="r")

"""Después de observar los test de normalidad y el papel probabilistico, concluimos que la distribución no sigue una normal, por lo que vamos a tratar de aproximarla haciendo uso de una transformación:"""

log_sample = sample.apply(lambda x: np.log10(x) if np.issubdtype(x.dtype, np.number) else x)
square_sample = sample.apply(lambda x: np.square(x) if np.issubdtype(x.dtype, np.number) else x)
reciproco_sample = sample.apply(lambda x: (1/x) if np.issubdtype(x.dtype, np.number) else x)

fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(15,10))
plt.subplots_adjust(wspace=0.5)
fig.suptitle("Transformaciones para Visión por minuto")

sns.histplot(x=log_sample.vspm, stat="density", common_norm=False, bins=30,ax=axes[0,0])
axes[0,0].set_title("Transformación Logaritmica")
sns.histplot(x=square_sample.vspm, stat="density", common_norm=False, bins=30,ax=axes[0,1])
axes[0,1].set_title("Transformación Raiz cuadrada")
sns.histplot(x=reciproco_sample.vspm, stat="density", common_norm=False, bins=30,ax=axes[1,0])
axes[1,0].set_title("Transformación Reciproca")

"""Después de hacer las transformaciones y analizar los histogramas, vemos que la transformación logaritmica es la única que parece acercarse más a una normal que la distribución original, vamos a ver si realmente se aproxima a una normal por el papel probabilístico:"""

# papel probabilistico
plot = sm.ProbPlot( log_sample['vspm'], dist="norm" )
plot.probplot(line="r")

"""Podemos concluir que despues de realizar una transformación logaritmica, nuestra distribución si que se aproxima bastante a una normal.

## Daño inflingido por minuto

Por último vamos a analizar la normalidad de la distribución de daño inflingido por minuto (dpm):
"""

print( stats.skewtest( sample['dpm'] ) , "\n -> H1:  El coeficiente de asimetría (CA) de la población sigue una asimetría diferente de una población normal (CA != 0)\n" )
print( stats.kurtosistest( sample['dpm'] ) , "\n -> H0:  El coeficiente de curtosis (CC) de la población sigue una curtosis propia de una población normal (CC=0)\n")
print( stats.kstest( (sample['dpm']-sample['dpm'].mean())/sample['dpm'].std(), 'norm'  ) , "\n -> H0:  La función de distribución acumulada se comporta como la función de distribución acumulada de una distribución normal.\n")
print( stats.shapiro( sample['dpm'] ) , "\n -> H1: La variable aleatoria estudiada no procede de una población normal\n")

sns.histplot(x=sample.dpm, stat="density", common_norm=False, bins=30)

# papel probabilistico
plot = sm.ProbPlot( sample['dpm'], dist="norm" )
plot.probplot(line="r")

"""Después de observar el histograma, el papel probabilístico y los test de normalidad, asumiendo el hecho que hemos visto en el analisis descriptivo, se trata de una distribución aproximadamente normal.

## Construir Intervalos de Confianza de la media (Daño inflingido por minuto)
"""

x = sample.dpm.mean()
n = sample.shape[0]
s = sample.dpm.std()

# intervalo del 95%
lower_bound_95 = 0
upper_bound_95 = 0
t_95 = stats.t.ppf(0.975, n-1 )
lower_bound_95 = x - t_95 * s/math.sqrt(n)
upper_bound_95 = x + t_95 * s/math.sqrt(n)
print("Intervalo del 95%: ( " , lower_bound_95, upper_bound_95 , ")")

# intervalo del 90%
lower_bound_90 = 0
upper_bound_90 = 0
t_90 = stats.t.ppf(0.95, n-1 )
lower_bound_90 = x - t_90 * s/math.sqrt(n)
upper_bound_90 = x + t_90 * s/math.sqrt(n)
print("Intervalo del 90%: ( " , lower_bound_90, upper_bound_90 , ")")

# intervalo del 70%
lower_bound_70 = 0
upper_bound_70 = 0
t_70 = stats.t.ppf(0.85, n-1 )
lower_bound_70 = x - t_70 * s/math.sqrt(n)
upper_bound_70 = x + t_70 * s/math.sqrt(n)
print("Intervalo del 70%: ( " , lower_bound_70, upper_bound_70 , ")")
print("")

data_dict = {}
data_dict['category'] = ['95%','90%','70%']
data_dict['lower'] = [lower_bound_95,lower_bound_90,lower_bound_70]
data_dict['upper'] = [upper_bound_95,upper_bound_90,upper_bound_70]
intervalos = pd.DataFrame(data_dict)

plots = {}
count = 0
for lower,upper,y in zip(intervalos['lower'],intervalos['upper'],range(len(intervalos))):
    plots[count] = plt.plot((lower,upper),(y,y),'ro-',color='blue')
    count += 1
plt.yticks(range(len(intervalos)),list(intervalos['category']))

"""Después de analizar el intervalo de confianza para un nivel de significancia de 95%, 90% y 70%. Podemos ver como en un nivel del 70%, como tiene que encontrarse la media el 70% de las veces en este intervalo, se trata de un intervalo más estrecho que el de 90% o 95%.

## Construir Intervalos de Confianza de la varianza (Daño inflingido por minuto)

Vamos a realizar un analisis de los intervalos de confianza de la varianza haciendo uso de chi-cuadrado:
"""

x = sample.dpm.mean()
n = sample.shape[0]
s = sample.dpm.std()

# intervalo del 95%
lower_bound_95 = 0
upper_bound_95 = 0
chi1_95 = stats.chi2.ppf(1-0.05/2, n - 1 )
chi2_95 = stats.chi2.ppf(0.05/2, n - 1 )
lower_bound_95 = (n-1)*s/chi1_95
upper_bound_95 = (n-1)*s/chi2_95
print("Intervalo de varianza del 95%: ( " , lower_bound_95, upper_bound_95 , ")")

# intervalo del 90%
lower_bound_90 = 0
upper_bound_90 = 0
chi1_90 = stats.chi2.ppf(1-0.5/2, n - 1 )
chi2_90 = stats.chi2.ppf(0.5/2, n - 1 )
lower_bound_90 = (n-1)*s/chi1_90
upper_bound_90 = (n-1)*s/chi2_90
print("Intervalo de varianza del 90%: ( " , lower_bound_90, upper_bound_90 , ")")

data_dict = {}
data_dict['category'] = ['95%','90%']
data_dict['lower'] = [lower_bound_95,lower_bound_90]
data_dict['upper'] = [upper_bound_95,upper_bound_90]
intervalos = pd.DataFrame(data_dict)

plots = {}
count = 0
for lower,upper,y in zip(intervalos['lower'],intervalos['upper'],range(len(intervalos))):
    plots[count] = plt.plot((lower,upper),(y,y),'ro-',color='black')
    count += 1
plt.yticks(range(len(intervalos)),list(intervalos['category']))

"""En el intervalo de confianza de varianza, podemos ver como el intervalo del 90%, dado que tiene menos precisión que el del 95%, se trata de un intervalo más pequeño.

# ANOVA

## ANOVA con un factor

Ahora vamos a realizar un analisis de varianza para ver las diferencias en la duración de las partidas entre cada una de las ligas:
 - Primero vamos a crear columnas binarias para cada una de las ligas y vamos a ver como se distribuyen estas (
"""

ligas = list(pd.unique(games_sample['league']))
for liga in ligas:
  r = games_sample.league.apply(lambda x: liga in x)
  games_sample=games_sample.assign( **{liga: r} )
games_sample.head(5)

fig_leagues, axes_leagues = plt.subplots(nrows=2,ncols=2,figsize=(15,10))
plt.subplots_adjust(wspace=0.5)
fig_leagues.suptitle("Histograma por liga")

lck_gamelength = games_sample[games_sample['LCK']].gamelength
lpl_gamelength = games_sample[games_sample['LPL']].gamelength
lec_gamelength = games_sample[games_sample['LEC']].gamelength
lcs_gamelength = games_sample[games_sample['LCS']].gamelength

sns.histplot(x=lck_gamelength, ax=axes_leagues[0,0])
axes_leagues[0,0].set_title("LCK")
sns.histplot(x=lpl_gamelength, bins=30,ax=axes_leagues[0,1])
axes_leagues[0,1].set_title("LPL")
sns.histplot(x=lec_gamelength, bins=30,ax=axes_leagues[1,0])
axes_leagues[1,0].set_title("LEC")
sns.histplot(x=lcs_gamelength, bins=30,ax=axes_leagues[1,1])
axes_leagues[1,1].set_title("LCS")

fig_leagues, axes_leagues = plt.subplots(nrows=2,ncols=2,figsize=(15,10))
plt.subplots_adjust(wspace=0.5,hspace=0.3)
fig_leagues.suptitle("Papel probabilístico por liga")

lck_gamelength = games_sample[games_sample['LCK']].gamelength
lpl_gamelength = games_sample[games_sample['LPL']].gamelength
lec_gamelength = games_sample[games_sample['LEC']].gamelength
lcs_gamelength = games_sample[games_sample['LCS']].gamelength

sm.ProbPlot(lck_gamelength,dist="norm").probplot(line="r",ax=axes_leagues[0,0])
axes_leagues[0,0].set_title("LCK")
sm.ProbPlot(lpl_gamelength,dist="norm").probplot(line="r",ax=axes_leagues[0,1])
axes_leagues[0,1].set_title("LPL")
sm.ProbPlot(lec_gamelength,dist="norm").probplot(line="r",ax=axes_leagues[1,0])
axes_leagues[1,0].set_title("LEC")
sm.ProbPlot(lcs_gamelength,dist="norm").probplot(line="r",ax=axes_leagues[1,1])
axes_leagues[1,1].set_title("LCS")

"""Como podemos ver que las distribuciones para cada una de las variables siguen lo que parece ser una normal.

Vamos a comprobar que se cumple la hipótesis de homocedasticidad y posteriormente haremos una comparación de las medias entre cada una de las ligas haciendo uso de ANOVA (ANalysis Of VAriance).

Usaremos la prueba de **Levene** ya que es una prueba relativamente robusta ante desviaciones de la normal. Este contraste de hipotesis establece las siguientes hipótesis:

$H_0: \sigma_1 = \sigma_2 = \dots = \sigma_k$

$H_1: \exists i,j , \; \sigma_i \neq \sigma_k$
"""

lck_gamelength = games_sample[games_sample['LCK']].gamelength
lpl_gamelength = games_sample[games_sample['LPL']].gamelength
lec_gamelength = games_sample[games_sample['LEC']].gamelength
lcs_gamelength = games_sample[games_sample['LCS']].gamelength

stats.levene(lck_gamelength,lpl_gamelength,lec_gamelength,lcs_gamelength)
print("[", lck_gamelength.std(), " - ", lpl_gamelength.std(), " - ", lec_gamelength.std(), " - ", lcs_gamelength.std(),"]")
print("La prueba de Levene nos dice que parece que las varianzas son distintas")

"""Aunque la normalidad de las poblaciones no es demasiado precisa y que la prueba de levene nos dice que parece que al menos hay 2 varianzas distintas (aunque vemos que las diferencias son leves si nos fijamos que 3 de las varianzas son bastante similares), pero ANOVA es relativamente robusto ante estas pequeñas variaciones por lo que no hay problema.

Ahora vamos a realizar un **ANOVA** con la liga como factor para encontrar las diferencias significativas en la media de las diferentes poblaciones, las hipotesis que establece ANOVA son:

$H_0: \mu_1 = \mu_2 = \dots = \mu_k$ (provienen de la misma población)

$H_1: \exists i,j \;\; \mu_i\neq \mu_j$ (provienen de poblaciones diferentes)


"""

lm = ols( 'gamelength ~ C(league)', data=games_sample ).fit(cov_type="HC3")
anova_table = sm.stats.anova_lm(lm, typ=1,  robust="HC3")
anova_table

"""El análisis de ANOVA nos devuelve un p-value que indica que para el nivel de significancia de 0.05 se acepta la hipotesis alternativa, por lo que consideramos que las medias son diferentes.

Después de realizar el ANOVA y que nos diga que al menos una media poblacional es diferente, vamos a analizar cual es, para esto usaremos los intervalos de Tukey:
"""

tukey = sm.stats.multicomp.pairwise_tukeyhsd(games_sample.gamelength, games_sample.league, alpha=0.01).plot_simultaneous(comparison_name="LPL")

print("LCK -> ",len(lck_gamelength))
print("LPL -> ", len(lpl_gamelength))
print("LEC -> ", len(lec_gamelength))
print("LCS -> ", len(lcs_gamelength))

# levene para comprobar el ANOVA
lck_gamelength = games_sample[games_sample['LCK']].gamelength
lpl_gamelength = games_sample[games_sample['LPL']].gamelength
lec_gamelength = games_sample[games_sample['LEC']].gamelength
lcs_gamelength = games_sample[games_sample['LCS']].gamelength

stats.levene(lck_gamelength,lpl_gamelength,lec_gamelength,lcs_gamelength)
print("[", lck_gamelength.std(), " - ", lpl_gamelength.std(), " - ", lec_gamelength.std(), " - ", lcs_gamelength.std(),"]")
print("La prueba de Levene nos dice que parece que las varianzas son distintas")

"""Según la prueba de levene tambien dice que se trata de varianzas diferentes, pero si nos fijamos en el tamaño de cada muestra podemos ver que la LPL (liga china) tiene un gran número de muestras en comparación con las demás poblaciones, si nos fijamos en los intervalos de tukey y en las diferencias que hay al analizar las varianzas, podemos concluir que el alto número de muestras con las que cuenta la LPL cause que se generen unas diferencias tan exageradas.

## ANOVA con múltiples factores

Para el ANOVA con múltiples factores vamos a analizar el efecto de la posición de los jugadores (top, jungla, mid, adc y support) y de la liga en la que se juega (LCK, LPL, LEC, LCS) en el daño inflingido por minuto.
"""

lm = ols( 'dpm ~ C(league) + C(position) + C(league)*C(position)', data=games_sample ).fit(cov_type="HC3")
anova_table = sm.stats.anova_lm(lm, typ=2, robust="HC3")
anova_table

"""Tras analizar los resultados, podemos ver que si existe una significancia en la interacción si consideramos un nivel de significancia de $\alpha=0.05$, ademas podemos ver que la variable de ligas si es significativa igual de las posiciones.

Como hemos obtenido que las interacciones SI son significativas, vamos a analizar los gráficos de interacción:
"""

sm.graphics.interaction_plot(x=games_sample.position, trace=games_sample.league, response=games_sample.dpm)

"""Con los gráficos de interacciones podemos observar ciertas cosas:
  - La interacción como se puede ver en las lineas es de una fuerte significancia ya que las lineas se cruzan.
  - Existe una gran similitud entre la LCK y la LPL, se denota una fuerte relación entre estas en todos los roles, esto tiene sentido ya que ambas son ligas asiaticas que destacan por un estilo de juego similar.
  - Los roles de bot, mid y top son notablemente más altos que los demas, esto tiene sentido ya que son los roles que cuentan con los objetos más poderosos de la partida.
"""

residuals = lm.resid
plt.figure( figsize=(10, 5) )
sns.pointplot( data=games_sample.assign(residuals=residuals, factor=games_sample.apply(lambda row: str(row.position)+"-"+str(row.league), axis=1)), x="factor", y="residuals", ci="sd", join=False )
plt.xticks(rotation=45)

plot = sm.ProbPlot(residuals, dist="norm")
plt.figure()
plot.probplot(line="r")
plt.figure()
sns.histplot(residuals)
plt.figure()
axes = sns.kdeplot( (residuals - residuals.mean() )/(residuals.std()), common_norm=False  )
sns.lineplot( y=[ stats.norm.pdf(x) for x in np.arange(-3, 3, 0.01) ], x= [x for x in np.arange(-3, 3, 0.01)] )

"""Después de analizar los residuos podemos ver que estos tienen un comportamiendo aproximadamente normal y las subpoblaciones son lo suficientemente grandes como para fiarnos, por lo que podemos asegurarnos del ANOVA realizado.

# Regresión lineal

## Regresión Lineal Simple

Vamos a realizar una regresión lineal con variable predictora X=duración de partidas y una variable dependiente (a predecir) Y=daño por minuto.
"""

# X = games_sample[ (games_sample['position'] == 'top') & (games_sample['league'] == 'LCK')].dpm
# Y = games_sample[ (games_sample['position'] == 'top') & (games_sample['league'] == 'LCK')].gamelength
# sample_regression = games_sample.sample(n=100)
X = games_sample.gamelength
Y = games_sample.dpm
sns.regplot( x=X, y=Y )

simple_r = stats.pearsonr(X,Y)
print("Coeficiente de Correlación: ", simple_r[0])
print("p-value: ", simple_r[1])

"""Obtenemos un coeficiente de correlación de 0.074 con un p-value de 0.018.

Observando el gráfico de dispersión y el coeficiente de correlación, estos muestran que existe relación lineal, con una intensidad de (r = 0.074) y significancia (p-value = 0.018). Vamos a generar un modelo de regresión lineal simple, aunque observando el gráfico de dispersión no parece dar buenas señales.
"""

lm = ols('Y ~ X', data=pd.DataFrame().assign(X=X, Y=Y) )
fitted_model = lm.fit()
print(fitted_model.summary())

"""Depués de realizar el análisis de regresión, podemos ver, como anteriormente que existe relación lineal entre las variables analizadas, aunque esto tambien puede deberse a que estamos trabajando con 1000 muestras y cabe la posibilidad de que esté introduciendo ruido en el analisis. 

Tambien se observa una $r^2$ de 0.005, lo que nos dice que el 0.5% de los datos de daño por minuto son explicados por la duración de las partidas, esto nos hace entender que se trata de un modelo con una capacidad predictiva casi nula.

La ecuación resultante es:
$y = 106.18 + 0.11x $

Ahora vamos a analizar los residuos para contrastar las hipotesis del analisis de regresión:
"""

# normalidad de los residuos
plot = sm.ProbPlot(data=fitted_model.resid).probplot(line="r")

# analisis del error (comprobar homocedasticidad)
f=sns.scatterplot(x=fitted_model.predict(), y=fitted_model.resid)
f.set_xlabel("Fitted value")
f.set_ylabel("Residual")

f=sns.scatterplot(x=[i for i in range(len(X))], y=fitted_model.resid)

"""Después de observar los residuos, podemos ver que:
 - Parece que los residuos no siguen bien una normal ya que parece que se están acercando a una ecuación cuadratica o a otra forma diferente de una normal.
 - Observamos tambien que al observar la predicción realizada para los datos de la muestra con respecto al residuo (error cometido en la prediccón) se puede ver que los residuos tienen un comportamiento heterocedastico.
 - Parece que tambien hay cierta independencia al observar los errores.

Después de observar los residuos, podemos contrastar, como esperabamos, que el modelo nos ofrece unas predicciones bastante pobres.

## Regresión Múltiple

Ahora vamos a realizar un analisis de regresión similar pero añadiendo otra variable predictora de manera que quedará:
 - X1 = duración de partidas (predictora 1)
 - X2 = visión por minuto (predictora 2)
 - Y = daño por minuto (dependiente)
"""

X1 = games_sample.gamelength
X2 = games_sample.vspm
Y = games_sample.dpm

# analisis de regresión
lm = ols('Y ~ X1 + X2', data=pd.DataFrame().assign(X1=X1, X2=X2, Y=Y) )
fitted_model = lm.fit()
print(fitted_model.summary())

"""Observando el analisis de regresión múltiple, podemos ver que esta vez si que parece existir una relación lineal significante, con un nivel de significancia aceptable (p-value = $ 1.57 * 10^-62 $)

En cuanto a las variables independientes introducidas, parece que ambas demuestran significancia en la relación lineal a nivel poblacional.

Observando la $r^2$ de ambos modelos:
 - regresión simple = 0.005
 - regresión múltiple = 0.247
Podemos concluir que el nuevo modelo tiene más predición que el anterior, ya que este explica un 24.7% del daño por minuto a partir de la duración de partidas junto con la visión por minuto.




`author: Carlos María Canut Domínguez`
"""