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
from statsmodels.formula.api import ols


# -----------------------------------------------------------------------------
# ---------------------------- EXTRACCIÓN DE DATOS ----------------------------
# -----------------------------------------------------------------------------

string_csv = requests.get("https://gitlab.com/drvicsana/ciencia-datos/-/raw/master/datasets/movies/movies.csv").content
string_stream = io.StringIO(string_csv.decode("utf-8"))
movies_df = pd.read_csv(string_stream)


# limpiar los datos vacios
movies_df = movies_df.dropna()
# quitar la columna de indices que ya no sirve
movies_df=movies_df.drop(labels=["Unnamed: 0"], axis=1)


# -----------------------------------------------------------------------------
# ----------------------- CREACIÓN DE NUEVOS DATOS (COLUMNAS) -----------------
# -----------------------------------------------------------------------------

# crear nueva columna para saber si es pelicula de USA, colaboración de USA u otra
def IS_USA(paises):
  list_paises = paises.split(',')
  if len(list_paises) > 1:
    for pais in list_paises:
      if pais == "United States":
        return "USA Collab"
      else:
        pass
  elif len(list_paises) == 1:
    return "USA"
  return "Other"
  
# crear nueva columna
movies_df = movies_df.assign(country_category = movies_df.apply( lambda row: IS_USA(row.Country),axis=1))


# crear una columna para cada uno de los generos más frecuentes
genres_to_consider=movies_df.Genres.apply( lambda x: x.split(",") ).explode().value_counts(normalize=True)[0:12].index.tolist()
for genre in genres_to_consider:
  r = movies_df.Genres.apply(lambda x: genre in x)
  movies_df=movies_df.assign( **{genre: r} )



# ---------------------------------------------------------------------------------------------------------------
# ------------- COMPARACIÓN DE MEDIAS ENTRE DOS POBLACIONES NORMALES O APROXIMADAMENTE NORMALES -----------------
# ---------------------------------------------------------------------------------------------------------------

# primero comparamos los histogramas para ver la forma de las distribuciones
pop1 = movies_df[ movies_df.Family ].IMDb
pop2 = movies_df[ movies_df.Fantasy ].IMDb
sns.histplot(x=pop1)
sns.histplot(x=pop2, color="red")

# gráficos de probabilidad normal
plot = sm.ProbPlot( pop1, dist="norm")
plot.probplot(line="r")
plot = sm.ProbPlot( pop2, dist="norm")
plot.probplot(line="r")

# como hemos visto que los ambas poblaciones son aproximadamente normales, vamos a hacer una prueba basada en la t de student
# - alternative: Puede ser two-sided, larger o smaller dependiendo de si la hipótesis alternativa es de dos colas ( ≠ ) 
# o una cola ( > ,  < ).
# - user_var: Este argumento es cierto cuando se asume que ambas varianzas poblacionales son diferentes y falso en caso contrario.
# La prueba devuelve el valor del estadístico de la t de Student, el p-valor, y los grados de libertad de la prueba. 
# Como podemos ver con el p-valor y asumiendo un  α=0.05
sm.stats.ttest_ind(pop1, pop2, alternative='two-sided', usevar='unequal')


# para comparar el posicionamiento de dos poblaciones y ver cual tiende a ser mayor a la otra usaremos pruebas no parametricas ->

# Prueba Mann-Whitney U
# H0:P(xi>yj)=1/2 , la probabilidad de encontrar valores más grandes de la primera población es exactamente 50%.
# H1:P(xi>yj)≠1/2 , implicando esto que en una de las dos poblaciones tendemos a encontrar valores más grandes.
# si pvalue<=0.05 -> H1
# si pvalue>0.05 -> H0
stats.mannwhitneyu(pop1, pop2)


# comparacion de las calificaciones de IMDb de las pelis de Horror y Drama y ver si alguno de los dos generos recibe mejores
# puntuaciones de la crítica
pop1 = movies_df[ movies_df.Drama ].IMDb
pop2 = movies_df[ movies_df.Horror ].IMDb

# 1º Comprobar la normalidad o cercanía a la normalidad de ambas poblaciones
sns.histplot(x=pop1, stat="density", common_norm=False)
sns.histplot(x=pop2, color="red", stat="density", common_norm=False)
plot = sm.ProbPlot( pop1, dist="norm")
plot.probplot(line="r")
plot = sm.ProbPlot( pop2, dist="norm")
plot.probplot(line="r")
axes = sns.kdeplot( (pop1 - pop1.mean() )/(pop1.std()), common_norm=False  )
sns.lineplot( y=[ stats.norm.pdf(x) for x in np.arange(-3, 3, 0.01) ], x= [x for x in np.arange(-3, 3, 0.01)] )
axes = sns.kdeplot( (pop2 - pop2.mean() )/(pop2.std()), common_norm=False  )
sns.lineplot( y=[ stats.norm.pdf(x) for x in np.arange(-3, 3, 0.01) ], x= [x for x in np.arange(-3, 3, 0.01)] )

# 2º Seleccionar la prueba más apropiada para el tipo de poblaciones tratadas y el objetivo establecido
sm.stats.ttest_ind(pop1, pop2, alternative='two-sided', usevar='unequal')
stats.mannwhitneyu(pop1, pop2)
print("Media Drama ",pop1.mean())
print("Media Horror ",pop2.mean())
# ** Observando que la hipotesis de Mann-Whitney nos dice que en Horror y en Drama encontramos que las dos distribuciones son diferentes,
# y sabiendo tambien que la media se va 1 punto de IMDb entre ambas, podemos decir que existen diferencias significativas. 
# Si que pienso que la diferencia es importante, ya que la diferencia de 1 punto entre ambas puede 
# considerarse relevante en una puntación de 10 puntos. **


# Ahora lo mismo analizando el runtime de las películas de aventuras y ciencia ficción
pop1 = movies_df[ movies_df.Adventure ].Runtime
pop2 = movies_df[ movies_df["Sci-Fi"] ].Runtime
sns.histplot(x=pop1, stat="density", common_norm=False)
sns.histplot(x=pop2, color="red", stat="density", common_norm=False)
plot = sm.ProbPlot( pop1, dist="norm")
plot.probplot(line="r")
plot = sm.ProbPlot( pop2, dist="norm")
plot.probplot(line="r")
print(sm.stats.ttest_ind(pop1, pop2, alternative='two-sided', usevar='unequal'))
print(stats.mannwhitneyu(pop1, pop2))
print("Media Adventure ",pop1.mean())
print("Media Sci-Fi ",pop2.mean())
# ** Podemos decir que no es seguro que ambas distribuciones sigan una distribución normal. 
# Para el analisis de diferencias entre Ciencia Ficción y Aventuras usaremos el test de Mann-Whitney. 
# Entre ambas poblaciones, las diferencias no son muy notables, únicamente varia en 1 la diferencia de medias y ademas 
# con los histogramas podemos ver que tienen una forma similar. **


# Ahora lo mismo analizando las puntuaciones en IMDb de las películas de Netflix con las Prime Video
pop1 = movies_df[ movies_df["Netflix"] == 1 ].IMDb
print("Numero de peliculas de Netflix -> ",movies_df[ movies_df["Netflix"] == 1 ].shape[0])
pop2 = movies_df[ movies_df["Prime Video"] == 1 ].IMDb
print("Numero de peliculas de Prime Video -> ",movies_df[ movies_df["Prime Video"] == 1 ].shape[0])
sns.histplot(x=pop1, stat="density", common_norm=False)
sns.histplot(x=pop2, color="red", stat="density", common_norm=False)
plot = sm.ProbPlot( pop1, dist="norm")
plot.probplot(line="r")
plot = sm.ProbPlot( pop2, dist="norm")
plot.probplot(line="r")
print(sm.stats.ttest_ind(pop1, pop2, alternative='two-sided', usevar='unequal'))
print(stats.mannwhitneyu(pop1, pop2))
print("Media Netflix ",pop1.mean())
print("Media Prime Video ",pop2.mean())
# ** Observando los valores de las medias y del histograma se puede observar que hay una pequeña diferencia entre las dos poblaciones, 
# por otra parte, los test y el histograma nos dice que las poblaciones son diferentes entre ellas. 
# Diría que las diferencias no son importantes, no por nada, sino porque para Prime Video contamos con 1802 peliculas, 
# mientras que para Netlix tenemos tan solo 1000, por lo que tal vez la diferencia en la forma de las poblaciones 
# se deba a esta diferencia de valores. **


# -----------------------------------------------------------------------
# ------------- COMPARACIÓN DE VARIANZAS EN POBLACIONES -----------------
# -----------------------------------------------------------------------

# -- Prueba de Levene --
# prueba para evaluar la igualdad de las varianzas para una variable calculada para dos o más grupos
# - H0:σ1=σ2=⋯=σk       // todas las varianzas son iguales
# - H1:∃i,j,σi≠σk       // al menos dos varianzas no son iguales
# si pvalue<=0.05 -> H1
# si pvalue>0.05 -> H0
pop1 = movies_df[ movies_df.Drama ].IMDb
pop2 = movies_df[ movies_df.Horror ].IMDb
print(stats.levene(pop1, pop2))
print(pop1.std(), pop2.std())

# Ahora vamos a ver si tenemos evidencias para pensar que las varianzas de las puntuaciones de IMDb de las peliculas de Netflix 
# y Prime Video son diferentes
pop1 = movies_df[ movies_df["Netflix"] == 1 ].IMDb
pop2 = movies_df[ movies_df["Prime Video"] == 1 ].IMDb
print(stats.levene(pop1, pop2))
print(pop1.std(), pop2.std())
# ** El contraste de hipótesis arroja que no existen evidencias suficientes para pensar que las varianzas son diferentes. **



# -----------------------------------------------------------------------
# ------------- ANOVA CON UN FACTOR Y MÚLTIPLES FACTORES ----------------
# -----------------------------------------------------------------------

# ANOVA (ANalysis Of VAriance) / Análisis de varianza de Fisher (debido al uso de la distribución F de Fisher)
# colección de modelos estadísticos y sus procedimientos en el cual la varianza está particionada en ciertos componentes debidos
# a diferentes variables explicativas.
# - H0:μ1=μ2=⋯=μk 
# - H1:∃i,jμi≠μj
lm = ols( 'IMDb ~ C(Age)', data=movies_df ).fit(cov_type="HC3")
anova_table = sm.stats.anova_lm(lm, typ=1,  robust="HC3")
anova_table

# - ols = (ANOVA por mínimos cuadrados)
# - 'IMDb ~ C(Age)' 
# IMDb:variable cuya respuesta media se quiere analizar. 
# C(Age): Factores o variables que comprondrán las subpoblaciones de estudio ( la C indica que es una variable categórica ).

# Si el analisis de ANOVA parece que diga que una media poblacional es diferente al resto, hay que analizar cual es,
# esto puede hacerse con intervalos de Tukey

# Tukey HSD
# intervalos que se solapan en caso de que no haya evidencias de diferencias significativas en las medias, 
# NO se solapan cuando si existen
tukey = sm.stats.multicomp.pairwise_tukeyhsd(movies_df.IMDb, movies_df.Age, alpha=0.01)
tukey.plot_simultaneous(comparison_name="18+")

# Despues de hacer el ANOVA está bien comprobarlo, usaremos una prueba de Levene para esto
populations = []
for (values, group) in movies_df.groupby(['Age']):
  p = movies_df.IMDb[ group.index ]
  populations.append(p)
  print( p.std() )
stats.levene(*populations)

# Ahora vamos a analizar los residuos para determinar la posible heterocedasticidad de las poblaciones.
# si existe homocedasticidad, la media de los residuos debería estar centrada en 0
residuals = lm.resid
sns.pointplot( data=movies_df.assign(residuals=residuals, factor=movies_df.apply(lambda row: row.Age, axis=1)), x="factor", y="residuals", ci="sd", join=False )

# por último habría que comprobar la normalidad de las subpoblaciones, sabiendo que cuando las poblaciones son normales, 
# los residuos tambien deberían serlo
plot = sm.ProbPlot(residuals, dist="norm")
plot.probplot(line="r")
sns.histplot(residuals)
axes = sns.kdeplot( (residuals - residuals.mean() )/(residuals.std()), common_norm=False  )
sns.lineplot( y=[ stats.norm.pdf(x) for x in np.arange(-3, 3, 0.01) ], x= [x for x in np.arange(-3, 3, 0.01)] )


# ANOVA con un factor con IMDb como variable de respuesta y decada como variable a analizar
def get_decade(year):
  return year//10 * 10

movies_df = movies_df.assign(decade= movies_df.Year.apply(get_decade) )
new_df = movies_df[ (movies_df.decade>1970) & (movies_df.decade<2020) ]

lm = ols( 'IMDb ~ C(decade)', data=new_df ).fit(cov_type="HC3")
anova_table = sm.stats.anova_lm(lm, typ=1,  robust="HC3")
tukey = sm.stats.multicomp.pairwise_tukeyhsd(new_df.IMDb, new_df.decade, alpha=0.01)
tukey.plot_simultaneous()


# ANOVA con múltiples factores
movies_df = movies_df.assign( Adult= movies_df.Age.apply(lambda x: x=="18+") )
lm = ols( 'IMDb ~ C(Adult) + C(country_category) + C(Adult)*C(country_category)', data=movies_df ).fit(cov_type="HC3")
anova_table = sm.stats.anova_lm(lm, typ=2, robust="HC3")

# como la interacción no es significativa (p-valor=0,187), por lo que vamos a repetir el análisis para una suma de cuadrados de tipo III
lm = ols( 'IMDb ~ C(Adult) + C(country_category) + C(Adult)*C(country_category)', data=movies_df ).fit(cov_type="HC3")
anova_table = sm.stats.anova_lm(lm, typ=3, robust="HC3")

# Los factores simples si que resultan significativos, por lo que estudiaremos individual cada factor con Tukey
# - country_category:
tukey = sm.stats.multicomp.pairwise_tukeyhsd(movies_df.IMDb, movies_df.country_category, alpha=0.05)
tukey.plot_simultaneous(comparison_name="USA")

# - Adult (para este usaremos una simple comparación de las medias ya que se trata de únicamente 2 niveles):
print( movies_df[movies_df.Adult].IMDb.mean(), movies_df[movies_df.Adult==False].IMDb.mean() )


# Aquí analizamos mediante un gráfico de interacciones las medias de country_category y Adult respecto de IMDb
sm.graphics.interaction_plot(x=movies_df.country_category, trace=movies_df.Adult, response=movies_df.IMDb)

# Ahora analizaremos la homocedasticidad de las varianzas
residuals = lm.resid
plt.figure( figsize=(10, 5) )
sns.pointplot( data=movies_df.assign(residuals=residuals, factor=movies_df.apply(lambda row: row.country_category+"-"+str(row.Adult), axis=1)), x="factor", y="residuals", ci="sd", join=False )
plt.xticks(rotation=45)

# Analizamos la normalidad de los residuos
plot = sm.ProbPlot(residuals, dist="norm")
plot.probplot(line="r")
sns.histplot(residuals)
axes = sns.kdeplot( (residuals - residuals.mean() )/(residuals.std()), common_norm=False  )
sns.lineplot( y=[ stats.norm.pdf(x) for x in np.arange(-3, 3, 0.01) ], x= [x for x in np.arange(-3, 3, 0.01)] )


# --- ANOVA de (Age y Disney) sobre la media de las puntuaciones IMDb ---
disney_df = movies_df[ (movies_df.Age!="18+") & (movies_df.Age!="16+") ]
disney_df = disney_df.assign( disney=movies_df["Disney+"] )
lm = ols( 'IMDb ~ C(Age) + C(disney) + C(Age)*C(disney)', data=disney_df ).fit(cov_type="HC3")
anova_table = sm.stats.anova_lm(lm, typ=2, robust="HC3")
anova_table

# vamos a analizar los niveles que se obtiene una respuesta diferente de la media haciendo uso de gráficos de interacción
sm.graphics.interaction_plot(x=disney_df.disney, trace=disney_df.Age, response=disney_df.IMDb)

# analisis de los residuos mediante Tukey
residuals = lm.resid
plt.figure( figsize=(10, 5) )
sns.pointplot( data=disney_df.assign(residuals=residuals, factor=disney_df.apply(lambda row: row.Age+"-"+str(row.disney), axis=1)), x="factor", y="residuals", join=False, ci="sd" )
plt.xticks(rotation=45)

# ahora analizamos la normalidad de los residuos
plot = sm.ProbPlot(residuals, dist="norm")
plot.probplot(line="r")
sns.histplot(residuals)
axes = sns.kdeplot( (residuals - residuals.mean() )/(residuals.std()), common_norm=False  )
sns.lineplot( y=[ stats.norm.pdf(x) for x in np.arange(-3, 3, 0.01) ], x= [x for x in np.arange(-3, 3, 0.01)] )


# --- ANOVA de (decada y Disney) sobre la media de las puntuaciones IMDb ---
new_df = new_df[ (new_df.Age!="18+") & (new_df.Age!="16+") ]
new_df = new_df.assign( disney=new_df["Disney+"] )
lm = ols( 'IMDb ~ C(decade) + C(disney) + C(decade)*C(disney)', data=new_df ).fit(cov_type="HC3")
anova_table = sm.stats.anova_lm(lm, typ=2, robust="HC3")

tukey = sm.stats.multicomp.pairwise_tukeyhsd(new_df.IMDb, new_df.disney, alpha=0.05)
tukey.plot_simultaneous(comparison_name=1)
print( movies_df[movies_df.Adult].IMDb.mean(), movies_df[movies_df.Adult==False].IMDb.mean() )

sm.graphics.interaction_plot(x=new_df.decade, trace=new_df.disney, response=new_df.IMDb)

residuals = lm.resid
plt.figure( figsize=(10, 5) )
sns.pointplot( data=new_df.assign(residuals=residuals, factor=new_df.apply(lambda row: str(row.disney)+"-"+str(row.decade), axis=1)), x="factor", y="residuals", ci="sd", join=False )
plt.xticks(rotation=45)

plot = sm.ProbPlot(residuals, dist="norm")
plt.figure()
plot.probplot(line="r")
plt.figure()
sns.histplot(residuals)
plt.figure()
axes = sns.kdeplot( (residuals - residuals.mean() )/(residuals.std()), common_norm=False  )
sns.lineplot( y=[ stats.norm.pdf(x) for x in np.arange(-3, 3, 0.01) ], x= [x for x in np.arange(-3, 3, 0.01)] )

# --- CONCLUSIONES ---
# - El factor de Disney individual si es estadísticamente significativo, en cambio la de decada no.
#
# - La interacción como se puede ver en el gráfico de interacciones si que es significativa, y fuerte ya que las lineas se cruzan.
# - Las películas de antes de los noventa, años en los que se sacaron clásicos en Disney y a partir del 2000, 
#   que pueden considerarse como las últimas películas son las mejor valoradas, esto puede deberse a que, 
#   entre los 1990 y los 2000, se públicaron muchas películas que no causaron mucho impacto.
#
# - Como podemos ver que los residuos son normales y las subpoblaciones son lo suficientemente grandes como para fiarnos, 
#   podemos asegurarnos del ANOVA realizado.



