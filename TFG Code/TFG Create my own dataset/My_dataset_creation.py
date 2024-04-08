from datasets import load_dataset
import matplotlib.pyplot as plt

# %%
traf = load_dataset("monash_tsf", "traffic_hourly")

# %% Transform dataset in a DataFrame (Theres is json, pandas...)
traf = traf.to_pandas()

# %%
traf_dic = traf.to_dict(orient="index")

# %% REPRESENTAR

# Accede al primer diccionario en la lista 'reddit'
primer_diccionario = traf_dic[0]
test_ex = primer_diccionario
# Accede al array 'target' dentro del primer diccionario
primer_array = primer_diccionario['target']

# Crea un gráfico de línea con el primer array
plt.plot(primer_array)

# Ajusta los límites del eje y para que se adapten a los valores del array
plt.ylim(min(primer_array), max(primer_array))

# Ajusta los límites del eje x para que se adapten a los valores del array
plt.xlim(17000, len(primer_array))

# Etiqueta del eje x
plt.xlabel('Índice')

# Etiqueta del eje y
plt.ylabel('Valor')

# Título del gráfico
plt.title('Primer array de la lista')

# Muestra el gráfico
plt.show()
