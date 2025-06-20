# IMPORTS
# importar la biblioteca para análisis de datos
import numpy as np
# importar la biblioteca para graficación
import matplotlib.pyplot as plt
# importar la biblioteca para manipulación y tratamiento de datos
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from dotenv import load_dotenv
import os

def entrenarModelo():
    # Cargar variables
    load_dotenv()
    # 1. Cargar los datos desde el archivo CSV
    # Asegúrate de que los archivos csv estén en el mismo directorio
    # o proporciona la ruta completa a los archivos.
    lista_archivos = ['archivos_apoyo/bitacora_del_todo.csv']
    lista_de_dataframes = []
    for archivo in lista_archivos:
        try:
            df = pd.read_csv(archivo)
            lista_de_dataframes.append(df)
        except FileNotFoundError:
            print(f"¡Advertencia! El archivo '{archivo}' no fue encontrado y será omitido.")
        except Exception as e:
            print(f"¡Error al leer el archivo '{archivo}': {e}")
    data = pd.concat(lista_de_dataframes, ignore_index=True)


    # 2. Preprocesar los datos
    #Mantener columnas necesarias
    data = data[["sucursal","id_cliente","evento","estatus_cli","limite","diasDesdeIngreso","monto"]]

    # Identificar y manejar valores faltantes
    print("Valores nulos antes del tratamiento:")
    print(data.isnull().sum())
    data = data.dropna()  # Elimina filas con valores faltantes
    print("Valores nulos después del tratamiento:")
    print(data.isnull().sum())

    # 2.1 Eliminar registros con un valor -1 en la columna 'estatus_cli'
    valor_a_eliminar = -1  # Cambia esto al valor que deseas eliminar
    print(f"Eliminando registros donde 'estatus_cli' es igual a {valor_a_eliminar}")
    data = data[data['estatus_cli'] != valor_a_eliminar]
    print(f"Registros restantes: {len(data)}")

    # valores distintos de cada columna por cliente
    data.groupby('id_cliente').nunique()

    # registros por evento
    data.groupby('evento').size()

    # registros por clase
    data.groupby('estatus_cli').size()

    # Cantidad de datos distintos para cada columna de cada cliente
    data[data['evento'] == 'PAG'].groupby('id_cliente').nunique()

    # monto acumulado de cada cliente de cada evento para cada día
    data.groupby(['id_cliente','diasDesdeIngreso','evento'])['monto'].sum().reset_index()

    # Completa los días faltantes para cada cliente en un DataFrame.

    # 1. Encuentra el primer y último día para cada cliente.
    min_max_dias = data.groupby('id_cliente')['diasDesdeIngreso'].agg(['min', 'max']).reset_index()

    # 2. Crea un DataFrame con todos los días para cada cliente.
    df_todos_los_dias = pd.DataFrame()
    for _, row in min_max_dias.iterrows():
        cliente_id = row['id_cliente']
        min_dia = row['min']
        max_dia = row['max']
        rango_dias = pd.DataFrame({'diasDesdeIngreso': range(0, max_dia + 1)})
        rango_dias['id_cliente'] = cliente_id
        df_todos_los_dias = pd.concat([df_todos_los_dias, rango_dias], ignore_index=True)

    # 3. Combina los DataFrames y rellena los días faltantes.
    print("Columnas en df_todos_los_dias:", df_todos_los_dias.columns)
    print("Columnas en data:", data.columns)
    df_completo = pd.merge(df_todos_los_dias, data, how='left', on=['id_cliente', 'diasDesdeIngreso'])
    df_completo['monto'] = df_completo['monto'].fillna(0)
    df_completo['evento'] = df_completo['evento'].fillna('PAG')
    df_completo[['limite', 'estatus_cli']] = df_completo[['limite', 'estatus_cli']].ffill()
    df_completo

    # Crea una tabla pivote para obtener todos los registros de un cliente en un renglón
    data = df_completo
    df_pivot = data.pivot_table(index=['id_cliente', 'diasDesdeIngreso'],
                                columns='evento',
                                values='monto',
                                aggfunc='sum',
                                fill_value=0)  # Llena los NaN con 0
    df_maximos = data.pivot_table(index=['id_cliente', 'diasDesdeIngreso'],
                                    values=['limite', 'estatus_cli'],
                                    aggfunc='max')
    # Combina los resultados
    data = pd.concat([df_pivot, df_maximos], axis=1).reset_index()
    data


    # Utilizaremos solo los datos de una cantidad específica de los últimos días (DATA_DIAS)
    # dejando fuera los últimos OFF_DIAS días. Se eliminan los distribuidores que tengan menos todos esos días.

    DATA_DIAS = int(os.getenv("DATA_DIAS"))  # cantidad de días que se utilizarán para entrenar y evaluar el modelo
    OFF_DIAS = int(os.getenv("OFF_DIAS"))  # cantidad de los últimos días que se dejarán fuera para evitar patrones obvios en los distribuidores que desertaron

    # 1. Cuenta los días únicos por cliente.
    conteo_dias_por_cliente = data.groupby('id_cliente')['diasDesdeIngreso'].max()

    # 2. Obtiene los clientes con al menos DATA_DIAS + OFF_DIAS días.
    MIN_DIAS = DATA_DIAS + OFF_DIAS
    clientes_con_suficientes_dias = conteo_dias_por_cliente[conteo_dias_por_cliente >= MIN_DIAS].index

    # 3. Filtra el DataFrame para mantener solo los clientes con al menos MIN_DIAS días.
    df_filtrado = data[data['id_cliente'].isin(clientes_con_suficientes_dias)].copy()

    # 4. Ordena los días de cada cliente de mayor a menor.
    df_filtrado.sort_values(by=['id_cliente', 'diasDesdeIngreso'], ascending=[True, False], inplace=True)

    # 5. Elimina los primeros OFF_DIAS renglones de cada cliente
    df_filtrado['group_row_number'] = df_filtrado.groupby('id_cliente').cumcount()
    df_filtrado = df_filtrado[df_filtrado['group_row_number'] >= OFF_DIAS].reset_index(drop=True)
    df_filtrado = df_filtrado.drop(columns=['group_row_number'])

    # 6. Agrupa por cliente y se queda con los primeros DATA_DIAS días.
    df_top_DATA_DIAS_dias = df_filtrado.groupby('id_cliente').head(DATA_DIAS).copy()

    # 7. Renumera los días para cada cliente, comenzando desde 0 y de mayor a menor.
    df_top_DATA_DIAS_dias['diasAnteriores'] = df_top_DATA_DIAS_dias.groupby('id_cliente')['diasDesdeIngreso'].transform(lambda x: x.max() - x)

    # Elimina numeración de días desde el ingreso de los clientes
    df_top_DATA_DIAS_dias.drop(columns=['diasDesdeIngreso'], inplace=True)

    # Reduce un DataFrame agrupando renglones consecutivos de cada cliente.

    GPO_SZ = int(os.getenv("GPO_SZ")) # cantidad de días por grupo

    # Obtiene un nuevo DataFrame con un renglón por cada GPO_SZ renglones consecutivos
    # por cliente.
    grupos = []
    for cliente_id, grupo_cliente in df_top_DATA_DIAS_dias.groupby('id_cliente'):
        num_renglones = len(grupo_cliente)
        num_grupos_completos = num_renglones // GPO_SZ
        for i in range(num_grupos_completos):
            inicio = i * GPO_SZ
            fin = (i + 1) * GPO_SZ
            grupo_GPO_SZ_renglones = grupo_cliente.iloc[inicio:fin]
            grupos.append({
                'id_cliente': cliente_id,
                'dia_ini': inicio,
                'dia_fin': fin-1,
                'BOD': grupo_GPO_SZ_renglones['BOD'].sum(),
                'CM': grupo_GPO_SZ_renglones['CM'].sum(),
                'Canje Vale': grupo_GPO_SZ_renglones['Canje Vale'].sum(),
                'Incremento de Limite': grupo_GPO_SZ_renglones['Incremento de Limite'].sum(),
                'PAG': grupo_GPO_SZ_renglones['PAG'].sum(),
                'limite': grupo_GPO_SZ_renglones['limite'].mean(),
                'estatus_cli': grupo_GPO_SZ_renglones['estatus_cli'].iloc[0]
            })
    df_grupo_GPO_SZ_renglones = pd.DataFrame(grupos)
    df_grupo_GPO_SZ_renglones


    # 7. Pivota el DataFrame para convertir cada día en una columna.
    df_pivot = df_grupo_GPO_SZ_renglones.pivot_table(index='id_cliente',
                                        columns='dia_ini',
                                        values=['BOD','CM','Canje Vale','Incremento de Limite','PAG','limite'],
                                        aggfunc='first')  # Toma el primer valor de cada día

    # Renombra las columnas para que sean más descriptivas
    df_pivot.columns = [f'{col[0]}_gpo_{col[1] // GPO_SZ}' for col in df_pivot.columns]
    df_estatus = df_top_DATA_DIAS_dias.groupby('id_cliente')['estatus_cli'].max().reset_index()
    df_pivot = pd.merge(df_pivot, df_estatus, on='id_cliente')
    df_pivot.head().T

    data = df_pivot
    data.groupby('id_cliente').size()
    data.info()

    # Seleccionar las características y la variable objetivo
    # Excluimos 'id_cliente' porque no es relevante para el modelo
    # 'estatus_cli' es nuestra variable objetivo
    X_raw = data.drop(columns=['id_cliente','estatus_cli'], axis=1)
    y = data['estatus_cli']

    # Normaliza los datos utilizando el escalador de datos
    from sklearn.preprocessing import StandardScaler

    dataScaler = StandardScaler()
    scaler = dataScaler.fit(X_raw)
    dataScaled = scaler.transform(X_raw)

    # muestra el arreglo resultante
    dataScaled

    # crea un dataframe con los datos normalizados
    X = pd.DataFrame(dataScaled)
    X.columns = X_raw.columns

    # 3. Dividir los datos en conjuntos de entrenamiento y prueba
    # Usamos una división 80/20 para entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # muestra la forma de los distintos conjuntos de datos obtenidos
    print("Datos de entrenamiento=", X_train.shape, y_train.shape)
    print("Datos de prueba=", X_test.shape, y_test.shape)

    # utiliza validación cruzada de 10 folds para evaluar el desempeño promedio
    # de bosques aleatorios
    from sklearn.model_selection import cross_validate
    from sklearn.ensemble import RandomForestClassifier

    rfcInicial_model = RandomForestClassifier()
    scores = pd.DataFrame(cross_validate(rfcInicial_model, X_train, y_train, cv=10, return_train_score=True))

    # despliega los score promedio de entrenamiento y validación, así como los
    # resultados obtenidos para cada uno de los 10 folds
    print("score promedio de entrenamiento = ", scores['train_score'].mean())
    print("score promedio de validación = ", scores['test_score'].mean())
    scores

    # determina, entre algunas alternativas, los mejores valores de hiperparámetros
    # para construir un bosque aleatorio para el problema
    from sklearn.model_selection import GridSearchCV

    parameters = {'max_depth': [5, 7, 9, 11],
                'max_features': ['sqrt', 'log2', None],
                'n_estimators': [10, 30, 60, 100]}
    rfc_grid = GridSearchCV(RandomForestClassifier(random_state=1), param_grid = parameters,
                            return_train_score=True)
    rfc_grid.fit(X_train, y_train)

    # despliega los mejores hiperparámetros encontrados
    print("Mejores hiperparámetros\n",rfc_grid.best_params_)

    # quédate con el Bosque Aleatorio con los mejores hiperparámetros encontrados y
    # despliega su score con los datos del conjunto de prueba.
    rfc_model = rfc_grid.best_estimator_
    rfc_model.score(X_test, y_test)

    # calcula las matriz de confusión y las métricas de evaluación con el conjunto
    # de prueba para el mejor Bosque Aleatorio


    ConfusionMatrixDisplay.from_estimator(rfc_model, X_test, y_test)
    print(classification_report(y_test, rfc_model.predict(X_test)))
    print(f"Number of test samples: {X_test.shape[0]}")
    print(f"Number of test labels: {y_test.shape[0]}")

    pd.Series(rfc_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

    # GUARDAR MODELO Y SCALER
    import pickle
    modelo_dict = {
        'modelo': rfc_model,
        'scaler': scaler,
        'columnas': X_train.columns.tolist()
    }

    with open('modelo_entrenado/modelo_abandono.pkl', 'wb') as f:
        pickle.dump(modelo_dict, f)
    # Optional: return something useful
    return "Modelo y scaler guardados correctamente."

entrenarModelo()