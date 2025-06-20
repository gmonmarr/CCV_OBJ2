# funciones_apoyo.py
# Este archivo contiene funciones de apoyo para la función main.py
# NO REALIZAR CAMBIOS

import pandas as pd 
import numpy as np
import math
import os
import joblib
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import verbose_print

class FuncionesParaPrediccion:

    # inicializamos con el contructor
    def __init__(self, path, verbose = False):
        if verbose:
            print("Verificando archivos .csv")
        
        try:
            self.bitacora_cambios = pd.read_csv(os.path.join(path, 'bitacora_cambios.csv'))
        except Exception as e:
            print(f"Error al leer 'bitacora_cambios.csv': {e}")

        try:
            self.canje_vales = pd.read_csv(os.path.join(path, 'canje_vales.csv'))
        except Exception as e:
            print(f"Error al leer 'canje_vales.csv': {e}")

        try:
            self.distribuidores = pd.read_csv(os.path.join(path, 'distribuidores.csv'))
        except Exception as e:
            print(f"Error al leer 'distribuidores.csv': {e}")

        try:
            self.pagos_bonos = pd.read_csv(os.path.join(path, 'pagos_bonos.csv'))
        except Exception as e:
            print(f"Error al leer 'pagos_bonos.csv': {e}")

        if verbose:
            print("Archivos csv leidos con éxito")

        try:
            self.limpiar_pagos_bonos()
        except Exception as e:
            print(f"Error al limpiar pagos_bonos: {e}")

        try:
            # Se almacena las fechas de ingreso de los distribuidores en un diccionario
            fecha_ingreso = self.distribuidores.set_index('id_cliente')['fecha_ingreso'].to_dict()
        except Exception as e:
            print(f"Error al construir el diccionario de fechas de ingreso: {e}")
            fecha_ingreso = {}

        try:
            self.bitacora_cambios = self.dias_desde_ingreso_DF(self.bitacora_cambios, 'fecha_bc', fecha_ingreso)
            self.bitacora_cambios = self.bitacora_cambios.drop(columns=['fecha_bc'])
        except Exception as e:
            print(f"Error procesando 'bitacora_cambios': {e}")

        try:
            self.canje_vales = self.dias_desde_ingreso_DF(self.canje_vales, 'fecha_vale', fecha_ingreso)
            self.canje_vales = self.canje_vales.drop(columns=['fecha_vale'])
        except Exception as e:
            print(f"Error procesando 'canje_vales': {e}")

        try:
            self.pagos_bonos = self.dias_desde_ingreso_DF(self.pagos_bonos, 'fecha_pb', fecha_ingreso)
            self.pagos_bonos = self.pagos_bonos.drop(columns=['fecha_pb'])
        except Exception as e:
            print(f"Error procesando 'pagos_bonos': {e}")


    def limpiar_pagos_bonos(self, verbose =  True):
        try:
            self.pagos_bonos['tipo_pago_bono'] = self.pagos_bonos['tipo_pago_bono'].str.strip()
            self.pagos_bonos['tipo_pago_bono'] = self.pagos_bonos['tipo_pago_bono'].replace({'P': 'PAG', 'B': 'BOD'})
            if verbose:
                print("Datos de archivos estandarizados con éxito")

        except KeyError:
            print("Error: La columna 'tipo_pago_bono' no existe en el DataFrame 'pagos_bonos'.")
        except AttributeError:
            print("Error: 'pagos_bonos' no está definido correctamente o la columna contiene valores nulos no convertibles.")
        except Exception as e:
            print(f"Error inesperado al limpiar 'pagos_bonos': {e}")


    def dias_desde_ingreso_DF(self, df, fecha_DF, fecha_ingreso):
        try:
            # Convertir la columna de fecha del DataFrame
            df[fecha_DF] = pd.to_datetime(df[fecha_DF], format='%Y-%m-%d', errors='coerce')
        except Exception as e:
            print(f"Error al convertir '{fecha_DF}' con formato '%Y-%m-%d': {e}")
            try:
                df[fecha_DF] = pd.to_datetime(df[fecha_DF], format='%d/%m/%Y', errors='coerce', dayfirst=True)
            except Exception as e:
                print(f"Error alternativo al convertir '{fecha_DF}' con formato '%d/%m/%Y': {e}")

        try:
            # Mapear la fecha de ingreso desde el diccionario
            df['diasDesdeIngreso'] = df['id_cliente'].map(fecha_ingreso)
        except Exception as e:
            print(f"Error al mapear fechas de ingreso: {e}")

        try:
            df['diasDesdeIngreso'] = pd.to_datetime(df['diasDesdeIngreso'], format='%Y-%m-%d', errors='coerce')
        except Exception as e:
            print(f"Error al convertir fechas de ingreso con formato '%Y-%m-%d': {e}")
            try:
                df['diasDesdeIngreso'] = pd.to_datetime(df['diasDesdeIngreso'], format='%d/%m/%Y', errors='coerce', dayfirst=True)
            except Exception as e:
                print(f"Error alternativo al convertir fechas de ingreso con formato '%d/%m/%Y': {e}")

        try:
            df['diasDesdeIngreso'] = (df[fecha_DF] - df['diasDesdeIngreso']).dt.days
            df2 = df[df['diasDesdeIngreso'] >= 0].copy()
            return df2
        except Exception as e:
            print(f"Error al calcular la diferencia de días: {e}")
            return df  # O devuelve un df vacío si prefieres: pd.DataFrame()

    
    def BitacoraDelTodo(self, status_cli=True, colSucursal=True):
        try:
            df_bc = self.bitacora_cambios.copy()
            df_cv = self.canje_vales.copy()
            df_dist = self.distribuidores.copy()
            df_pb = self.pagos_bonos.copy()
        except Exception as e:
            print(f"Error al copiar los DataFrames: {e}")
            return pd.DataFrame()

        try:
            df_cv = df_cv[df_cv['tipo'].isin(['NORMAL', 'SEMANASANTA2024', 'VERANO2024', 'DISTRIBUIDOR PROMOCION', 'DV'])]
        except Exception as e:
            print(f"Error al filtrar canje_vales: {e}")

        try:
            df_bc_final = df_bc
            df_bc_final['monto'] = df_bc_final['nuevo_limite'] - df_bc_final['limite_bc']
            df_bc_final = df_bc_final[df_bc_final['monto'] != 0]
            df_bc_final = df_bc_final[['id_cliente', 'nuevo_limite', 'diasDesdeIngreso', 'motivo', 'monto', 'estatus_cli']]
            df_bc_final = df_bc_final.rename(columns={'nuevo_limite': 'limite', 'motivo': 'evento'})
            df_bc_final['evento'] = 'Incremento de Limite'
        except Exception as e:
            print(f"Error al procesar bitacora_cambios: {e}")

        try:
            df_cv = df_cv[['id_cliente', 'total', 'diasDesdeIngreso', 'tipo']].rename(
                columns={'total': 'monto', 'tipo': 'evento'})
            df_cv['evento'] = 'Canje Vale'
        except Exception as e:
            print(f"Error al procesar canje_vales: {e}")

        try:
            df_pb = df_pb[['id_cliente', 'tipo_pago_bono', 'monto', 'diasDesdeIngreso']]
            df_pb['tipo_pago_bono'] = df_pb['tipo_pago_bono'].str.strip()
            df_pb = df_pb.rename(columns={'tipo_pago_bono': 'evento'})
        except Exception as e:
            print(f"Error al procesar pagos_bonos: {e}")

        try:
            df_final = pd.concat([df_bc_final, df_cv, df_pb], ignore_index=True)
        except Exception as e:
            print(f"Error al concatenar los DataFrames: {e}")
            return pd.DataFrame()

        def get_latest_limit(row):
            try:
                client = row['id_cliente']
                event_day = row['diasDesdeIngreso']
                subset = df_bc_final[(df_bc_final['id_cliente'] == client) & (df_bc_final['diasDesdeIngreso'] <= event_day)]

                if subset.empty:
                    subset = df_bc[(df_bc['id_cliente'] == client) & (df_bc['diasDesdeIngreso'] <= event_day)]
                    subset = subset.rename(columns={'nuevo_limite': 'limite'})

                    if subset.empty:
                        subset = df_bc[df_bc['id_cliente'] == client].head(1)
                        subset = subset.rename(columns={'limite_bc': 'limite'})

                    if subset.empty:
                        subset = df_dist[df_dist['id_cliente'] == client].head(1)
                        subset = subset.rename(columns={'limite_distribuidor': 'limite'})
                        return subset['limite'].values[0] if not subset.empty else 0

                max_day = subset['diasDesdeIngreso'].max()
                same_day_entries = subset[subset['diasDesdeIngreso'] == max_day]
                return same_day_entries['limite'].max()
            except Exception as e:
                print(f"Error en get_latest_limit para cliente {row['id_cliente']}: {e}")
                return 0

        try:
            df_final['limite'] = df_final.apply(get_latest_limit, axis=1)
        except Exception as e:
            print(f"Error al aplicar get_latest_limit: {e}")

        def get_etiqueta_cliente(row):
            try:
                client = row['id_cliente']
                historial = df_bc[df_bc['id_cliente'] == client]
                estatus_actual = df_dist[df_dist['id_cliente'] == client]['estatus_actual_distribuidor'].values[0] \
                    if client in df_dist['id_cliente'].values else None
                morosos = ['JURIDICO', 'GESTORIA', 'CONVENIO COMERCIAL']

                if estatus_actual in morosos:
                    return 1
                if not historial.empty and historial['nuevo_estatus_cli'].isin(morosos).any():
                    return 1
                if estatus_actual == 'ACTIVO':
                    if not historial['nuevo_estatus_cli'].isin(['BLOQUEADO', 'JURIDICO', 'GESTORIA', 'CONVENIO COMERCIAL']).any():
                        return 0
                return -1
            except Exception as e:
                print(f"Error en get_etiqueta_cliente para cliente {row['id_cliente']}: {e}")
                return -1

        try:
            if status_cli:
                df_final['estatus_cli'] = df_final.apply(get_etiqueta_cliente, axis=1)
                df_final = df_final[['id_cliente', 'diasDesdeIngreso', 'evento', 'monto', 'limite', 'estatus_cli']]
                if colSucursal:
                    df_final = pd.merge(df_final, df_dist[['id_cliente', 'sucursal']], on='id_cliente', how='left')
                    df_final = df_final.rename(columns={'sucursal': 'Sucursal'})
                    df_final = df_final[['id_cliente', 'diasDesdeIngreso', 'evento', 'monto', 'limite', 'estatus_cli', 'Sucursal']]
            else:
                df_final = df_final[['id_cliente', 'diasDesdeIngreso', 'evento', 'monto', 'limite']]
                if colSucursal:
                    df_final = pd.merge(df_final, df_dist[['id_cliente', 'sucursal']], on='id_cliente', how='left')
                    df_final = df_final.rename(columns={'sucursal': 'Sucursal'})
                    df_final = df_final[['id_cliente', 'diasDesdeIngreso', 'evento', 'monto', 'limite', 'Sucursal']]
        except Exception as e:
            print(f"Error al aplicar filtros de columnas y merges finales: {e}")

        try:
            df_final = df_final.sort_values(by=['id_cliente', 'diasDesdeIngreso'])
            df_final.to_csv('archivos_apoyo/bitacora_del_todo.csv', index=False)
            print("Archivo CSV exportado como 'bitacora_del_todo.csv'")
        except Exception as e:
            print(f"Error al guardar archivo CSV: {e}")

        return df_final

    # FUNCION
    import joblib
    def predecir_abandono(self, datos, umbral):
        try:
            data = datos[['diasDesdeIngreso', 'id_cliente', 'evento', 'monto', 'limite', 'estatus_cli']]
            data = data.dropna()
        except Exception as e:
            print(f"Error al preparar las columnas iniciales: {e}")
            return pd.DataFrame()

        try:
            min_max_dias = data.groupby('id_cliente')['diasDesdeIngreso'].agg(['min', 'max']).reset_index()
            df_todos_los_dias = pd.DataFrame()

            for _, row in min_max_dias.iterrows():
                cliente_id = row['id_cliente']
                max_dia = row['max']
                rango_dias = pd.DataFrame({'diasDesdeIngreso': range(0, max_dia + 1)})
                rango_dias['id_cliente'] = cliente_id
                df_todos_los_dias = pd.concat([df_todos_los_dias, rango_dias], ignore_index=True)

            df_completo = pd.merge(df_todos_los_dias, data, how='left', on=['id_cliente', 'diasDesdeIngreso'])
            df_completo['monto'] = df_completo['monto'].fillna(0)
            df_completo['evento'] = df_completo['evento'].fillna('PAG')
            df_completo[['limite', 'estatus_cli']] = df_completo[['limite', 'estatus_cli']].ffill()

            all_events = np.array(['BOD', 'CM', 'Canje Vale', 'Incremento de Limite', 'PAG'])
            data_events = df_completo['evento'].unique()

            if not np.array_equal(np.sort(all_events), np.sort(data_events)):
                for e in np.setdiff1d(all_events, data_events):
                    df_completo = pd.concat([df_completo, pd.DataFrame([{'evento': e}])], ignore_index=True)

            df_completo['monto'] = df_completo['monto'].fillna(0)
            df_completo[['diasDesdeIngreso', 'id_cliente', 'limite', 'estatus_cli']] = df_completo[['diasDesdeIngreso', 'id_cliente', 'limite', 'estatus_cli']].ffill()
        except Exception as e:
            print(f"Error al completar días faltantes: {e}")
            return pd.DataFrame()

        try:
            df_pivot = df_completo.pivot_table(index=['id_cliente', 'diasDesdeIngreso'], columns='evento', values='monto', aggfunc='sum', fill_value=0)
            df_maximos = df_completo.pivot_table(index=['id_cliente', 'diasDesdeIngreso'], values=['limite', 'estatus_cli'], aggfunc='max')
            data = pd.concat([df_pivot, df_maximos], axis=1).reset_index()
        except Exception as e:
            print(f"Error al crear tabla pivote: {e}")
            return pd.DataFrame()

        try:
            DATA_DIAS = 30
            OFF_DIAS = 0
            MIN_DIAS = DATA_DIAS + OFF_DIAS

            conteo_dias_por_cliente = data.groupby('id_cliente')['diasDesdeIngreso'].max()
            clientes_con_suficientes_dias = conteo_dias_por_cliente[conteo_dias_por_cliente >= MIN_DIAS].index
            df_filtrado = data[data['id_cliente'].isin(clientes_con_suficientes_dias)].copy()
            df_filtrado.sort_values(by=['id_cliente', 'diasDesdeIngreso'], ascending=[True, False], inplace=True)
            df_filtrado['group_row_number'] = df_filtrado.groupby('id_cliente').cumcount()
            df_filtrado = df_filtrado[df_filtrado['group_row_number'] >= OFF_DIAS].reset_index(drop=True)
            df_filtrado = df_filtrado.drop(columns=['group_row_number'])

            df_top_DATA_DIAS_dias = df_filtrado.groupby('id_cliente').head(DATA_DIAS).copy()
            df_top_DATA_DIAS_dias['diasAnteriores'] = df_top_DATA_DIAS_dias.groupby('id_cliente')['diasDesdeIngreso'].transform(lambda x: x.max() - x)
            df_top_DATA_DIAS_dias.drop(columns=['diasDesdeIngreso'], inplace=True)
        except Exception as e:
            print(f"Error al filtrar días y clientes: {e}")
            return pd.DataFrame()

        try:
            GPO_SZ = 6
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
                        'dia_fin': fin - 1,
                        'BOD': grupo_GPO_SZ_renglones['BOD'].sum(),
                        'CM': grupo_GPO_SZ_renglones['CM'].sum(),
                        'Canje Vale': grupo_GPO_SZ_renglones['Canje Vale'].sum(),
                        'Incremento de Limite': grupo_GPO_SZ_renglones['Incremento de Limite'].sum(),
                        'PAG': grupo_GPO_SZ_renglones['PAG'].sum(),
                        'limite': grupo_GPO_SZ_renglones['limite'].mean(),
                        'estatus_cli': grupo_GPO_SZ_renglones['estatus_cli'].iloc[0]
                    })

            df_grupo_GPO_SZ_renglones = pd.DataFrame(grupos)
        except Exception as e:
            print(f"Error al agrupar por GPO_SZ: {e}")
            return pd.DataFrame()

        try:
            df_pivot = df_grupo_GPO_SZ_renglones.pivot_table(index='id_cliente',
                                                            columns='dia_ini',
                                                            values=['BOD', 'CM', 'Canje Vale', 'Incremento de Limite', 'PAG', 'limite'],
                                                            aggfunc='first')
            df_pivot.columns = [f'{col[0]}_gpo_{col[1] // GPO_SZ}' for col in df_pivot.columns]
            df_estatus = df_top_DATA_DIAS_dias.groupby('id_cliente')['estatus_cli'].max().reset_index()
            df_pivot = pd.merge(df_pivot, df_estatus, on='id_cliente')

            data = df_pivot
        except Exception as e:
            print(f"Error al pivotear datos finales: {e}")
            return pd.DataFrame()

        try:
            X_raw = data.drop(columns=['id_cliente', 'estatus_cli'], axis=1)
            y = data['estatus_cli']

            for col in self.columnas_entrenamiento:
                if col not in X_raw.columns:
                    X_raw[col] = 0

            X_raw = X_raw[self.columnas_entrenamiento]
            dataScaled = self.scaler.transform(X_raw)
            X = pd.DataFrame(dataScaled, columns=X_raw.columns)

            probabilidades = self.modelo.predict_proba(X)[:, 1]
            predicciones = (probabilidades >= umbral).astype(int)

            ids = data['id_cliente'].reset_index(drop=True)
            df_resultado = pd.DataFrame({
                'id_cliente': ids,
                'prediccion': predicciones,
                'probabilidad': probabilidades
            })

            return df_resultado
        except Exception as e:
            print(f"Error en la predicción: {e}")
            return pd.DataFrame()

    
    def predecir_abandono_to_csv(self, df_bdt, umbral):
        try:
            df = self.predecir_abandono(df_bdt, umbral)

            if df.empty:
                print("Advertencia: La predicción no generó resultados. No se exportará el archivo.")
                return

            df.to_csv('resultados/predecir_abandono.csv', index=False)
            print("Archivo CSV exportado como 'resultados/predecir_abandono.csv'")
        except Exception as e:
            print(f"Error al generar o guardar las predicciones: {e}")



    def hacer_prediccion(self, umbral):
        import joblib
        import pickle

        try:
            with open('modelo_entrenado/modelo_abandono.pkl', 'rb') as f:
                modelo_dict = pickle.load(f)

            self.modelo = modelo_dict['modelo']
            self.scaler = modelo_dict['scaler']
            self.columnas_entrenamiento = modelo_dict['columnas']
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            return

        try:
            df_bdt = pd.read_csv('archivos_apoyo/bitacora_del_todo.csv')
        except Exception as e:
            print(f"Error al leer el archivo CSV de entrada: {e}")
            return

        try:
            valores_a_eliminar = [0, 1]
            print(f"Eliminando registros donde 'estatus_cli' es igual a {valores_a_eliminar}")
            df_bdt = df_bdt[~df_bdt['estatus_cli'].isin(valores_a_eliminar)]
            print(f"Registros restantes: {len(df_bdt)}")
        except Exception as e:
            print(f"Error al filtrar los registros: {e}")
            return

        try:
            df_filtered = self.predecir_abandono_to_csv(df_bdt, umbral)
            return df_filtered
        except Exception as e:
            print(f"Error al ejecutar la predicción o exportar resultados: {e}")
            return
