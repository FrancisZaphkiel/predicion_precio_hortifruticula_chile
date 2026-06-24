import os
import re
import pandas as pd

def clean_numeric_str(val):
    if pd.isna(val):
        return 0.0
    val_str = str(val).strip().replace('"', '')
    if val_str == '' or val_str == '0,0000':
        return 0.0
    return float(val_str.replace(',', '.'))

def parse_weight_kg(unit_str):
    unit_str = str(unit_str).lower().strip()
    if unit_str.startswith('$/kilo') or unit_str == '$/kilo':
        return 1.0
    if '0,5 a 1 kilo' in unit_str or '0.5 a 1 kilo' in unit_str:
        return 0.75
    match = re.search(r'(\d+[\.,]?\d*)\s*kilo', unit_str)
    if match:
        val = match.group(1).replace(',', '.')
        return float(val)
    return None

def get_season_chile(month):
    if month in [12, 1, 2]:
        return 'Verano'
    elif month in [3, 4, 5]:
        return 'Otoño'
    elif month in [6, 7, 8]:
        return 'Invierno'
    elif month in [9, 10, 11]:
        return 'Primavera'
    return 'Desconocido'

def main():
    print("Iniciando pipeline de unificación y preprocesamiento cronológico...")
    
    # 1. Cargar y concatenar datasets
    csv_2025 = 'dataset/2025.csv'
    csv_2026 = 'dataset/2026.csv'
    
    if not os.path.exists(csv_2025) or not os.path.exists(csv_2026):
        print("Error: Los datasets originales no se encuentran en la carpeta dataset/")
        return
        
    df25 = pd.read_csv(csv_2025)
    df26 = pd.read_csv(csv_2026)
    
    print(f"Cargados exitosamente. 2025: {df25.shape[0]} filas | 2026: {df26.shape[0]} filas")
    df = pd.concat([df25, df26], ignore_index=True)
    print(f"Dataset combinado: {df.shape[0]} filas totales")
    
    # 2. Limpieza de ceros y valores nulos
    df['Precio_Promedio_Limpio'] = df['Precio promedio'].apply(clean_numeric_str)
    df['Volumen_Limpio'] = df['Volumen'].apply(clean_numeric_str)
    
    # Filtrar ceros
    df_clean = df[(df['Precio_Promedio_Limpio'] > 0) & (df['Volumen_Limpio'] > 0)].copy()
    print(f"Registros después de remover ceros y nulos: {df_clean.shape[0]}")
    
    # 3. Procesar unidades de comercialización (kilogramos)
    df_clean['weight_kg'] = df_clean['Unidad de comercializacion'].apply(parse_weight_kg)
    
    # Filtrar solo unidades con peso válido
    df_weighted = df_clean[df_clean['weight_kg'].notna()].copy()
    print(f"Registros después de filtrar por unidades basadas en peso (kg): {df_weighted.shape[0]}")
    
    # Calcular precio por kilogramo (Variable Objetivo)
    df_weighted['Precio_Promedio_Por_Kilo'] = df_weighted['Precio_Promedio_Limpio'] / df_weighted['weight_kg']
    
    # 4. Procesamiento temporal y estaciones (Hemisferio Sur)
    df_weighted['Fecha'] = pd.to_datetime(df_weighted['Fecha'])
    df_weighted['Mes'] = df_weighted['Fecha'].dt.month
    df_weighted['Dia_Semana'] = df_weighted['Fecha'].dt.dayofweek
    df_weighted['Estacion'] = df_weighted['Mes'].apply(get_season_chile)
    
    # Mapeo de día de semana en español
    dias_map = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}
    df_weighted['Dia_Semana_Nombre'] = df_weighted['Dia_Semana'].map(dias_map)
    
    # 5. Crear directorio de destino y guardar dataset unificado
    output_dir = 'dataset/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    unified_path = os.path.join(output_dir, 'dataset_unificado.csv')
    df_weighted.to_csv(unified_path, index=False)
    print(f"Dataset unificado guardado en: {unified_path}")
    
    # 6. Partición Cronológica Estratégica
    # Entrenamiento: Todo el año 2025
    df_train = df_weighted[df_weighted['Fecha'].dt.year == 2025].copy()
    
    # Validación y Prueba: Se extraen del año 2026
    df_2026 = df_weighted[df_weighted['Fecha'].dt.year == 2026].copy()
    
    # Ordenar cronológicamente 2026 para dividirlo exactamente a la mitad por fecha
    df_2026_sorted = df_2026.sort_values(by='Fecha').reset_index(drop=True)
    mid_point = len(df_2026_sorted) // 2
    
    # Validación: Primera mitad de 2026
    df_val = df_2026_sorted.iloc[:mid_point].copy()
    # Prueba: Segunda mitad de 2026
    df_test = df_2026_sorted.iloc[mid_point:].copy()
    
    # Guardar los datasets particionados
    train_path = os.path.join(output_dir, 'dataset_entrenamiento.csv')
    val_path = os.path.join(output_dir, 'dataset_validacion.csv')
    test_path = os.path.join(output_dir, 'dataset_prueba.csv')
    
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)
    
    print(f"Dataset de entrenamiento (2025) guardado en: {train_path} ({df_train.shape[0]} filas)")
    print(f"Dataset de validación (Inicio 2026) guardado en: {val_path} ({df_val.shape[0]} filas)")
    print(f"Dataset de prueba (Resto 2026) guardado en: {test_path} ({df_test.shape[0]} filas)")
    
    # 7. Imprimir validación de proporciones de división y rangos de fechas
    print("\n--- Validación de la División Cronológica ---")
    total_weighted = len(df_weighted)
    print(f"Entrenamiento (2025): {df_train.shape[0]} filas ({df_train.shape[0] / total_weighted * 100:.2f}%)")
    print(f"   Rango fechas: {df_train['Fecha'].min().strftime('%Y-%m-%d')} a {df_train['Fecha'].max().strftime('%Y-%m-%d')}")
    print(f"Validación (1a mitad 2026): {df_val.shape[0]} filas ({df_val.shape[0] / total_weighted * 100:.2f}%)")
    print(f"   Rango fechas: {df_val['Fecha'].min().strftime('%Y-%m-%d')} a {df_val['Fecha'].max().strftime('%Y-%m-%d')}")
    print(f"Prueba (2a mitad 2026): {df_test.shape[0]} filas ({df_test.shape[0] / total_weighted * 100:.2f}%)")
    print(f"   Rango fechas: {df_test['Fecha'].min().strftime('%Y-%m-%d')} a {df_test['Fecha'].max().strftime('%Y-%m-%d')}")
    
    print("\nProceso finalizado con éxito.")

if __name__ == '__main__':
    main()
