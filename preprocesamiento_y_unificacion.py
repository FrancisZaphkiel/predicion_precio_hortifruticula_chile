import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

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
    print("Iniciando pipeline de unificación y preprocesamiento...")
    
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
    
    # 6. Partición estratificada por estaciones (70% train, 15% val, 15% test)
    # Primera partición: Separamos el 15% para Test
    df_train_val, df_test = train_test_split(
        df_weighted,
        test_size=0.15,
        random_state=42,
        stratify=df_weighted['Estacion']
    )
    
    # Segunda partición: Del 85% restante, separamos 15% para Validación (15/85 ≈ 0.17647)
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=(15/85),
        random_state=42,
        stratify=df_train_val['Estacion']
    )
    
    # Guardar los datasets particionados
    train_path = os.path.join(output_dir, 'dataset_entrenamiento.csv')
    val_path = os.path.join(output_dir, 'dataset_validacion.csv')
    test_path = os.path.join(output_dir, 'dataset_prueba.csv')
    
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)
    
    print(f"Dataset de entrenamiento guardado en: {train_path} ({df_train.shape[0]} filas)")
    print(f"Dataset de validación guardado en: {val_path} ({df_val.shape[0]} filas)")
    print(f"Dataset de prueba guardado en: {test_path} ({df_test.shape[0]} filas)")
    
    # 7. Imprimir validación de proporciones de estratificación
    print("\n--- Validación de Proporciones Estratificadas (Estacion) ---")
    print(f"{'Estación':<12} | {'Unificado':<10} | {'Entrenamiento':<13} | {'Validación':<10} | {'Prueba':<10}")
    print("-" * 65)
    
    seasons = ['Verano', 'Otoño', 'Invierno', 'Primavera']
    for s in seasons:
        prop_uni = (df_weighted['Estacion'] == s).mean() * 100
        prop_train = (df_train['Estacion'] == s).mean() * 100
        prop_val = (df_val['Estacion'] == s).mean() * 100
        prop_test = (df_test['Estacion'] == s).mean() * 100
        print(f"{s:<12} | {prop_uni:>8.2f}% | {prop_train:>11.2f}% | {prop_val:>8.2f}% | {prop_test:>8.2f}%")
        
    print("\nProceso finalizado con éxito.")

if __name__ == '__main__':
    main()
