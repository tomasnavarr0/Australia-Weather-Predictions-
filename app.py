from dash import Dash, dcc, html, callback, Output, Input, dash_table, State
import pandas as pd
import numpy as np
import joblib

from funciones import ScalerTransformer, CleanAndTransformation

df = pd.read_csv("weatherAUS.csv", encoding="UTF-8")

# Filtra el DataFrame para incluir solo las ciudades especificadas
selected_cities = ['Sydney', 'SydneyAirport', 'Canberra', 'Melbourne', 'MelbourneAirport']
df = df[df['Location'].isin(selected_cities)]

pipeline_clasification = joblib.load('regresion_logistica_tp.joblib')
df_transformed_clasification = pipeline_clasification['Clean and Transformation'].transform(df)
df_transformed_clasification = pipeline_clasification['Standard Scaler'].transform(df_transformed_clasification)

# Obtén las predicciones y conviértelas a cadenas
predictions_clasification = pipeline_clasification['Model'].predict(df_transformed_clasification)

predictions_clasification = np.where(predictions_clasification == 0, 'No', 'Si')

# Cargar el modelo y realizar transformaciones en los datos
pipeline = joblib.load('redes_regresion_tp.joblib')
df_transformed = pipeline['Clean and Transformation'].transform(df)
df_transformed = pipeline['Standard Scaler'].transform(df_transformed)

# Obtener las predicciones y desescalarlas
predictions = pipeline['Model'].predict(df_transformed)

# Desescalar las predicciones
predictions_original = pipeline['Standard Scaler'].inverse_transform(df_transformed)
predictions_original = predictions_original.iloc[:, 0].values

# Agregar una nueva columna "Prediction" al DataFrame original con las predicciones
df_cp = df.copy().head(len(predictions_clasification))

df_cp['Lluvia_Mañana'] = predictions_clasification
df_cp['Cantidad_LLuvia_Mañana'] = predictions_original

# Obtener el rango de fechas disponible en el DataFrame filtrado
date_range = pd.to_datetime(df_cp['Date'])
min_date = date_range.min().date()
max_date = date_range.max().date()

# Lista de columnas a excluir
columns_to_exclude = ['Date', 'Location', 'RainTomorrow', 'RainfallTomorrow', 'Cantidad_LLuvia_Mañana', 'Lluvia_Mañana']

# Lista de todas las columnas excepto las excluidas
all_columns_except_excluded = [col for col in df_cp.columns if col not in columns_to_exclude]

# Crear la aplicación Dash
app = Dash(__name__)

# Definir el diseño de la aplicación
app.layout = html.Div([
    html.H1("Predictor de lluvia", style={'text-align': 'center', 'font-size': '46px', 'margin-bottom': '0px'}),
    html.H2("Selecciona un día:", style={'margin-bottom': '2px'}),
    
    # Menú desplegable para seleccionar una fecha dentro de un rango específico
    dcc.DatePickerSingle(
        id='date-picker',
        display_format='DD-MM-YYYY',
        date=min_date,
        min_date_allowed=min_date,
        max_date_allowed=max_date,
    ),
    
    html.H2("Selecciona una ciudad:", style={'margin-bottom': '2px'}),
    
    # Menú desplegable para seleccionar una ciudad
    dcc.Dropdown(
        id='city-selector',
        options=[{'label': city, 'value': city} for city in selected_cities],
        value=selected_cities[2]
    ),
    
    # Título de la tabla
    html.H2(id='table-title', style={'margin-top': '10px', 'margin-bottom': '2px'}),
    
    # Tabla que muestra los valores correspondientes
    dash_table.DataTable(
        id='selected-date-table',
        columns=[{'name': col, 'id': col} for col in all_columns_except_excluded],
        style_table={'height': '100px', 'overflowY': 'auto'},
    ),
    
    # Título de las predicciones
    html.H2("Predicciones", style={'margin-top': '20px', 'margin-bottom': '2px'}),
    
    # Tabla para mostrar las nuevas columnas
    dash_table.DataTable(
        id='predictions-table',
        columns=[
            {'name': 'Lluvia_Mañana', 'id': 'Lluvia_Mañana'},
            {'name': 'Cantidad_LLuvia_Mañana', 'id': 'Cantidad_LLuvia_Mañana'}
        ],
        style_table={'height': '100px', 'overflowY': 'auto'},
    ),
])


# Devolución de llamada para actualizar las tablas con los valores correspondientes y los títulos
@app.callback(
    [Output('selected-date-table', 'data'),
     Output('table-title', 'children'),
     Output('predictions-table', 'data')],
    [Input('date-picker', 'date'),
     Input('city-selector', 'value')]
)
def update_tables(selected_date, selected_city):
    # Filtra el DataFrame para obtener las filas correspondientes a la fecha y ciudad seleccionadas
    selected_rows = df_cp[(df_cp['Date'] == selected_date) & (df_cp['Location'] == selected_city)]
    
    if not selected_rows.empty:
        # Filtra las columnas
        selected_row = selected_rows[all_columns_except_excluded]
        predictions_data = selected_rows[['Lluvia_Mañana', 'Cantidad_LLuvia_Mañana']].to_dict('records')
        
        table_data = selected_row.to_dict('records')
        title = f"Valores del día {selected_date} en la ciudad {selected_city}"
        return table_data, title, predictions_data
    else:
        return [], f"No hay datos disponibles para el día {selected_date} en la ciudad {selected_city}", []


if __name__ == '__main__':
    app.run_server(debug=True)
