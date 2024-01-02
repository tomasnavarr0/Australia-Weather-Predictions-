from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.preprocessing import StandardScaler


def create_dummy_variables(df, column_name):
    """
    Genera variables dummy para los valores únicos en una columna categórica del DataFrame.

    Args:
    - df: DataFrame de pandas.
    - column_name: Nombre de la columna para la cual se generarán las variables dummy.

    Returns:
    - DataFrame modificado con las variables dummy.
    """
    dummies = pd.get_dummies(df[column_name], prefix=column_name, drop_first=True)

    df = pd.concat([df, dummies], axis=1)

    df.drop(column_name, axis=1, inplace=True)

    return df

class CleanAndTransformation(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # No es necesario hacer nada en fit para un transformador

    def transform(self, X):
        ciudades = ["Melbourne", "MelbourneAirport", "Canberra", "Sydney", "SydneyAirport"]
        X["Date"] = pd.to_datetime(X["Date"])
        X['Month'] = X["Date"].dt.month
        X['Season'] = X["Month"].map({1: 'Summer', 2: 'Summer', 3: 'Autumn', 4: 'Autumn', 5: 'Autumn', 6: 'Winter', 7: 'Winter', 8: 'Winter', 9: 'Spring', 10: 'Spring', 11: 'Spring', 12: 'Summer'})
        X = X.query("Location.isin(@ciudades)")
        # Supongo que create_dummy_variables es una función que crea variables dummy
        X = create_dummy_variables(X, 'Location')
        X = create_dummy_variables(X, 'Season')
        X = create_dummy_variables(X, 'WindGustDir')
        X = create_dummy_variables(X, 'WindDir3pm')
        X = create_dummy_variables(X, 'WindDir9am')
        X['RainToday'] = X['RainToday'].replace({'Yes': 1, 'No': 0})
        X= X.drop(columns=["RainTomorrow", "RainfallTomorrow", "Unnamed: 0", "Date"])
        X = X.dropna()
        return X

class ScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_scale):
        self.columns_to_scale = columns_to_scale
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns_to_scale])
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        X_transformed[self.columns_to_scale] = self.scaler.transform(X_transformed[self.columns_to_scale])
        return X_transformed

    def inverse_transform(self, X):
        X_inverse = X.copy()
        X_inverse[self.columns_to_scale] = self.scaler.inverse_transform(X_inverse[self.columns_to_scale])
        return X_inverse
