from sklearn.preprocessing import StandardScaler
import joblib

def scale_features(df, feature_cols, save_path=None):
    scaler = StandardScaler()
    X = df[feature_cols].values
    X_scaled = scaler.fit_transform(X)
    if save_path:
        joblib.dump(scaler, save_path)
    return X_scaled, scaler

def transform_with_scaler(df, feature_cols, scaler):
    X = df[feature_cols].values
    return scaler.transform(X)
