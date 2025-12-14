from sklearn.decomposition import PCA
import joblib
import pandas as pd

class PCAEngine:
    def __init__(self, n_components=3):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=42)

    def fit(self, X):
        self.pca.fit(X)
        return self

    def transform(self, X):
        pcs = self.pca.transform(X)
        cols = [f"PC{i+1}" for i in range(pcs.shape[1])]
        return pd.DataFrame(pcs, columns=cols)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def save(self, path):
        joblib.dump(self.pca, path)

    def load(self, path):
        self.pca = joblib.load(path)
        self.n_components = self.pca.n_components_
        return self
