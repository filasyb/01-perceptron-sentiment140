import numpy as np
import yaml

def load_config(path="config/local.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config()


class Perceptron:

    def __init__(self, lr=None, epochs=None, random_seed=None):
        self.lr = lr if lr is not None else config["model"]["learning_rate"]
        self.epochs = epochs if epochs is not None else config["model"]["epochs"]
        self.random_seed = random_seed if random_seed is not None else config["model"]["random_seed"]

        np.random.seed(self.random_seed)
        self.w = None
        self.b = None

    def _unit_step(self, x):
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        """
        Entrena el perceptrón usando la regla clásica de Rosenblatt.
        X: matriz de features (numpy array)
        y: etiquetas (0 o 1)
        """
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0


        for epoch in range(self.epochs):
            for xi, yi in zip(X, y):
                linear_output = np.dot(xi, self.w) + self.b
                y_pred = self._unit_step(linear_output)

                update = self.lr * (yi - y_pred)

                self.w += update * xi
                self.b += update

        return self

    def predict(self, X):
        """
        Devuelve predicciones binarias (0/1)
        """
        linear_output = np.dot(X, self.w) + self.b
        return self._unit_step(linear_output)

    def decision_function(self, X):
        """
        Devuelve el valor crudo (antes del step).
        Útil para análisis y visualización.
        """
        return np.dot(X, self.w) + self.b
