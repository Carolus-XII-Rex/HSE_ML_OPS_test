import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split


class CatBoostModel:
    def __init__(
        self,
        iterations=5,
        learning_rate=0.1,
        depth=6,
        loss_function="MultiClass",
        verbose=0,
    ):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.loss_function = loss_function
        self.verbose = verbose
        self.model = None

    def fit(self, X, y):
        self.model = CatBoostClassifier(
            text_features=[0],
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            loss_function=self.loss_function,
            verbose=self.verbose,
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

    def save_model(self, filename="model/catboost_model"):
        self.model.save_model(filename, format="cbm")

    @staticmethod
    def load_model(filename="model/catboost_model"):
        loaded_model = CatBoostClassifier()
        loaded_model.load_model(filename)

        model_instance = CatBoostModel()
        model_instance.model = loaded_model
        return model_instance


if __name__ == "__main__":
    df_train = pd.read_parquet("data/train/default_train_0000.parquet")[
        ["text", "coarse_label"]
    ].rename(columns={"coarse_label": "label"})

    X_train, X_test, y_train, y_test = train_test_split(
        df_train[["text"]], df_train["label"], test_size=0.2, random_state=42
    )

    model = CatBoostModel()
    model.fit(X_train, y_train)
    model.save_model()

    val_acc = round(model.score(X_test, y_test), 2)
    print(f"Model trained (val_acc = {val_acc}) and saved in model/")
