import pandas as pd
from sklearn.metrics import accuracy_score
from train import CatBoostModel

if __name__ == "__main__":
    df_test = pd.read_parquet("data/test/default_test_0000.parquet")[
        ["text", "coarse_label"]
    ].rename(columns={"coarse_label": "label"})

    loaded_model = CatBoostModel.load_model()
    predictions = loaded_model.predict(df_test[["text"]])

    test_acc = accuracy_score(df_test[["label"]], predictions)
    print(f"test_acc = {test_acc}")
