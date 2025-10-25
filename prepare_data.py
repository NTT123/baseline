import os
import pandas as pd


def export_to_file(data, file_path, eos=b"\x00"):
    """export to binary file using utf-8 encoding"""
    with open(file_path, "wb") as f:
        for row in data.itertuples(index=False):
            f.write(eos)
            f.write(row.text.encode("utf-8"))


if __name__ == "__main__":
    os.mkdir("data")
    dfs = []
    for i in range(5):
        file_name = f"train-{i:05d}-of-00234.parquet"
        URL = f"https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/resolve/main/fineweb-edu-dedup/{file_name}"
        dfs.append(pd.read_parquet(URL, columns=["text"]))

    df = pd.concat(dfs)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    L = df.shape[0] * 9 // 10
    train_df = df.iloc[:L]
    test_df = df.iloc[L:]

    export_to_file(train_df, "./data/train.bin")
    export_to_file(test_df, "./data/test.bin")
    del dfs, df, L, train_df, test_df
