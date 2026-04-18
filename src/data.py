from __future__ import annotations

import pandas as pd


def convert_windows1252_to_utf8(src_path: str, dest_path: str) -> None:
    """
    Equivalent to notebook's encoding conversion step:
      with open("spam.csv", "r", encoding="windows-1252") as src: ...
      with open("spam_utf8.csv", "w", encoding="utf-8") as dest: ...
    """
    with open(src_path, "r", encoding="windows-1252") as src:
        data = src.read()
    with open(dest_path, "w", encoding="utf-8") as dest:
        dest.write(data)


def load_and_clean_dataframe(csv_path_utf8: str) -> pd.DataFrame:
    """
    Equivalent cleaning steps:
      - read spam_utf8.csv
      - drop Unnamed: 2/3/4
      - rename v1->target, v2->text
      - LabelEncode target
      - drop duplicates
    """
    df = pd.read_csv(csv_path_utf8)

    # Removing extra unnecessary columns (if present)
    drop_cols = [
        c for c in ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"] if c in df.columns
    ]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)

    # Renaming columns to match semantics
    df.rename(columns={"v1": "target", "v2": "text"}, inplace=True)

    if "target" not in df.columns or "text" not in df.columns:
        raise ValueError(
            f"Expected columns after rename: target,text. Found: {list(df.columns)}"
        )

    # Convert textual labels to numeric (ham/spam -> 0/1)
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    df["target"] = encoder.fit_transform(df["target"])

    # Remove duplicates
    df = df.drop_duplicates(keep="first").reset_index(drop=True)

    return df
