from collections.abc import Sequence
import pandas as pd


def convert_columns_type(
    df: pd.DataFrame,
    columns: str | Sequence[str],
    dtype: str | type,
) -> pd.DataFrame:
    """
    Convertit une ou plusieurs colonnes vers un type donné.

    La conversion modifie le DataFrame reçu et le retourne.

    Raises:
        TypeError: Si df n'est pas un DataFrame.
        ValueError: Si aucune colonne n'est fournie ou si une conversion échoue.
        KeyError: Si une colonne demandée est absente.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df doit être un DataFrame pandas")

    if isinstance(columns, str):
        columns = [columns]
    else:
        columns = list(columns)

    if not columns:
        raise ValueError("Au moins une colonne doit être fournie")

    missing_columns = [column for column in columns if column not in df.columns]

    if missing_columns:
        raise KeyError(f"Colonnes absentes : {missing_columns}")

    target_dtype = "string" if dtype in {"str", "string"} else dtype

    for column in columns:
        try:
            if dtype == "datetime":
                df[column] = pd.to_datetime(df[column], errors="raise")
            else:
                df[column] = df[column].astype(target_dtype)

        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Impossible de convertir la colonne '{column}' "
                f"du type '{df[column].dtype}' vers '{dtype}'"
            ) from error

    return df