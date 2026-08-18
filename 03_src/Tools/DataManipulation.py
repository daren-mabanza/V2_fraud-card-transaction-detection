import pandas as pd

def convert_columns_type(df, columns, dtype, inplace=True):
    """
    Convertit le type de plusieurs colonnes d'un DataFrame
    
    Parameters
    ----------
    df : DataFrame
        Le DataFrame à modifier
    columns : list or str
        Liste des noms de colonnes à convertir (ou nom unique en str)
    dtype : str or type
        Type de données cible : 'str', 'int', 'float', 'category', 'datetime', etc.
    inplace : bool, default=True
        Si True, modifie le DataFrame original. Si False, retourne une copie
    
    Returns
    -------
    DataFrame or None
        DataFrame modifié (si inplace=False)
    
    Examples
    --------
    >>> convert_columns_type(data_housing, ['postal_code', 'city'], 'str')
    >>> convert_columns_type(data_housing, ['price', 'size'], 'float')
    >>> convert_columns_type(data_housing, 'property_type', 'category')
    """
    
    # Gère le cas d'une seule colonne (str au lieu de list)
    if isinstance(columns, str):
        columns = [columns]
    
    # DataFrame à modifier
    df_work = df if inplace else df.copy()
    
    # Vérifie que les colonnes existent
    missing_cols = [col for col in columns if col not in df_work.columns]
    if missing_cols:
        print(f"Colonnes inexistantes ignorées : {missing_cols}")
        columns = [col for col in columns if col in df_work.columns]
    
    if not columns:
        print("Aucune colonne valide à convertir")
        return None if not inplace else None
    
    # Mapping des types courants
    type_mapping = {
        'str': str,
        'string': str,
        'int': int,
        'integer': int,
        'float': float,
        'bool': bool,
        'boolean': bool,
        'category': 'category',
        'datetime': 'datetime64[ns]'
    }
    
    # Résout le type
    target_dtype = type_mapping.get(dtype, dtype)
    
    # Conversion avec gestion d'erreurs
    errors = []
    success = []
    for col in columns:
        try:
            # Type avant
            dtype_before = df_work[col].dtype
            
            # Conversion spéciale pour strings avec codes postaux
            if target_dtype == str:
                df_work[col] = df_work[col].astype('string')  # ← Avec guillemets
                # Si c'est un code postal, force 5 chiffres
                if 'postal' in col.lower() or 'cp' in col.lower() or 'code' in col.lower():
                    df_work[col] = df_work[col].str.replace('nan', '').str.zfill(5)
                    df_work[col] = df_work[col].replace('00000', None)
            
            # Conversion category
            elif target_dtype == 'category':
                df_work[col] = df_work[col].astype('category')
            
            # Conversion datetime
            elif target_dtype == 'datetime64[ns]':
                df_work[col] = pd.to_datetime(df_work[col], errors='coerce')
            
            # Autres conversions
            else:
                df_work[col] = df_work[col].astype(target_dtype)
            
            dtype_after = df_work[col].dtype
            success.append(col)
        
            
        except Exception as e:
            errors.append((col, str(e)))
            print(f"{col}: ERREUR - {str(e)}")

    
    return None if inplace else df_work