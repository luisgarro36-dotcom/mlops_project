import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def calculo_nan(df):
    """Imprime columnas con más del 80% de nulos."""
    for c in df.columns:
        porc_nan = df[c].isna().sum() / len(df[c]) * 100
        if porc_nan > 80:
            print(f"{c}: {porc_nan:.2f}% de nulos")

def fun_fillna0(df, columns):
    """Reemplaza valores nulos por 0 en columnas numéricas."""
    for c in columns:
        df[c] = df[c].fillna(0)

def fun_conv(df, columns, tipo):
    """Convierte columnas al tipo de dato indicado."""
    for c in columns:
        df[c] = df[c].astype(tipo)

def limpieza(df):
    """Elimina variables con más del 80% de nulos."""
    calculo_nan(df)
    df = df.drop(columns=[
        'prm_saltotrdpj03m', 'prm_saltotrdpj12m',
        'dsv_diasatrrdpj12m', 'max_pctsalimpago12m',
        'prm_diasatrrdpj03m'
    ], errors='ignore')
    return df

def imputacion(df):
    """Imputa valores faltantes según reglas definidas."""
    cols_fillna0=['p_codmes', 'key_value', 'target', 'monto', 'ctd_prod_rccsf_m01',
       'prom_salvig_entprinc_pp_rccsf_03m', 'max_usotcrrstsf06m',
       'max_usotcrrstsf03m', 'prm_lintcribksf06m', 'lin_tcribksf03m',
       'lin_tcribksf06m', 'cre_salvig_pp_rccsf_m02', 'prm_usotcrrstsf06m',
       'prm_lintcribksf03m', 'prm_usotcrrstsf03m', 'ratuso_tcrrstsf_m13',
       'promctdprodrccsf3m', 'ratpct_saldopprcc_m13', 'promctdprodrccsf6m',
       'ctd_actrccsf_6m', 'ctd_flgact_rccsf_m01', 'sld_ep1ppeallsfm01',
       'prm_sldvigrstsf12m', 'sldtot_tcrsrcf', 'sldvig_tcrsrcf',
       'flg_lintcrripsaga', 'flg_svtcrsrcf', 'flg_sdtcrripsaga',
       'flg_sttcrsrcf', 'flg_svltcrsrcf', 'max_camptottlv06m',
       'min_camptottlv06m', 'frc_camptottlv06m', 'rec_camptottlv06m',
       'grp_camptottlv06m', 'ctd_camptottlv06m', 'prm_camptottlv06m',
       'max_campecstlv06m', 'min_campecstlv06m', 'frc_campecstlv06m',
       'rec_campecstlv06m', 'grp_campecstlv06m', 'ctd_campecstlv06m',
       'prm_campecstlv06m', 'prob_value', 'max_camptot06m', 'min_camptot06m',
       'frc_camptot06m', 'rec_camptot06m','ctd_camptot06m',
       'prm_camptot06m', 'max_campecs06m', 'min_campecs06m', 'frc_campecs06m',
       'rec_campecs06m', 'ctd_campecs06m', 'prm_campecs06m',
       'ing_brt', 'sexo',  'flg_notdeseaecs01m', 'flg_notdeseaecs02m',
       'flg_notdeseaecs03m', 'sgt_cem', 'ctd_proac', 'ctd_cttefeproac',
       'ctd_cttefetot', 'rng_pst',
       'flg_saltothip12m', 'flg_saltotppe12m', 'prm_pctsaltototrent12m',
       'prm_pctsaltotcaja12m', 'ant_saltot24m', 'ant_saltot12m',
       'min_difsaltottcr12m', 'num_incrsaldispefe06m', 'max_difent12m',
       'num_dismsalppecons06m', 'seg_un', 'beta_pctusotcr12m',
       'prm_pctusosaltottcr03m', 'dsv_saltotppe03m', 'prm_diasatrrdpn12m',
       'dsv_numentrdlintcr03m', 'rat_disefepnm01', 'prm_diasatrrdpn06m',
       'pct_usotcrm01', 'dsv_numentrdlintcr06m', 'beta_saltotppe12m',
       'prm_entrd03m', 'ctd_entrdm01', 'beta_saltotppe06m',
       'prm_diasatrrd03m']
    fun_fillna0(df, cols_fillna0)

    # Imputacion por mediana
    df.edad = df.edad.fillna(df.edad.median())

    # Imputacion criterio experto!
    df.ubigeo_buro = df.ubigeo_buro.fillna('Otros')
    df.grp_camptot06m = df.grp_camptot06m.fillna('Otros')
    df.grp_campecs06m = df.grp_campecs06m.fillna('Otros')
    df.region = df.region.fillna('Otros')
    df.grp_riesgociiu = df.grp_riesgociiu.fillna('grupo_0')

    df["seg_un"]  = pd.Series(np.where(df.seg_un.isin([0,3]),0,df.seg_un)) # 3 -> 0
    df["grp_riesgociiu"] = pd.Series(np.where(df.grp_riesgociiu.isin(['grupo_2','grupo_3','grupo_9','grupo_8','grupo_1']),'grupo_11',df.grp_riesgociiu))
    return df

def encodificacion(df):
    """Codifica variables categóricas a numéricas."""
    features_encoder = ['grp_camptottlv06m','grp_campecstlv06m','grp_camptot06m','grp_campecs06m','region', 'grp_riesgociiu', 'ubigeo_buro']
    for columna in features_encoder:
        print(columna)
        ds_clase = LabelEncoder()
        ds_clase.fit(df[columna]) # entrenando
        df[columna] = ds_clase.transform(df[columna]) # inferencia
    return df

def asignacion_tipo_datos(df):
    """Ajusta tipos de datos finales."""
    columns_int32 = ['rec_campecs06m', 'grp_campecs06m', 'ctd_campecs06m']
    columns_float64 = ['p_codmes', 'monto', 'prom_salvig_entprinc_pp_rccsf_03m', 'max_usotcrrstsf06m', 'max_usotcrrstsf03m', 'prm_lintcribksf06m', 'lin_tcribksf03m', 'lin_tcribksf06m', 'cre_salvig_pp_rccsf_m02', 'prm_usotcrrstsf06m', 'prm_lintcribksf03m', 'prm_usotcrrstsf03m', 'ratuso_tcrrstsf_m13', 'ratpct_saldopprcc_m13', 'sld_ep1ppeallsfm01', 'prm_sldvigrstsf12m', 'sldtot_tcrsrcf', 'sldvig_tcrsrcf', 'prm_camptottlv06m', 'ctd_campecstlv06m', 'prm_campecstlv06m', 'prob_value', 'max_camptot06m', 'min_camptot06m', 'frc_camptot06m', 'rec_camptot06m', 'ctd_camptot06m', 'prm_camptot06m', 'max_campecs06m', 'min_campecs06m', 'frc_campecs06m', 'prm_campecs06m', 'ing_brt', 'sexo', 'flg_notdeseaecs01m', 'flg_notdeseaecs02m', 'flg_notdeseaecs03m', 'sgt_cem', 'ctd_proac', 'ctd_cttefeproac', 'ctd_cttefetot', 'rng_pst', 'edad', 'flg_saltothip12m', 'flg_saltotppe12m', 'prm_pctsaltototrent12m', 'prm_pctsaltotcaja12m', 'ant_saltot24m', 'ant_saltot12m', 'min_difsaltottcr12m', 'num_incrsaldispefe06m', 'max_difent12m', 'num_dismsalppecons06m', 'seg_un', 'beta_pctusotcr12m', 'prm_pctusosaltottcr03m', 'dsv_saltotppe03m', 'prm_diasatrrdpn12m', 'dsv_numentrdlintcr03m', 'rat_disefepnm01', 'prm_diasatrrdpn06m', 'pct_usotcrm01', 'dsv_numentrdlintcr06m', 'beta_saltotppe12m', 'prm_entrd03m', 'ctd_entrdm01', 'beta_saltotppe06m', 'prm_diasatrrd03m']
    cols_object=['key_value']
    cols_int64=['target', 'ctd_prod_rccsf_m01', 'promctdprodrccsf3m', 'promctdprodrccsf6m', 'ctd_actrccsf_6m', 'ctd_flgact_rccsf_m01', 'flg_lintcrripsaga', 'flg_svtcrsrcf', 'flg_sdtcrripsaga', 'flg_sttcrsrcf', 'flg_svltcrsrcf', 'max_camptottlv06m', 'min_camptottlv06m', 'frc_camptottlv06m', 'rec_camptottlv06m', 'grp_camptottlv06m', 'ctd_camptottlv06m', 'max_campecstlv06m', 'min_campecstlv06m', 'frc_campecstlv06m', 'rec_campecstlv06m', 'grp_campecstlv06m', 'grp_camptot06m', 'region', 'ubigeo_buro', 'grp_riesgociiu']
    fun_conv(df, columns_int32, 'int32')
    fun_conv(df, columns_float64, 'float64')
    fun_conv(df, cols_object, 'object')
    fun_conv(df, cols_int64, 'int64')
    return df


if __name__ == "__main__":
    print(" Iniciando preprocesamiento...")

    # 1. Lectura del dataset crudo
    df = pd.read_csv('data/raw/dataset.csv')
    print(" Datos cargados correctamente:", df.shape)

    # 2. Limpieza
    df = limpieza(df)

    # 3. Imputación
    df = imputacion(df)

    # 4. Codificación
    df = encodificacion(df)

    # 5. Asignacion de tipo de datos
    df = asignacion_tipo_datos(df)

    # 6. Data Split: División en train, test y val
    df_val = df[df['p_codmes'] == 201912.0]
    df_main = df[df['p_codmes'] != 201912.0]
    df_train, df_test = train_test_split(df_main, test_size=0.3, random_state=123)

    #Guardado de archivos procesados
    df_train.to_csv('data/processed/train.csv', index=False)
    df_test.to_csv('data/processed/test.csv', index=False)
    df_val.to_csv('data/processed/val.csv', index=False)

    print(" Archivos guardados en data/processed/")
    print("Train:", df_train.shape, "| Test:", df_test.shape, "| Val:", df_val.shape)