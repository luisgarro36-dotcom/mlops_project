import pandas as pd
import numpy as np
from scipy.stats import entropy
import mlflow

# =========================================
# Función para calcular Population Stability Index (PSI)
# =========================================
def calculate_psi(expected, actual, buckets=10):
    def scale_range(input, min_val, max_val):
        input_std = (input - input.min()) / (input.max() - input.min())
        input_scaled = input_std * (max_val - min_val) + min_val
        return input_scaled

    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
    breakpoints = np.percentile(expected, breakpoints)
    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    psi_value = np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))
    return psi_value

# =========================================
# Cargar datasets procesados
# =========================================
train = pd.read_csv('data/processed/train.csv')
test = pd.read_csv('data/processed/test.csv')
val = pd.read_csv('data/processed/val.csv')

# =========================================
# Calcular PSI para cada conjunto
# =========================================
mlflow.set_experiment("monitoreo_data_drift")

with mlflow.start_run(run_name="psi_drift_evaluation"):
    psi_results = {}
    for col in train.columns[3:]:  # omitimos columnas ID o target
        psi_test = calculate_psi(train[col], test[col])
        psi_val = calculate_psi(train[col], val[col])
        psi_results[col] = {'PSI_test': psi_test, 'PSI_val': psi_val}

    # Convertir resultados a DataFrame
    psi_df = pd.DataFrame(psi_results).T
    psi_df.to_csv('data/processed/psi_results.csv')
    mlflow.log_artifact('data/processed/psi_results.csv')

    # Promedio general
    psi_mean_test = psi_df['PSI_test'].mean()
    psi_mean_val = psi_df['PSI_val'].mean()

    print(f"✅ PSI promedio Test: {psi_mean_test:.4f}")
    print(f"✅ PSI promedio Val: {psi_mean_val:.4f}")

    mlflow.log_metric("psi_mean_test", psi_mean_test)
    mlflow.log_metric("psi_mean_val", psi_mean_val)
