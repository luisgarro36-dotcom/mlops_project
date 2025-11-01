import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
# 💡 Importamos joblib para guardar el modelo físicamente para DVC
import joblib 


# ===============================
# 1️⃣ Cargar los datasets procesados
# ===============================
df_train = pd.read_csv('data/processed/train.csv')
df_test = pd.read_csv('data/processed/test.csv')

# ===============================
# 2️⃣ Separar variables predictoras y objetivo
# ===============================
# Las variables X e Y dependen de la estructura de tu CSV
x_train = df_train.iloc[:, 3:]
y_train = df_train.iloc[:, 2]

x_val = df_test.iloc[:, 3:]
y_val = df_test.iloc[:, 2]

# ===============================
# 3️⃣ Configurar MLflow y Entrenar
# ===============================
mlflow.set_experiment("ventas_xgboost_experiment")

with mlflow.start_run(run_name="modelo_xgboost_final"):
    # Definición y entrenamiento del modelo
    model = xgb.XGBClassifier(
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    model.fit(x_train, y_train)

    # 🛑 CÁLCULO DE MÉTRICAS (RESOLVIENDO EL NAMEERROR)
    # Estas líneas deben estar AQUÍ para definir 'accuracy'
    y_pred = model.predict(x_val)
    accuracy = (y_pred == y_val).mean()

    print(f"✅ Exactitud del modelo: {accuracy:.4f}")

    # Registro de parámetros y métricas en MLflow
    mlflow.log_param("model_type", "XGBClassifier")
    mlflow.log_metric("accuracy", accuracy) 
    mlflow.xgboost.log_model(model, "xgboost_model")

    # 💾 GUARDA EL MODELO FÍSICAMENTE PARA DVC
    # Esto resuelve el error "output 'models\model.pkl' does not exist"
    joblib.dump(model, 'models/model.pkl') 

print("✅ Modelo registrado correctamente en MLflow y guardado para DVC.")