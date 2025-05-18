#!/usr/bin/env python
# -*- coding: utf-8 -*-

import optuna
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

###############################
# 1. Cargar Datos
###############################

BASE_DIR = Path.cwd()

def load_train_data():
    """
    Carga X_train, y_train, groups_train desde un pickle
    que hayas creado previamente en tu notebook.
    El pickle debe contener una tupla (X_train, y_train, groups_train).
    """
    path = BASE_DIR / "data_storage" / "train_test_splits" / "rectocele" / "train_data_60.pkl"
    print("Cargando datos de:", path)
    X_train, y_train, groups_train = joblib.load(path)

    print("Datos cargados de 'train_data.pkl'")
    print("Shapes -> X_train:", X_train.shape, "y_train:", y_train.shape)
    return X_train, y_train, groups_train

###############################
# 2. Definir la validación
###############################

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

###############################
# 3. Función objetivo (Optuna)
###############################

def objective(trial):
    """
    Función que Optuna llama en cada trial para evaluar
    un conjunto de hiperparámetros de LogisticRegression (con escalado).
    Se optimiza la AUC calculada a partir de las probabilidades out-of-fold.
    """
    # ?? C tiene que ser pequeño
    C = trial.suggest_float("C", 1e-3, 1e3, log=True)
    
    # Pipeline que escala los datos y aplica LogisticRegression
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42))
    ])
    
    cv_probs = cross_val_predict(
        pipeline,
        X_train_global,
        y_train_global,
        cv=sgkf,
        groups=groups_train_global,
        method="predict_proba",
        n_jobs=-1
    )
    
    auc_value = roc_auc_score(y_train_global, cv_probs[:, 1])
    return auc_value

###############################
# 4. Programa principal
###############################

def main():
    global X_train_global, y_train_global, groups_train_global
    X_train_global, y_train_global, groups_train_global = load_train_data()
    
    study = optuna.create_study(direction="maximize")
    print("Iniciando optimización de hiperparámetros (LogisticRegression con escalado) con out-of-fold predict_proba...")
    study.optimize(objective, n_trials=20)
    
    print("\n===== OPTIMIZACIÓN FINALIZADA =====")
    print(f"Mejor valor de la métrica (AUC): {study.best_value:.4f}")
    print(f"Mejores hiperparámetros: {study.best_params}")
    
    best_params = study.best_params
    best_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=best_params["C"],
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=42
        ))
    ])
    
    best_pipeline.fit(X_train_global, y_train_global)
    print("Entrenamiento completo del modelo final con los mejores hiperparámetros (LogisticRegression con escalado).")
    
    # Guardar el modelo entrenado
    path = BASE_DIR /"data_storage" / "models" / "rectocele" / "lr"
    joblib.dump(best_pipeline, path / "best_model.pkl")
    print("Modelo guardado en 'best_model.pkl'.")

if __name__ == "__main__":
    main()
