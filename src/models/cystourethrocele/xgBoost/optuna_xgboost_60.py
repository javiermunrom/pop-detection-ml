#!/usr/bin/env python
# -*- coding: utf-8 -*-

import optuna
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

#######################
# 1. Cargar Datos
#######################

BASE_DIR = Path.cwd()

def load_train_data():
    """
    Carga X_train, y_train, groups_train desde un pickle
    que habrás creado previamente en tu notebook.
    El pickle debe contener una tupla (X_train, y_train, groups_train).
    """
    path = BASE_DIR / "data_storage" / "train_test_splits" / "cystourethrocele" / "train_data_60.pkl"
    print("Cargando datos de:", path)
    X_train, y_train, groups_train = joblib.load(path)

    print("Datos cargados de 'train_data_60.pkl'")
    print("Shapes -> X_train:", X_train.shape, "y_train:", y_train.shape)
    return X_train, y_train, groups_train

#######################
# 2. Definir la validación
#######################

sgkf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

#######################
# 3. Función objetivo (Optuna)
#######################

def objective(trial):
    """
    Función que Optuna llama en cada trial (iteración) para evaluar
    un conjunto de hiperparámetros de XGBClassifier.
    """

    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 2, 20)
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    gamma = trial.suggest_float("gamma", 0.0, 5.0)

    clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        random_state=42,
        eval_metric="auc", 
        n_jobs=-1
    )

    cv_probs = cross_val_predict(
        clf,
        X_train_global,
        y_train_global,
        cv=sgkf,
        groups=groups_train_global,
        method='predict_proba',
        n_jobs=-1
    )

    auc_value = roc_auc_score(y_train_global, cv_probs[:, 1])

    return auc_value

#######################
# 4. Programa principal
#######################

def main():
    global X_train_global, y_train_global, groups_train_global

    X_train_global, y_train_global, groups_train_global = load_train_data()

    study = optuna.create_study(direction="maximize")
    print("Iniciando optimización de hiperparámetros (XGBoost) con cross_val_predict...")

    study.optimize(objective, n_trials=20)

    print("\n===== OPTIMIZACIÓN FINALIZADA =====")
    print(f"Mejor valor de la métrica (AUC): {study.best_value:.4f}")
    print(f"Mejores hiperparámetros: {study.best_params}")

    best_params = study.best_params
    best_model = XGBClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        subsample=best_params["subsample"],
        colsample_bytree=best_params["colsample_bytree"],
        gamma=best_params["gamma"],
        random_state=42,
        eval_metric="auc",
        n_jobs=-1
    )

    best_model.fit(X_train_global, y_train_global)

    print("Entrenamiento completo del modelo final con los mejores hiperparámetros (XGBClassifier).")

    # Guardar el modelo entrenado
    joblib.dump(best_model, BASE_DIR / "data_storage" / "models" / "cystourethrocele" / "xgboost" / "best_model_60.pkl")

    print("Modelo guardado en 'best_model_60.pkl'.")


if __name__ == "__main__":
    main()
