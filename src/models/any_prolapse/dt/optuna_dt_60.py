#!/usr/bin/env python
# -*- coding: utf-8 -*-

import optuna
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

###############################
# 1. Cargar Datos
###############################


BASE_DIR = Path.cwd()

def load_train_data():
    """
    Carga X_train, y_train, groups_train desde un pickle
    que habrás creado previamente en tu notebook.
    El pickle debe contener una tupla (X_train, y_train, groups_train).
    """

    path = BASE_DIR / "data_storage" / "train_test_splits" / "any_prolapse" / "train_data_60.pkl"
    X_train, y_train, groups_train = joblib.load(path)
    print("Datos cargados de 'train_data_60.pkl'")
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
    Función que Optuna llama en cada trial (iteración) para evaluar
    un conjunto de hiperparámetros de DecisionTreeClassifier.
    """

    max_depth = trial.suggest_int("max_depth", 2, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        random_state=42
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

###############################
# 4. Programa principal
###############################

def main():
    global X_train_global, y_train_global, groups_train_global

    X_train_global, y_train_global, groups_train_global = load_train_data()

    study = optuna.create_study(direction="maximize")
    print("Iniciando optimización de hiperparámetros (DecisionTreeClassifier) ...")

    study.optimize(objective, n_trials=20)

    print("\n===== OPTIMIZACIÓN FINALIZADA =====")
    print(f"Mejor valor de la métrica (AUC): {study.best_value:.4f}")
    print(f"Mejores hiperparámetros: {study.best_params}")

    best_params = study.best_params
    best_model = DecisionTreeClassifier(
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        criterion=best_params["criterion"],
        random_state=42
    )

    best_model.fit(X_train_global, y_train_global)
    print("Entrenamiento completo del modelo final con los mejores hiperparámetros (DecisionTreeClassifier).")

    path = BASE_DIR / "data_storage" / "models" / "any_prolapse" / "dt" / "best_model_60.pkl"
    joblib.dump(best_model, path)
    print("Modelo guardado en 'best_model_60.pkl'.")


if __name__ == "__main__":
    main()
