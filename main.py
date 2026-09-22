import os
import random
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV

# Импортируем вашу функцию предобработки
try:
    from preprocessing_with_val import process_v1
except ImportError:
    # Если функция лежит в другом файле или том же модуле, убедитесь в правильности импорта
    pass


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def predict_ensemble(models: list, X) -> np.ndarray:
    preds = np.array([model.predict(X) for model in models])
    return np.mean(preds, axis=0)


def main():
    random_state = 42
    seed_everything(random_state)

    # 1. Загрузка данных
    train_path = "datasets/train.csv"
    test_path = "datasets/test.csv"
    sample_sub_path = "datasets/sample_submission.csv"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            "Не найдена папка 'datasets' или файлы train.csv / test.csv!"
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # 2. Подготовка данных с помощью process_v1
    X_train, Y_train_log, X_test = process_v1(train, test, target_col="SalePrice")

    # 3. Настройка кросс-валидации и поиска
    search_cv = KFold(n_splits=3, shuffle=True, random_state=random_state)
    N_ITER = 10

    cat_param_dist = {
        "n_estimators": randint(300, 1200),
        "max_depth": randint(3, 7),
        "learning_rate": uniform(0.01, 0.14),
        "l2_leaf_reg": uniform(1, 8),
    }

    # 4. Инициализация и запуск RandomizedSearchCV для CatBoost
    cat_search = RandomizedSearchCV(
        CatBoostRegressor(random_state=random_state, verbose=0),
        cat_param_dist,
        n_iter=N_ITER,
        scoring="neg_mean_squared_error",
        cv=search_cv,
        random_state=random_state,
        n_jobs=-1,
    )

    cat_search.fit(X_train, Y_train_log)

    # 5. Преобразуем результаты поиска в DataFrame и сортируем по метрике (рангу)
    results_df = pd.DataFrame(cat_search.cv_results_)
    top5_results = results_df.sort_values("rank_test_score").head(5)

    # 6. Создаем и обучаем 5 лучших моделей на ВСЕХ обучающих данных (X_train, Y_train_log)
    top5_models = []

    print("--- CV RMSE для отдельных лучших моделей CatBoost ---")
    for i, (idx, row) in enumerate(top5_results.iterrows(), start=1):
        params = row["params"]

        # Рассчитываем RMSE из mean_test_score
        model_cv_rmse = np.sqrt(-row["mean_test_score"])
        print(f"Модель {i} CV RMSE: {model_cv_rmse:.16f}")

        # Инициализируем и обучаем модель
        model = CatBoostRegressor(
            **params, random_state=random_state + len(top5_models), verbose=0
        )
        model.fit(X_train, Y_train_log)
        top5_models.append(model)

    # 7. Расчет CV RMSE для всего ансамбля
    ensemble_cv_scores = []

    for train_idx, val_idx in search_cv.split(X_train, Y_train_log):
        if isinstance(X_train, pd.DataFrame):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = Y_train_log.iloc[train_idx], Y_train_log.iloc[val_idx]
        else:
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = Y_train_log[train_idx], Y_train_log[val_idx]

        fold_models = []
        for m in top5_models:
            params = m.get_params()
            fold_model = CatBoostRegressor(**params)
            fold_model.fit(X_tr, y_tr)
            fold_models.append(fold_model)

        fold_preds = predict_ensemble(fold_models, X_val)
        mse = np.mean((y_val - fold_preds) ** 2)
        ensemble_cv_scores.append(mse)

    ensemble_cv_rmse = np.sqrt(np.mean(ensemble_cv_scores))

    print("\n--- Итоговая метрика ансамбля ---")
    print(f"CatBoost Top-5 Ensemble CV RMSE: {ensemble_cv_rmse:.16f}")

    # 8. Предсказание на тестовой выборке X_test и выгрузка файла
    test_preds_log_cat_ensemble = predict_ensemble(top5_models, X_test)
    test_preds_cat_ensemble = np.expm1(test_preds_log_cat_ensemble)

    sub = pd.read_csv(sample_sub_path)
    sub["SalePrice"] = test_preds_cat_ensemble
    output_filename = "submission_catboost_ensemble.csv"
    sub.to_csv(output_filename, index=False)

    print(f"\nФайл '{output_filename}' успешно создан!")
    print(sub.head())


if __name__ == "__main__":
    main()