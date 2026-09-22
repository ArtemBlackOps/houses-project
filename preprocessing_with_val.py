import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import (
    LogisticRegression,
    Perceptron,
    SGDClassifier,
)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# machine learning
from sklearn.base import clone

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, QuantileTransformer, OrdinalEncoder 

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, KFold, RepeatedKFold, RandomizedSearchCV

from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostRegressor, CatBoostClassifier, Pool

from pathlib import Path
from scipy.stats import skew

from sklearn.metrics import (
    log_loss,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,

    average_precision_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    mean_squared_error, 
    root_mean_squared_error
)




def process_v1(train, test, target_col="SalePrice"):
    y_train_full = train[target_col].copy()
    train_features = train.drop(columns=[target_col])
    n_train = train_features.shape[0]

    combined = pd.concat([train_features, test], axis=0, ignore_index=True)

    # 1. Заполнение спецификаций и исправлений
    if "Functional" in combined.columns:
        combined["Functional"] = combined["Functional"].fillna("Typ")

    if "Exterior2nd" in combined.columns:
        corrections = {
            "Wd Shng": "Wd Sdng",
            "CmentBd": "CemntBd",
            "Brk Cmn": "BrkComm",
        }
        combined["Exterior2nd"] = combined["Exterior2nd"].replace(corrections)

    if "LotFrontage" in combined.columns and "Neighborhood" in combined.columns:
        combined["LotFrontage"] = combined.groupby("Neighborhood")[
            "LotFrontage"
        ].transform(lambda s: s.fillna(s.median()))

    # 2. Маппинг качественных признаков
    qual_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0}
    quality_cols = [
        "PoolQC",
        "GarageQual",
        "GarageCond",
        "BsmtQual",
        "BsmtCond",
        "ExterQual",
        "ExterCond",
        "KitchenQual",
        "HeatingQC",
        "FireplaceQu",
    ]
    for col in quality_cols:
        if col in combined.columns:
            combined[col] = combined[col].map(qual_map).fillna(0).astype(int)

    # 3. Генерация новых признаков
    if {"TotalBsmtSF", "1stFlrSF", "2ndFlrSF"}.issubset(combined.columns):
        combined["TotalSF"] = (
            combined["TotalBsmtSF"].fillna(0)
            + combined["1stFlrSF"]
            + combined["2ndFlrSF"]
        )

    if {"YrSold", "YearBuilt"}.issubset(combined.columns):
        combined["HouseAge"] = combined["YrSold"] - combined["YearBuilt"]

    bath_cols = {"FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"}
    if bath_cols.issubset(combined.columns):
        combined["TotalBath"] = (
            combined["FullBath"].fillna(0)
            + 0.5 * combined["HalfBath"].fillna(0)
            + combined["BsmtFullBath"].fillna(0)
            + 0.5 * combined["BsmtHalfBath"].fillna(0)
        )

    if {"OverallQual", "GrLivArea"}.issubset(combined.columns):
        combined["QualityLivArea"] = (
            combined["OverallQual"] * combined["GrLivArea"]
        )

    if {"YrSold", "YearRemodAdd"}.issubset(combined.columns):
        combined["YearsSinceRemodel"] = (
            combined["YrSold"] - combined["YearRemodAdd"]
        )

    has_flag_specs = {
        "HasPool": "PoolArea",
        "Has2ndFloor": "2ndFlrSF",
        "HasBsmt": "TotalBsmtSF",
        "HasFireplace": "Fireplaces",
        "HasGarage": "GarageArea",
    }
    for flag_name, source_col in has_flag_specs.items():
        if source_col in combined.columns:
            combined[flag_name] = (
                combined[source_col].fillna(0) > 0
            ).astype(int)

    # 4. Выделение Neighborhood под безопасный Target Encoding
    HAS_NEIGHBORHOOD = "Neighborhood" in combined.columns
    if HAS_NEIGHBORHOOD:
        neighborhood_all = (
            combined["Neighborhood"].fillna("Unknown").reset_index(drop=True)
        )
        combined = combined.drop(columns=["Neighborhood"])

    # 5. Группировка редких категорий
    RARE_THRESHOLD = 0.01
    categorical_cols = combined.select_dtypes(
        include="object"
    ).columns.tolist()
    for col in categorical_cols:
        freq = combined[col].value_counts(normalize=True, dropna=True)
        rare_categories = freq[freq < RARE_THRESHOLD].index
        if len(rare_categories) > 0:
            combined[col] = combined[col].where(
                ~combined[col].isin(rare_categories), "Other"
            )

    # 6. Обработка пропусков
    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    categorical_cols = combined.select_dtypes(include=["object"]).columns

    for col in numeric_cols:
        if combined[col].isnull().any():
            combined[col] = combined[col].fillna(combined[col].median())

    for col in categorical_cols:
        if combined[col].isnull().any():
            combined[col] = combined[col].fillna("None")

    # 7. Логарифмирование скошенных признаков
    SKEW_THRESHOLD = 0.75
    exclude_from_skew = (
        {"Id"} | set(has_flag_specs.keys()) | set(quality_cols)
    )

    numeric_cols = [
        c
        for c in combined.select_dtypes(include=[np.number]).columns
        if c not in exclude_from_skew
    ]
    skewed = combined[numeric_cols].apply(lambda s: skew(s.dropna()))
    skewed_cols = skewed[skewed.abs() > SKEW_THRESHOLD].index.tolist()

    for col in skewed_cols:
        combined[col] = np.log1p(combined[col].clip(lower=0))

    # 8. OHE и разделение обратно на train / val
    cat_cols = combined.select_dtypes(include="object").columns.tolist()
    combined = pd.get_dummies(combined, columns=cat_cols, dummy_na=False)

    drop_cols = ["Id"] if "Id" in combined.columns else []
    X_train = (
        combined.iloc[:n_train, :].drop(columns=drop_cols).reset_index(drop=True)
    )
    X_test = (
        combined.iloc[n_train:, :].drop(columns=drop_cols).reset_index(drop=True)
    )
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    # 9. Применение Target Encoding для Neighborhood внутри фолда
    if HAS_NEIGHBORHOOD:
        neigh_tr = neighborhood_all.iloc[:n_train].reset_index(drop=True)
        neigh_val = neighborhood_all.iloc[n_train:].reset_index(drop=True)

        target_map = y_train_full.groupby(neigh_tr).mean()
        global_mean = y_train_full.mean()

        X_train["Neighborhood_TE"] = neigh_tr.map(target_map).fillna(global_mean)
        X_test["Neighborhood_TE"] = neigh_val.map(target_map).fillna(global_mean)

    # 10. Логарифмирование целевой переменной (приведение к единому масштабу)
    Y_train_log = np.log1p(y_train_full).reset_index(drop=True)

    return X_train, Y_train_log, X_test

def process_v2(train, test, target_col="SalePrice"):
    y_train_full = train[target_col].copy()
    train_features = train.drop(columns=[target_col])
    n_train = train_features.shape[0]

    combined = pd.concat([train_features, test], axis=0, ignore_index=True)

    # 1. Заполнение спецификаций и исправлений
    if "Functional" in combined.columns:
        combined["Functional"] = combined["Functional"].fillna("Typ")

    if "Exterior2nd" in combined.columns:
        corrections = {
            "Wd Shng": "Wd Sdng",
            "CmentBd": "CemntBd",
            "Brk Cmn": "BrkComm",
        }
        combined["Exterior2nd"] = combined["Exterior2nd"].replace(corrections)

    if "LotFrontage" in combined.columns and "Neighborhood" in combined.columns:
        combined["LotFrontage"] = combined.groupby("Neighborhood")[
            "LotFrontage"
        ].transform(lambda s: s.fillna(s.median()))

    # 2. Маппинг качественных признаков
    qual_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0}
    quality_cols = [
        "PoolQC",
        "GarageQual",
        "GarageCond",
        "BsmtQual",
        "BsmtCond",
        "ExterQual",
        "ExterCond",
        "KitchenQual",
        "HeatingQC",
        "FireplaceQu",
    ]
    for col in quality_cols:
        if col in combined.columns:
            combined[col] = combined[col].map(qual_map).fillna(0).astype(int)

    # 3. Генерация новых признаков
    if {"TotalBsmtSF", "1stFlrSF", "2ndFlrSF"}.issubset(combined.columns):
        combined["TotalSF"] = (
            combined["TotalBsmtSF"].fillna(0)
            + combined["1stFlrSF"]
            + combined["2ndFlrSF"]
        )

    if {"YrSold", "YearBuilt"}.issubset(combined.columns):
        combined["HouseAge"] = combined["YrSold"] - combined["YearBuilt"]

    bath_cols = {"FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"}
    if bath_cols.issubset(combined.columns):
        combined["TotalBath"] = (
            combined["FullBath"].fillna(0)
            + 0.5 * combined["HalfBath"].fillna(0)
            + combined["BsmtFullBath"].fillna(0)
            + 0.5 * combined["BsmtHalfBath"].fillna(0)
        )

    if {"OverallQual", "GrLivArea"}.issubset(combined.columns):
        combined["QualityLivArea"] = (
            combined["OverallQual"] * combined["GrLivArea"]
        )

    if {"YrSold", "YearRemodAdd"}.issubset(combined.columns):
        combined["YearsSinceRemodel"] = (
            combined["YrSold"] - combined["YearRemodAdd"]
        )

    has_flag_specs = {
        "HasPool": "PoolArea",
        "Has2ndFloor": "2ndFlrSF",
        "HasBsmt": "TotalBsmtSF",
        "HasFireplace": "Fireplaces",
        "HasGarage": "GarageArea",
    }
    for flag_name, source_col in has_flag_specs.items():
        if source_col in combined.columns:
            combined[flag_name] = (
                combined[source_col].fillna(0) > 0
            ).astype(int)

    # 4. Выделение Neighborhood под безопасный Target Encoding
    HAS_NEIGHBORHOOD = "Neighborhood" in combined.columns
    if HAS_NEIGHBORHOOD:
        neighborhood_all = (
            combined["Neighborhood"].fillna("Unknown").reset_index(drop=True)
        )
        combined = combined.drop(columns=["Neighborhood"])

    # 5. Группировка редких категорий
    RARE_THRESHOLD = 0.01
    categorical_cols = combined.select_dtypes(
        include="object"
    ).columns.tolist()
    for col in categorical_cols:
        freq = combined[col].value_counts(normalize=True, dropna=True)
        rare_categories = freq[freq < RARE_THRESHOLD].index
        if len(rare_categories) > 0:
            combined[col] = combined[col].where(
                ~combined[col].isin(rare_categories), "Other"
            )

    # 6. Обработка пропусков
    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    categorical_cols = combined.select_dtypes(include=["object"]).columns

    for col in numeric_cols:
        if combined[col].isnull().any():
            combined[col] = combined[col].fillna(combined[col].median())

    for col in categorical_cols:
        if combined[col].isnull().any():
            combined[col] = combined[col].fillna("None")

    # 7. Логарифмирование скошенных признаков
    SKEW_THRESHOLD = 0.75
    exclude_from_skew = (
        {"Id"} | set(has_flag_specs.keys()) | set(quality_cols)
    )

    numeric_cols = [
        c
        for c in combined.select_dtypes(include=[np.number]).columns
        if c not in exclude_from_skew
    ]
    skewed = combined[numeric_cols].apply(lambda s: skew(s.dropna()))
    skewed_cols = skewed[skewed.abs() > SKEW_THRESHOLD].index.tolist()

    for col in skewed_cols:
        combined[col] = np.log1p(combined[col].clip(lower=0))

    # 8. OHE и разделение обратно на train / val
    cat_cols = combined.select_dtypes(include="object").columns.tolist()
    combined = pd.get_dummies(combined, columns=cat_cols, dummy_na=False)

    # Заменяем boolean-столбцы от get_dummies на float32
    bool_cols = combined.select_dtypes(include="bool").columns
    combined[bool_cols] = combined[bool_cols].astype(np.float32)

    drop_cols = ["Id"] if "Id" in combined.columns else []
    X_train = (
        combined.iloc[:n_train, :].drop(columns=drop_cols).reset_index(drop=True)
    )
    X_test = (
        combined.iloc[n_train:, :].drop(columns=drop_cols).reset_index(drop=True)
    )
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    # 9. Применение Target Encoding для Neighborhood внутри фолда
    if HAS_NEIGHBORHOOD:
        neigh_tr = neighborhood_all.iloc[:n_train].reset_index(drop=True)
        neigh_val = neighborhood_all.iloc[n_train:].reset_index(drop=True)

        # Обучаем Target Encoding на log(y_train_full)
        y_log_train = np.log1p(y_train_full)
        target_map = y_log_train.groupby(neigh_tr).mean()
        global_mean = y_log_train.mean()

        X_train["Neighborhood_TE"] = neigh_tr.map(target_map).fillna(global_mean)
        X_test["Neighborhood_TE"] = neigh_val.map(target_map).fillna(global_mean)

    # 10. ВАЖНО ДЛЯ НЕЙРОСЕТЕЙ: Стандартизация всех признаков X
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        dtype=np.float32,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        dtype=np.float32,
    )

    # 11. Логарифмирование целевой переменной
    Y_train_log = np.log1p(y_train_full).reset_index(drop=True)

    return X_train_scaled, Y_train_log, X_test_scaled

def evaluate_pipeline_with_cv_regression(
    train_df, processing_func, target_col="SalePrice", pipeline_name="Pipeline"
):
    models_dict = {
        "Linear Regression": LinearRegression(),
        "Ridge (L2)": Ridge(random_state=42),
        "Lasso (L1)": Lasso(random_state=42),
        "ElasticNet": ElasticNet(random_state=42),
        "Support Vector Regression": SVR(),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
        "Stochastic Gradient Descent": SGDRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            random_state=42, n_estimators=100
        ),
        # --- Градиентный бустинг ---
        "CatBoost": CatBoostRegressor(verbose=0, random_state=42),
        "XGBoost": XGBRegressor(random_state=42),
        "LightGBM": LGBMRegressor(random_state=42, verbose=-1),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for name, model in models_dict.items():
        rmse_scores = []

        # Разбиваем train_df на 5 фолдов
        for train_idx, val_idx in kf.split(train_df):
            fold_train = train_df.iloc[train_idx]
            fold_val = train_df.iloc[val_idx]

            # Изолированная обработка данных ВНУТРИ фолда
            X_tr, Y_tr, X_val = processing_func(
                fold_train, fold_val, target_col
            )

            # Приводим целевую переменную валидации к логарифмическому масштабу (как и Y_tr)
            Y_val = np.log1p(fold_val[target_col])

            model.fit(X_tr, Y_tr)

            # Предсказание и расчёт RMSE (через np.sqrt для совместимости со свежим scikit-learn)
            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(Y_val, preds))
            rmse_scores.append(rmse)

        # Среднее значение RMSE (RMSLE) по 5 фолдам
        results.append(
            {
                "Model": name,
                f"{pipeline_name}_CV_RMSE": round(np.mean(rmse_scores), 4),
                f"{pipeline_name}_Std": round(np.std(rmse_scores), 4),
            }
        )

    # Для RMSE чем меньше значение, тем лучше (ascending=True)
    return (
        pd.DataFrame(results)
        .sort_values(by=f"{pipeline_name}_CV_RMSE", ascending=True)
        .reset_index(drop=True)
    )

































