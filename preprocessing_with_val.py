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

def evaluate_pipeline_with_cv(
    train_df, processing_func, pipeline_name='Pipeline'
):
    models_dict = {
        'Logistic Regression': LogisticRegression(
            random_state=42, max_iter=1000
        ),
        'Logistic Regression (L2)': LogisticRegression(
            penalty='l2', solver='saga', random_state=42, max_iter=2000
        ),
        'Logistic Regression (L1)': LogisticRegression(
            penalty='l1', solver='saga', random_state=42, max_iter=2000
        ),
        'Logistic Regression (ElasticNet)': LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            l1_ratio=0.5,
            random_state=42,
            max_iter=2000,
        ),
        'Support Vector Machines': SVC(random_state=42, probability=True),
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'Naive Bayes': GaussianNB(),
        'Perceptron': Perceptron(random_state=42),
        'Linear SVC': LinearSVC(random_state=42, max_iter=2000),
        'Stochastic Gradient Decent': SGDClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(
            random_state=42, n_estimators=100
        ),
        # --- Градиентный бустинг ---
        'CatBoost': CatBoostClassifier(verbose=0, random_state=42),
        'XGBoost': XGBClassifier(
            random_state=42, eval_metric='logloss', use_label_encoder=False
        ),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for name, model in models_dict.items():
        cv_scores = []

        # Разбиваем train_df на 5 фолдов
        for train_idx, val_idx in skf.split(train_df, train_df['Survived']):
            fold_train = train_df.iloc[train_idx]
            fold_val = train_df.iloc[val_idx]

            # Изолированная обработка данных ВНУТРИ фолда!
            X_tr, Y_tr, X_val = processing_func(fold_train, fold_val)

            model.fit(X_tr, Y_tr)

            # Оценка качества на валидационном фолде
            val_score = round(model.score(X_val, fold_val['Survived']) * 100, 2)
            cv_scores.append(val_score)

        # Среднее значение точности по 5 фолдам
        results.append(
            {
                'Model': name,
                f'{pipeline_name}_CV_Acc': round(np.mean(cv_scores), 2),
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values(by=f'{pipeline_name}_CV_Acc', ascending=False)
        .reset_index(drop=True)
    )




def process_v1(train_df, test_df):
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']
    X_tr, X_te = train_df[features].copy(), test_df[features].copy()

    age_median = X_tr['Age'].median()
    X_tr['Age'] = X_tr['Age'].fillna(age_median)
    X_te['Age'] = X_te['Age'].fillna(age_median)

    bins = [-1, 0, 2, np.inf]
    labels = ['Alone', 'Few_persons', 'Family']
    X_tr['SibSp_new'] = pd.cut(
        X_tr['SibSp'], bins=bins, labels=labels, include_lowest=True
    )
    X_te['SibSp_new'] = pd.cut(
        X_te['SibSp'], bins=bins, labels=labels, include_lowest=True
    )

    sex_map = {'female': 1, 'male': 0}
    pclass_map = {1: 'Upper', 2: 'Middle', 3: 'Lower'}
    embarked_map = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}

    for df in [X_tr, X_te]:
        df['Sex'] = df['Sex'].map(sex_map)
        df['Pclass'] = df['Pclass'].map(pclass_map)
        df['Embarked'] = df['Embarked'].map(embarked_map)

    cols_to_use = ['Pclass', 'Sex', 'SibSp_new', 'Parch', 'Embarked', 'Age']
    X_tr_encoded = pd.get_dummies(X_tr[cols_to_use], drop_first=True)
    X_te_encoded = pd.get_dummies(X_te[cols_to_use], drop_first=True)

    X_tr_encoded, X_te_encoded = X_tr_encoded.align(
        X_te_encoded, join='left', axis=1, fill_value=0
    )

    for df in [X_tr_encoded, X_te_encoded]:
        bool_cols = df.select_dtypes(include='bool').columns
        df[bool_cols] = df[bool_cols].astype(int)

    return X_tr_encoded, train_df['Survived'], X_te_encoded


def process_v2(train_df, test_df):
    age_median = train_df['Age'].median()
    train_data = train_df[
        ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']
    ].copy()
    test_data = test_df[
        ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']
    ].copy()

    for df in [train_data, test_data]:
        df['Age'] = df['Age'].fillna(age_median).astype(int)
        df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})
        df['Embarked'] = df['Embarked'].map(
            {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}
        )
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        df['Age'] = pd.cut(
            df['Age'],
            bins=[-float('inf'), 16, 32, 48, 64, float('inf')],
            labels=[0, 1, 2, 3, 4],
        ).astype(int)

    encoder = TargetEncoder(cols=['Embarked'])
    train_data['Embarked_encoded'] = encoder.fit_transform(
        train_data['Embarked'], train_data['Survived']
    ).round(2)
    test_data['Embarked_encoded'] = encoder.transform(
        test_data['Embarked']
    ).round(2)

    for df in [train_data, test_data]:
        df['Age*Pclass'] = df['Age'] * df['Pclass']
        df.drop(columns=['Embarked'], inplace=True)

    X_train = train_data.drop(columns=['Survived', 'Parch', 'Age'])
    Y_train = train_data['Survived']
    X_test = test_data.drop(columns=['Parch', 'Age'])

    return X_train, Y_train, X_test


def process_v3(train_df, test_df):
    age_median = train_df['Age'].median()
    raw_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']

    X_tr, X_te = train_df[raw_cols].copy(), test_df[raw_cols].copy()
    X_tr['Age'] = X_tr['Age'].fillna(age_median).astype(int)
    X_te['Age'] = X_te['Age'].fillna(age_median).astype(int)

    sex_map = {'female': 1, 'male': 0}
    pclass_map = {1: 'Upper', 2: 'Middle', 3: 'Lower'}
    embarked_map = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}
    bins = [0, 12, 18, 35, 60, 100]

    for df in [X_tr, X_te]:
        df['Sex'] = df['Sex'].map(sex_map)
        df['Pclass'] = df['Pclass'].map(pclass_map)
        df['Embarked'] = df['Embarked'].map(embarked_map)
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        df['AgeGroup'] = pd.cut(
            df['Age'], bins=bins, labels=False, right=False
        )

    features = [
        'Pclass',
        'Sex',
        'Parch',
        'Embarked',
        'FamilySize',
        'IsAlone',
        'AgeGroup',
    ]
    X_tr_encoded = pd.get_dummies(X_tr[features], dtype=int, drop_first=True)
    X_te_encoded = pd.get_dummies(X_te[features], dtype=int, drop_first=True)
    X_tr_encoded, X_te_encoded = X_tr_encoded.align(
        X_te_encoded, join='left', axis=1, fill_value=0
    )

    scaler = StandardScaler()
    X_tr_encoded['Age_scaled'] = scaler.fit_transform(X_tr[['Age']])
    X_te_encoded['Age_scaled'] = scaler.transform(X_te[['Age']])

    return X_tr_encoded, train_df['Survived'], X_te_encoded


def process_v4(train_df, test_df):
    df_train, df_test = train_df.copy(), test_df.copy()

    fare_median = df_train['Fare'].median()
    df_train['Fare'] = df_train['Fare'].fillna(fare_median)
    df_test['Fare'] = df_test['Fare'].fillna(fare_median)

    age_medians = df_train.groupby(['Sex', 'Pclass'])['Age'].median()

    def fill_age(df):
        return df.apply(
            lambda row: age_medians.get(
                (row['Sex'], row['Pclass']), df_train['Age'].median()
            )
            if pd.isna(row['Age'])
            else row['Age'],
            axis=1,
        ).astype(int)

    df_train['Age'] = fill_age(df_train)
    df_test['Age'] = fill_age(df_test)

    sex_map = {'male': 0, 'female': 1}
    embarked_map = {'S': 0, 'C': 1, 'Q': 2}

    for df in [df_train, df_test]:
        df['Sex'] = df['Sex'].map(sex_map)
        df['Embarked'] = df['Embarked'].fillna('S').map(embarked_map)
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    feature_cols = [
        'Pclass',
        'Sex',
        'Age',
        'SibSp',
        'Parch',
        'Fare',
        'Embarked',
        'FamilySize',
        'IsAlone',
    ]
    return (
        df_train[feature_cols],
        df_train['Survived'],
        df_test[feature_cols],
    )


# ==========================================
# 2. УНИВЕРСАЛЬНАЯ ОЦЕНКА МОДЕЛЕЙ (5-FOLD CV)
# ==========================================





