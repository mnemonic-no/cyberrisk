

from typing import Dict

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
    FunctionTransformer
)
from preprocessing.features import StringToListTransformer, CWEBinaryEncoder, safe_log

def create_preprocessing_pipeline(
    feature_columns: Dict[str, list]
) -> ColumnTransformer:
    transformers = []

    if "cwe_features" in feature_columns:
        cwe_pipeline = Pipeline(
            steps=[
                ("to_list", StringToListTransformer()),
                ("binary_encode", CWEBinaryEncoder()),
            ]
        )
        transformers.append(("cwe", cwe_pipeline, feature_columns["cwe_features"]))

    if "log_features" in feature_columns:
        log_pipeline = make_pipeline(
            SimpleImputer(strategy="constant", fill_value=1e-6),
            FunctionTransformer(safe_log, feature_names_out="one-to-one"),
            StandardScaler(),
        )
        transformers.append(("log_num", log_pipeline, feature_columns["log_features"]))

    if "other_numeric_features" in feature_columns:
        other_numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(
            (
                "other_num",
                other_numeric_transformer,
                feature_columns["other_numeric_features"],
            )
        )

    if "categorical_features" in feature_columns:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(
            ("cat", categorical_transformer, feature_columns["categorical_features"])
        )

    if "ordinal_features" in feature_columns:
        ordinal_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
                (
                    "ordinal",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                ),
            ]
        )
        transformers.append(
            ("ordinal", ordinal_transformer, feature_columns["ordinal_features"])
        )
    
    if "passthrough" in feature_columns:
        transformers.append(
            ("passthrough", "passthrough", feature_columns["passthrough"])
        )

    return ColumnTransformer(transformers=transformers)


def create_full_pipeline(
    classifier: BaseEstimator, feature_columns: Dict[str, list]
) -> Pipeline:
    preprocessor = create_preprocessing_pipeline(feature_columns)
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
