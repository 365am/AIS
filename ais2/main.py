"""Train and evaluate air-quality classification models."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


AQI_CATEGORIES = (
    (50, "Good"),
    (100, "Moderate"),
    (150, "Unhealthy for Sensitive Groups"),
    (200, "Unhealthy"),
    (300, "Very Unhealthy"),
    (500, "Hazardous"),
)


def decode_arff_values(dataframe):
    """Convert nominal ARFF byte strings into regular Python strings."""
    for column in dataframe.select_dtypes(include="object"):
        dataframe[column] = dataframe[column].map(
            lambda value: value.decode("utf-8")
            if isinstance(value, bytes)
            else value
        )
    return dataframe


def category_from_aqi(values):
    """Return the category for the maximum supplied pollutant-specific AQI."""
    if not values:
        raise ValueError("Enter at least one pollutant-specific AQI value.")

    aqi_values = [float(value) for value in values]
    if not all(np.isfinite(value) and 0 <= value <= 500 for value in aqi_values):
        raise ValueError("AQI values must be finite numbers from 0 to 500.")

    overall_aqi = max(aqi_values)
    for upper_bound, category in AQI_CATEGORIES:
        if overall_aqi <= upper_bound:
            return overall_aqi, category

    raise ValueError("AQI values must be between 0 and 500.")


def parse_current_aqi(entries):
    """Parse repeated command-line values in NAME=VALUE format."""
    current_values = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                f"Invalid current AQI value {entry!r}; use FEATURE_NAME=VALUE."
            )
        name, raw_value = entry.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError("Current AQI feature names cannot be empty.")
        if name in current_values:
            raise ValueError(f"Current AQI for {name!r} was provided more than once.")
        try:
            value = float(raw_value)
        except ValueError as error:
            raise ValueError(f"Invalid AQI number for {name!r}: {raw_value!r}.") from error
        current_values[name] = value
    return current_values


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Decision Tree and an MLP on air_quality.arff."
    )
    parser.add_argument(
        "--data",
        default=str(Path(__file__).with_name("air_quality.arff")),
        help="Path to the ARFF dataset (default: air_quality.arff beside this script).",
    )
    parser.add_argument(
        "--target",
        help="Name of the class column. Defaults to the dataset's last column.",
    )
    parser.add_argument(
        "--current-aqi",
        action="append",
        default=[],
        metavar="FEATURE=VALUE",
        help=(
            "Current pollutant-specific AQI; repeat for each pollutant, using "
            "feature names exactly as printed by this script."
        ),
    )
    parser.add_argument(
        "--reported-category",
        help="Category shown by the air-quality website, for comparison.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.data)
    if not dataset_path.is_file():
        parser.error(
            f"Dataset not found: {dataset_path}. Download air_quality.arff from "
            "your e-classroom and pass its location with --data."
        )

    raw_data, _ = arff.loadarff(str(dataset_path))
    dataframe = decode_arff_values(pd.DataFrame(raw_data))

    if dataframe.empty or len(dataframe.columns) < 2:
        raise ValueError("The ARFF file must contain rows, features, and a class column.")

    print(f"Dataset: {dataset_path}")
    print(f"Shape: {dataframe.shape[0]} rows x {dataframe.shape[1]} columns")
    print("Columns and data types:")
    print(dataframe.dtypes.to_string())
    print("\nFirst five rows:")
    print(dataframe.head().to_string(index=False))

    target_column = args.target or dataframe.columns[-1]
    if target_column not in dataframe.columns:
        raise ValueError(
            f"Target column {target_column!r} was not found. "
            f"Available columns: {dataframe.columns.tolist()}"
        )

    features = dataframe.drop(columns=[target_column])
    target = dataframe[target_column]

    if target.isna().any():
        raise ValueError(
            f"Target column {target_column!r} contains missing values; "
            "clean or correct those labels before training."
        )
    non_numeric_features = [
        column
        for column in features.columns
        if not pd.api.types.is_numeric_dtype(features[column])
    ]
    if non_numeric_features:
        raise ValueError(
            "All input features must be numeric AQI values. "
            f"Non-numeric columns: {non_numeric_features}"
        )

    print(f"\nTarget column: {target_column}")
    print("Class counts:")
    print(target.value_counts(dropna=False).to_string())

    class_labels = list(pd.unique(target))
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )
    print(
        f"\nTraining samples: {len(x_train)} "
        f"({len(x_train) / len(dataframe):.0%})"
    )
    print(f"Testing samples: {len(x_test)} ({len(x_test) / len(dataframe):.0%})")
    print(f"Confusion matrix class order: {class_labels}")

    models = {
        "Decision Tree": make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeClassifier(random_state=42),
        ),
        "Multilayer Perceptron": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            MLPClassifier(max_iter=1000, random_state=42),
        ),
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        trained_models[name] = model

        print(f"\n=== {name} ===")
        print(f"Test accuracy: {accuracy_score(y_test, predictions):.4f}")
        print("Confusion matrix (rows = actual, columns = predicted):")
        print(confusion_matrix(y_test, predictions, labels=class_labels))
        print("Per-class metrics:")
        print(
            classification_report(
                y_test,
                predictions,
                labels=class_labels,
                zero_division=0,
            )
        )

    if args.current_aqi:
        current_values = parse_current_aqi(args.current_aqi)
        unknown_features = sorted(set(current_values) - set(features.columns))
        if unknown_features:
            raise ValueError(
                "These current AQI feature names are not in the dataset: "
                f"{unknown_features}. Use the exact names printed above."
            )

        overall_aqi, direct_category = category_from_aqi(
            list(current_values.values())
        )
        print("\n=== Current AQI comparison ===")
        print(
            f"Direct maximum rule: AQI {overall_aqi:g} -> {direct_category}. "
            "This assumes every supplied value is a pollutant-specific AQI."
        )

        if set(current_values) == set(features.columns):
            current_row = pd.DataFrame(
                [[current_values[column] for column in features.columns]],
                columns=features.columns,
            )
            for name, model in trained_models.items():
                print(f"{name} prediction: {model.predict(current_row)[0]}")
        else:
            missing_features = [
                column for column in features.columns if column not in current_values
            ]
            print(
                "Skipped model predictions because current AQIs were not supplied "
                "for every model feature. Missing: "
                f"{missing_features}"
            )

        if args.reported_category:
            print(f"Website-reported category: {args.reported_category}")
            print(
                "Direct rule agrees with website: "
                f"{direct_category.casefold() == args.reported_category.strip().casefold()}"
            )
    else:
        print(
            "\nTo compare with current Ljubljana readings, rerun with --current-aqi "
            "FEATURE=VALUE for each pollutant and optionally --reported-category."
        )


if __name__ == "__main__":
    main()