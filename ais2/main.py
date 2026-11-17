"""Train and evaluate air-quality classification models."""

# argparse reads command-line options such as --data and --target.
import argparse
# Path provides reliable, operating-system-independent file paths.
from pathlib import Path

# numpy supplies numeric constants and finite-number checks.
import numpy as np
# pandas stores the ARFF dataset as a table (DataFrame).
import pandas as pd
# scipy loads the .arff dataset file.
from scipy.io import arff
# SimpleImputer replaces missing feature values.
from sklearn.impute import SimpleImputer
# These functions calculate accuracy and detailed classification results.
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# train_test_split divides the data into separate training and test sets.
from sklearn.model_selection import train_test_split
# MLPClassifier is a multilayer neural network classifier.
from sklearn.neural_network import MLPClassifier
# make_pipeline joins preprocessing and model steps into one object.
from sklearn.pipeline import make_pipeline
# StandardScaler normalizes feature values for the neural network.
from sklearn.preprocessing import StandardScaler
# DecisionTreeClassifier learns rules that divide the feature space.
from sklearn.tree import DecisionTreeClassifier


# AQI category boundaries, in ascending order. The first matching upper bound
# determines the category for a pollutant-specific or overall AQI value.
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
    # ARFF nominal/text values are sometimes decoded by scipy as byte strings.
    # This loop examines only columns containing Python object values.
    for column in dataframe.select_dtypes(include="object"):
        # Decode each byte value to UTF-8; keep already-decoded values unchanged.
        dataframe[column] = dataframe[column].map(
            lambda value: value.decode("utf-8")
            if isinstance(value, bytes)
            else value
        )
    # Return the modified table so it can be used by the rest of the script.
    return dataframe


def category_from_aqi(values):
    """Return the category for the maximum supplied pollutant-specific AQI."""
    # Refuse an empty list because an overall AQI cannot be calculated from it.
    if not values:
        raise ValueError("Enter at least one pollutant-specific AQI value.")

    # Convert every supplied value to float so that strings and numbers work alike.
    aqi_values = [float(value) for value in values]
    # All values must be finite and within the AQI range of 0 through 500.
    if not all(np.isfinite(value) and 0 <= value <= 500 for value in aqi_values):
        raise ValueError("AQI values must be finite numbers from 0 to 500.")

    # The direct US EPA-style rule uses the highest pollutant-specific AQI.
    overall_aqi = max(aqi_values)
    # Search the ordered boundaries from lowest to highest.
    for upper_bound, category in AQI_CATEGORIES:
        if overall_aqi <= upper_bound:
            return overall_aqi, category

    # This line is normally unreachable after the range validation above.
    raise ValueError("AQI values must be between 0 and 500.")


def parse_current_aqi(entries):
    """Parse repeated command-line values in NAME=VALUE format."""
    # Store parsed values in a dictionary to prevent duplicate feature names.
    current_values = {}
    # Process each --current-aqi argument independently.
    for entry in entries:
        # Every entry must contain one equals sign separating feature and value.
        if "=" not in entry:
            raise ValueError(
                f"Invalid current AQI value {entry!r}; use FEATURE_NAME=VALUE."
            )
        # Split only at the first "=" so the feature name and number are unambiguous.
        name, raw_value = entry.split("=", 1)
        # Remove surrounding whitespace from the feature name.
        name = name.strip()
        if not name:
            raise ValueError("Current AQI feature names cannot be empty.")
        # A feature can appear only once in the current-reading comparison.
        if name in current_values:
            raise ValueError(f"Current AQI for {name!r} was provided more than once.")
        try:
            # AQI input is numeric because it will be passed to models and calculations.
            value = float(raw_value)
        except ValueError as error:
            raise ValueError(f"Invalid AQI number for {name!r}: {raw_value!r}.") from error
        # Save the successfully parsed feature/value pair.
        current_values[name] = value
    # Return a dictionary such as {"pm25": 42.0, "pm10": 55.0}.
    return current_values


def main():
    # Resolve the normal and downloaded .txt dataset filenames relative to this script.
    default_dataset = Path(__file__).with_name("air_quality.arff")
    text_download = Path(__file__).with_name("air_quality.arff.txt")
    # Prefer the downloaded text filename only when the standard ARFF file is absent.
    if not default_dataset.is_file() and text_download.is_file():
        default_dataset = text_download

    # Create an argument parser that also generates a useful --help screen.
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Decision Tree and an MLP on air_quality.arff."
    )
    # --data allows the user to choose a dataset in another folder or filename.
    parser.add_argument(
        "--data",
        default=str(default_dataset),
        help=(
            "Path to the ARFF dataset (defaults to air_quality.arff, or "
            "air_quality.arff.txt if that is the downloaded filename)."
        ),
    )
    # --target overrides the default class column, which is otherwise the last column.
    parser.add_argument(
        "--target",
        help="Name of the class column. Defaults to the dataset's last column.",
    )
    # append allows repeated --current-aqi options, one for each pollutant feature.
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
    # --reported-category compares a website category with the direct AQI category.
    parser.add_argument(
        "--reported-category",
        help="Category shown by the air-quality website, for comparison.",
    )
    # Convert all command-line strings into the args object.
    args = parser.parse_args()

    # Convert the selected dataset argument into a Path object.
    dataset_path = Path(args.data)
    # Stop with a parser error rather than attempting to open a nonexistent file.
    if not dataset_path.is_file():
        parser.error(
            f"Dataset not found: {dataset_path}. Download air_quality.arff from "
            "your e-classroom, or pass the downloaded .arff.txt path with --data."
        )

    # scipy returns both the data and attribute metadata; metadata is not needed.
    raw_data, _ = arff.loadarff(str(dataset_path))
    # NumPy structured data is converted to a pandas table and nominal bytes decoded.
    dataframe = decode_arff_values(pd.DataFrame(raw_data))

    # ARFF must contain at least one row and at least one feature plus a target.
    if dataframe.empty or len(dataframe.columns) < 2:
        raise ValueError("The ARFF file must contain rows, features, and a class column.")

    # Print basic dataset information for inspection and assignment documentation.
    print(f"Dataset: {dataset_path}")
    print(f"Shape: {dataframe.shape[0]} rows x {dataframe.shape[1]} columns")
    print("Columns and data types:")
    print(dataframe.dtypes.to_string())
    print("\nFirst five rows:")
    print(dataframe.head().to_string(index=False))

    # Use --target when provided; otherwise assume the final ARFF column is the label.
    target_column = args.target or dataframe.columns[-1]
    # Verify that the chosen class column actually exists in the table.
    if target_column not in dataframe.columns:
        raise ValueError(
            f"Target column {target_column!r} was not found. "
            f"Available columns: {dataframe.columns.tolist()}"
        )

    # Features contain all columns except the class label.
    features = dataframe.drop(columns=[target_column])
    # Target contains the class labels that each model learns to predict.
    target = dataframe[target_column]

    if target.isna().any():
        raise ValueError(
            f"Target column {target_column!r} contains missing values; "
            "clean or correct those labels before training."
        )
    # Identify feature columns that cannot be used directly as numeric model inputs.
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

    # Show the target name and how many rows belong to each class.
    print(f"\nTarget column: {target_column}")
    print("Class counts:")
    print(target.value_counts(dropna=False).to_string())

    # Preserve stable class order for all manually labeled confusion matrices.
    class_labels = list(pd.unique(target))
    # Split 80% for training and 20% for testing; stratify preserves class proportions.
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )
    # Report the exact size of both independent portions of the dataset.
    print(
        f"\nTraining samples: {len(x_train)} "
        f"({len(x_train) / len(dataframe):.0%})"
    )
    print(f"Testing samples: {len(x_test)} ({len(x_test) / len(dataframe):.0%})")
    print(f"Confusion matrix class order: {class_labels}")

    # Define both classifiers and their required preprocessing in scikit-learn pipelines.
    models = {
        # Median imputation fills missing values; the tree itself needs no scaling.
        "Decision Tree": make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeClassifier(random_state=42),
        ),
        # Median imputation precedes scaling, which helps MLP training converge reliably.
        "Multilayer Perceptron": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            MLPClassifier(max_iter=1000, random_state=42),
        ),
    }

    # Keep fitted models so they can be reused for the optional current-reading prediction.
    trained_models = {}
    # Train and evaluate each model independently using the same held-out test set.
    for name, model in models.items():
        # Fit learns parameters only from the training portion.
        model.fit(x_train, y_train)
        # Predict applies the fitted model to the unseen test portion.
        predictions = model.predict(x_test)
        # Save the fitted pipeline for possible later use.
        trained_models[name] = model

        # Print a clearly labeled evaluation section for this classifier.
        print(f"\n=== {name} ===")
        # Accuracy is the percentage of test rows whose predicted label is correct.
        print(f"Test accuracy: {accuracy_score(y_test, predictions):.4f}")
        print("Confusion matrix (rows = actual, columns = predicted):")
        # Rows represent actual labels and columns represent predicted labels.
        print(confusion_matrix(y_test, predictions, labels=class_labels))
        print("Per-class metrics:")
        print(
            # Precision, recall, and F1 are printed for every class in class_labels.
            classification_report(
                y_test,
                predictions,
                labels=class_labels,
                zero_division=0,
            )
        )

    if args.current_aqi:
        # Convert repeated NAME=VALUE arguments into a feature-name dictionary.
        current_values = parse_current_aqi(args.current_aqi)
        # Find names the user supplied that do not correspond to dataset columns.
        unknown_features = sorted(set(current_values) - set(features.columns))
        if unknown_features:
            raise ValueError(
                "These current AQI feature names are not in the dataset: "
                f"{unknown_features}. Use the exact names printed above."
            )

        # The direct rule ignores all feature names and uses the maximum supplied AQI.
        overall_aqi, direct_category = category_from_aqi(
            list(current_values.values())
        )
        print("\n=== Current AQI comparison ===")
        print(
            f"Direct maximum rule: AQI {overall_aqi:g} -> {direct_category}. "
            "This assumes every supplied value is a pollutant-specific AQI."
        )

        # A model needs exactly one value for every input column.
        if set(current_values) == set(features.columns):
            # Build a one-row DataFrame in the same column order used during training.
            current_row = pd.DataFrame(
                [[current_values[column] for column in features.columns]],
                columns=features.columns,
            )
            # Ask each fitted pipeline to predict the category for the current reading.
            for name, model in trained_models.items():
                print(f"{name} prediction: {model.predict(current_row)[0]}")
        else:
            # Identify missing features explicitly instead of inventing values.
            missing_features = [
                column for column in features.columns if column not in current_values
            ]
            print(
                "Skipped model predictions because current AQIs were not supplied "
                "for every model feature. Missing: "
                f"{missing_features}"
            )

        # Optionally compare the direct category to the category stated on the website.
        if args.reported_category:
            print(f"Website-reported category: {args.reported_category}")
            print(
                "Direct rule agrees with website: "
                f"{direct_category.casefold() == args.reported_category.strip().casefold()}"
            )
    else:
        # Explain how to rerun the script for the real-time comparison analysis.
        print(
            "\nTo compare with current Ljubljana readings, rerun with --current-aqi "
            "FEATURE=VALUE for each pollutant and optionally --reported-category."
        )


if __name__ == "__main__":
    # Calling main() starts the complete program only when this file is executed,
    # rather than when it is imported as a Python module.
    main()