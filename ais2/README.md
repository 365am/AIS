# Air Quality Classification Using Python

This project trains a Decision Tree and a Multilayer Perceptron (MLP) to classify
air-quality categories from the supplied pollutant-specific AQI measurements. It
reports test accuracy and confusion matrices, and can compare model predictions
with current measurements and the direct maximum-AQI rule.

## 1. Install Python dependencies

Open a terminal in this folder and run:

```bash
python -m pip install pandas scipy scikit-learn
```

Python 3.9 or newer is recommended.

## 2. Inspect the dataset and run the models

Run:

```bash
python main.py
```

The script prints the dataset shape, column names and data types, a sample of
the rows, and the number of examples in each class. If the class column
has another name, pass it explicitly:

```bash
python main.py --target quality
```

Column names must match the ARFF file exactly. If the dataset is elsewhere, use
`--data`, for example:

```bash
python main.py --data "C:\path\to\air_quality.arff.txt" --target quality
```

For the supplied dataset, the class column is `quality`; the input columns are
`pm25`, `pm10`, `o3`, `no2`, and `so2`.

The script makes a stratified, reproducible 80% training / 20% testing split.
It trains:

- A scikit-learn `DecisionTreeClassifier` with its standard settings.
- An `MLPClassifier` with a median imputer and feature standardization. Scaling
  matters for neural networks because features with larger numeric ranges can
  otherwise dominate learning.

Both use a fixed random seed so that the split and model initialization are
reproducible. Missing numeric feature values are filled with the training
features' medians. The target column must not contain missing values, and all
input features must be numeric.

## 3. Evaluation Results

For each classifier the script prints:

- **Test accuracy**: the fraction of the held-out test examples classified
  correctly.
- **Confusion matrix**: rows are actual classes and columns are predicted
  classes. Diagonal values are correct predictions; off-diagonal values are
  errors. The class order is printed immediately before the matrices.
- **Per-class precision, recall, and F1 score**: additional measures of
  performance for each category.



## 4. Compare against current Ljubljana readings

Visit the Ljubljana air-quality page:

<https://aqicn.org/city/slovenia/lj-bezigrad/>

Record the pollutant-specific AQI values (for example PM2.5, PM10, O3, NO2, and
SO2), the overall category shown, and when you checked it. These readings
change. The current AQI values need to correspond to the same features used by
the dataset. Run the script with one `--current-aqi FEATURE=VALUE` option per
feature, using the exact feature names printed by `main.py`:

```bash
python main.py --current-aqi "pm25=42" --current-aqi "pm10=55" --current-aqi "o3=31" --current-aqi "no2=18" --current-aqi "so2=4" --reported-category "moderate"
```

Replace the example readings, feature names, and website category with the
values you actually observed. If the target column is not the last column, also
include `--target "YOUR CLASS COLUMN"`. If the ARFF feature names differ from
these examples, use the dataset's actual names.

The two models can make a prediction only when a current value is supplied for
**every** feature column. If the dataset contains additional features that are
not available on the website, the script will still calculate the direct rule
from the values you entered, but it will skip model predictions for the current
readings. Do not invent missing measurements.

## 5. Direct AQI calculation and interpretation

The script calculates the maximum of the supplied pollutant-specific AQI
values, then maps it to the assignment's categories:

| Maximum AQI | Category |
| --- | --- |
| 0–50 | Good |
| 51–100 | Moderate |
| 101–150 | Unhealthy for Sensitive Groups |
| 151–200 | Unhealthy |
| 201–300 | Very Unhealthy |
| 301–500 | Hazardous |

This rule is valid when the inputs are pollutant-specific **AQI values**, not
raw pollutant concentrations. Use all five pollutant readings when available.

In the report, say whether the direct category and model predictions agree
with the website, and discuss potential differences such as different
measurement times, the dataset's location or period, limited features, or
classification errors. If the dataset's target labels were themselves
calculated directly from the maximum of those same pollutant AQIs, the direct
rule already defines the answer; machine learning is not necessary to compute
it, though it is useful for comparing learned classifiers as the assignment
asks.
