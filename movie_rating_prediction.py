import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
df = pd.read_csv("IMDb Movies India.csv", encoding="ISO-8859-1")

# Drop rows where Rating is missing
df = df.dropna(subset=["Rating"])

# Clean and preprocess
df["Year"] = df["Year"].str.extract(r"(\d{4})")
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Votes"] = df["Votes"].str.replace(",", "", regex=False)
df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce")
df = df.drop(columns=["Duration", "Name"])

# Fill missing categorical values
categorical_cols = ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]
df[categorical_cols] = df[categorical_cols].fillna("Unknown")

# Reduce high-cardinality features
def reduce_cardinality(series, top_n=20):
    top = series.value_counts().nlargest(top_n).index
    return series.apply(lambda x: x if x in top else "Other")

for col in categorical_cols:
    df[col] = reduce_cardinality(df[col])

# One-hot encode
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

# Final dataset
X = pd.concat([df[["Year", "Votes"]].reset_index(drop=True), encoded_df], axis=1)
y = df["Rating"].reset_index(drop=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R²):", r2)