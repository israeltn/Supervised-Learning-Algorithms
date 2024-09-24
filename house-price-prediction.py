from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Sample house data (area, bedrooms, age, price)
houses = [
    (1500, 3, 10, 250000),
    (2000, 4, 5, 350000),
    (1200, 2, 15, 180000),
    (1800, 3, 8, 300000)
]

# Separate features and target
X, y = zip(*[(house[:-1], house[-1]) for house in houses])

# Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), [0, 1, 2])
    ])

# Create a pipeline with preprocessing and Random Forest Regressor
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
pipeline.fit(X, y)

# Predict prices for new houses
new_houses = [(1600, 3, 12), (2200, 4, 3)]
predictions = pipeline.predict(new_houses)

for house, prediction in zip(new_houses, predictions):
    print(f"House: {house}")
    print(f"Predicted price: ${prediction:.2f}\n")
