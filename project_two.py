import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

# Load the dataset
url = 'https://raw.githubusercontent.com/datasets/global-temp/master/data/annual.csv'
weather_df = pd.read_csv(url)

# Display the first few rows of the dataset
print(weather_df.head())

# Check for missing values
print(weather_df.isnull().sum())

# Summary statistics
print(weather_df.describe())

# Inspect the column types to find non-numeric columns
print(weather_df.dtypes)

# Drop non-numeric columns or convert them if appropriate
# Assuming 'Source' is the problematic non-numeric column
weather_df_numeric = weather_df.select_dtypes(include=[float, int])

# Drop rows with missing values
weather_df_numeric.dropna(inplace=True)

# Visualize the distribution of a parameter (e.g., 'Mean')
plt.boxplot(weather_df_numeric['Mean'])
plt.show()

# Removing outliers (e.g., values beyond 3 standard deviations from the mean)
weather_df_numeric = weather_df_numeric[(np.abs(stats.zscore(weather_df_numeric['Mean'])) < 3)]

# Export the cleaned dataset to a CSV file
cleaned_file_path = 'cleaned_weather_dataset.csv'
weather_df_numeric.to_csv(cleaned_file_path, index=False)

# Compute the correlation matrix
correlation_matrix = weather_df_numeric.corr()
print(correlation_matrix)

# Define the predictors and the target variable
# For demonstration purposes, let's use 'Year' to predict 'Mean' temperature
X = weather_df_numeric[['Year']]  # example predictor
y = weather_df_numeric['Mean']  # target variable

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Predictions
predictions = model.predict(X)

# Model coefficients
print(f"Coefficient: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")


import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
weather_df = pd.read_csv('cleaned_weather_dataset.csv')

# Line Chart: Temperature Trends Over Years
plt.figure(figsize=(10, 6))
sns.lineplot(data=weather_df, x='Year', y='Mean')
plt.title('Temperature Trends Over Years')
plt.xlabel('Year')
plt.ylabel('Mean Temperature')
plt.grid(True)
plt.show()


# Histogram: Distribution of Mean Temperatures
plt.figure(figsize=(10, 6))
sns.histplot(weather_df['Mean'], bins=30, kde=True)
plt.title('Distribution of Mean Temperatures')
plt.xlabel('Mean Temperature')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Create a new column for decade
weather_df['Decade'] = (weather_df['Year'] // 10) * 10

# Group by Decade and calculate mean temperature
decade_means = weather_df.groupby('Decade')['Mean'].mean().reset_index()

# Bar Chart: Average Temperature by Decade
plt.figure(figsize=(10, 6))
sns.barplot(data=decade_means, x='Decade', y='Mean')
plt.title('Average Temperature by Decade')
plt.xlabel('Decade')
plt.ylabel('Average Mean Temperature')
plt.grid(True)
plt.show()

