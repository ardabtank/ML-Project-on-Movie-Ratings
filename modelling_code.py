# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from scipy import stats

# Loading the dataset
df = pd.read_csv('movies_dataset.csv')

# Data overview
print("Data Information:")
print(df.info())
print("\nFirst few rows:")
print(df.head())

# Function to convert 'Duration' from 'X hours Y minutes' to total minutes.
def convert_duration(duration_str):
    if pd.isnull(duration_str):
        return np.nan
    pattern = r'(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?'
    match = re.match(pattern, duration_str.strip())
    if not match:
        return np.nan
    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2)) if match.group(2) else 0
    total_minutes = hours * 60 + minutes
    return total_minutes

# Applying the conversion to the 'Duration' column
df['Duration'] = df['Duration'].apply(convert_duration)

# Converting 'No of Persons Voted' to numeric
df['No of Persons Voted'] = pd.to_numeric(df['No of Persons Voted'], errors='coerce')

# Converting 'Release Date' to datetime and extract 'Release Year'
df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
df['Release Year'] = df['Release Date'].dt.year

# Handling missing values
df['Rating'].fillna(df['Rating'].median(), inplace=True)
df['No of Persons Voted'].fillna(df['No of Persons Voted'].median(), inplace=True)
df['Duration'].fillna(df['Duration'].median(), inplace=True)
df['Genres'].fillna('Unknown', inplace=True)
df['Directed by'].fillna('Unknown', inplace=True)
df['Written by'].fillna('Unknown', inplace=True)

# Simplifying 'Genres' to the first listed genre
df['Genres'] = df['Genres'].apply(lambda x: x.split(',')[0] if isinstance(x, str) else x)

# Encoding categorical variables
label_encoder = LabelEncoder()
df['Genres'] = label_encoder.fit_transform(df['Genres'])
df['Directed by'] = label_encoder.fit_transform(df['Directed by'])
df['Written by'] = label_encoder.fit_transform(df['Written by'])

# Creating a mapping of encoded genres to their original string values
genre_mapping = dict(zip(df['Genres_encoded'], df['Genres']))

# Printing the mapping of encoded values to genres (unique pairs)
print("Mapping of encoded genres to original genre strings:")
for encoded_value, genre in genre_mapping.items():
    print(f"{encoded_value}: {genre}")

# Preparing data for modeling
features = ['Duration', 'No of Persons Voted', 'Release Year', 
            'Genres_encoded', 'Directed by', 'Written by']
X = df[features]
y = df['Rating']

# Descriptive Statistics
descriptive_stats = df[['Rating', 'No of Persons Voted', 'Duration']].describe()
print("\nDescriptive Statistics:")
print(descriptive_stats)

# Saving descriptive statistics
count_rating = descriptive_stats.loc['count', 'Rating']
mean_rating = descriptive_stats.loc['mean', 'Rating']
std_rating = descriptive_stats.loc['std', 'Rating']
min_rating = descriptive_stats.loc['min', 'Rating']
max_rating = descriptive_stats.loc['max', 'Rating']

count_votes = descriptive_stats.loc['count', 'No of Persons Voted']
mean_votes = descriptive_stats.loc['mean', 'No of Persons Voted']
std_votes = descriptive_stats.loc['std', 'No of Persons Voted']
min_votes = descriptive_stats.loc['min', 'No of Persons Voted']
max_votes = descriptive_stats.loc['max', 'No of Persons Voted']

count_duration = descriptive_stats.loc['count', 'Duration']
mean_duration = descriptive_stats.loc['mean', 'Duration']
std_duration = descriptive_stats.loc['std', 'Duration']
min_duration = descriptive_stats.loc['min', 'Duration']
max_duration = descriptive_stats.loc['max', 'Duration']

# Distribution of Movie Ratings
plt.figure(figsize=(8, 6))
sns.histplot(df['Rating'], kde=True, bins=30)
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.savefig('rating_distribution.png')
plt.show()

# Top 10 Most Common Genres
plt.figure(figsize=(10, 6))
df['Genres_encoded'].value_counts().nlargest(10).plot(kind='bar')
plt.title('Top 10 Most Common Genres')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.savefig('genre_frequency.png')
plt.show()

# Correlation Matrix
corr_features = df[features + ['Rating']]
plt.figure(figsize=(10, 8))
sns.heatmap(corr_features.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.show()

# Spliting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and train the Random Forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Feature Importance
importances = rf_model.feature_importances_
feature_names = features

# Ploting feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.savefig('feature_importance.png')
plt.show()

# Hypothesis Testing for Hypothesis 1
# Hypothesis: Movies directed by prolific directors (more than 5 movies) have higher average ratings

# Creating a dataframe of director counts
director_counts = df['Directed by'].value_counts()

# Defining prolific directors as those who directed more than 5 movies
prolific_directors = director_counts[director_counts > 5].index.tolist()

# Adding a column indicating whether the director is prolific
df['Prolific Director'] = df['Directed by'].apply(lambda x: 'Prolific' if x in prolific_directors else 'Non-Prolific')

# Grouping data by prolific status
prolific_group = df[df['Prolific Director'] == 'Prolific']['Rating']
non_prolific_group = df[df['Prolific Director'] == 'Non-Prolific']['Rating']

# Performing independent samples t-test
t_statistic, p_value = stats.ttest_ind(prolific_group, non_prolific_group, equal_var=False)

print(f"\nHypothesis Test for Hypothesis 1:")
print(f"t-statistic: {t_statistic:.2f}")
print(f"p-value: {p_value:.4f}")

# Deciding whether to reject the null hypothesis
alpha = 0.05
if p_value < alpha:
    decision = "Reject the null hypothesis"
else:
    decision = "Fail to reject the null hypothesis"

print(f"Decision: {decision}")
if decision == "Reject the null hypothesis":
    interpretation = "There is significant evidence to suggest that movies directed by prolific directors have higher average ratings."
else:
    interpretation = "There is not enough evidence to suggest that movies directed by prolific directors have higher average ratings."

print(f"Interpretation: {interpretation}")

# Saving test results
t_statistic_value = t_statistic
p_value_formatted = f"{p_value:.4f}"
decision_text = decision
interpretation_text = interpretation
