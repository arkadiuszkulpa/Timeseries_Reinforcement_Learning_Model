def check_null_population(minimum_percentage, data):
    """
    Identifies features in the dataset where the proportion of null values exceeds a given threshold.

    Parameters:
    - minimum_percentage (float): The minimum percentage threshold (between 0 and 1) for null values to be considered significant.
    - data (DataFrame): The dataset to analyze.

    Returns:
    - dict: A dictionary where keys are feature names and values are dictionaries containing:
        - "number" (int): The count of null values in the feature.
        - "percentage" (float): The percentage of null values in the feature.

    Prints:
    - A summary of features with null values exceeding the threshold.
    """
    null_dict = {}
    for feature in data:
        number_of_nulls = data[feature].isnull().sum()
        number_of_values = data[feature].count()
        number_total = number_of_nulls + number_of_values
        null_percentage = number_of_nulls / number_total

        if null_percentage > minimum_percentage:
            null_dict[feature] = {
                "number": number_of_nulls,
                "percentage": null_percentage
            }

    for feature, stats in null_dict.items():
        print(f"Feature '{feature}' has {stats['number']} nulls, "
              f"which is {(stats['percentage'] * 100):.2f}% of the total.")

    return null_dict


def preprocess_categorical(data, feature_name, mapping_dict):
    """
    Preprocesses a categorical feature by filling missing values and applying a specified mapping.

    Parameters:
    - data (DataFrame): The dataset containing the categorical feature.
    - feature_name (str): The name of the categorical feature to preprocess.
    - mapping_dict (dict): A dictionary mapping old values to new values.

    Returns:
    - None: The function modifies the DataFrame in place.

    Prints:
    - Value counts of the feature before and after preprocessing.
    """
    print(f"Value Counts for '{feature_name}' before preprocessing:")
    print(data[feature_name].value_counts())

    data[feature_name] = data[feature_name].fillna(0)
    data[feature_name] = data[feature_name].replace(mapping_dict)

    # Explicitly infer data types
    data[feature_name] = data[feature_name].infer_objects()

    print(f"Value Counts for '{feature_name}' after preprocessing:")
    print(data[feature_name].value_counts())

def filter_dataframes_by_percentage(df_dict, min_threshold=0.8, max_threshold=0.9):
    return {
        key: df for key, df in df_dict.items()
        if df['percentage'].max() > min_threshold and df['percentage'].max() <= max_threshold
    }

def create_top_x_percentage_dict(data, min_percentage, max_percentage = 1):
  
  data = filter_dataframes_by_percentage(data, min_percentage, max_percentage)
  # data = {key: df for key, df in data.items() if ((df["percentage"] > min_percentage) & (df["percentage"] <= max_percentage)).any()}

  for x in data:
    dominant_value = data[x].query("binary == 1")[x].iloc[0]
    dominant_percentage = data[x].query("binary == 1")["percentage"].iloc[0]
    print(f"For Feature {x}, Value {dominant_value} constitutes {(dominant_percentage*100).round(2)}%")
    #x_dominant = [x[x["binary"]
    #print(f"{x.key} has a dominant value {x.}")

  return data
  #print(len(percentage_dict_95))

import datetime

def replace_year_feature_with_age(feature, new_feature_name, train_data):
  """
  Replaces a given year-based feature in a dataset with the corresponding age and removes the original feature.

  Parameters:
  -----------
  feature : str
      The column name in the dataset representing a year.
  new_feature_name : str
      The name of the new column that will store the computed age.
  data : pandas.DataFrame
      The dataset containing the feature column.

  Returns:
  --------
  pandas.DataFrame
      The modified dataset with the new age feature and the original feature removed.

  Notes:
  ------
  - The function calculates age by subtracting the year values in `feature` from the current year.
  - If `feature` is not found in the dataset, it prints "Feature Not Found".
  - If `feature` is already deleted or missing, it prints "Feature already deleted".
  - Prints the new feature values and the correlation check result using `check_correlation(new_feature_name)`.
  - Assumes `check_correlation` is a predefined function that takes the new feature name as input.
  """
  current_year = datetime.date.today().year
  try:
    train_data[new_feature_name] = current_year - train_data[feature]
  except:
    print("Feature Not Found")
  try:
    train_data = train_data.drop(columns = [feature], axis=1)
  except:
    print("Feature already deleted")
  return train_data



