import pandas as pd

def turn_value_counts_into_percentages(value_counts_series):
  value_counts_series = value_counts_series.sort_values(ascending=False)
  sum_values = value_counts_series.sum()
  value_percentages = [x/sum_values for x in value_counts_series]
  return value_percentages

def add_binary_2_value_counts(df):
  value_percentage_df_max = df["count"].max()
  df["binary"] = df["count"].map(lambda x: 1 if x==value_percentage_df_max else 0)
  return df

def calculate_percentages_of_features(data):
  value_percentage_dict = {}
  for feature in data:
    value_counts = data[feature].value_counts()
    value_percentages = turn_value_counts_into_percentages(value_counts)
    d = {feature: value_counts.index,
         "count": value_counts.values,
         "percentage": value_percentages}

    value_percentage_df = pd.DataFrame(d)



    # Add a binary column to the counts
    value_percentage_df = add_binary_2_value_counts(value_percentage_df)

    value_percentage_dict[feature] = value_percentage_df
  #print(f"There are {len(data.columns)} object type columns in the data and {len(value_percentage_dict.keys())} object columns in the value_percentage_dict")
  return value_percentage_dict

def check_correlation(feature_name, train_data, all_data):
  reintroducing_price_df = train_data.join(all_data['SalePrice'], how='left')
  feature_correlation = reintroducing_price_df[[feature_name, 'SalePrice']].corr()
  #print(feature_correlation)  # Check correlation
  return feature_correlation
