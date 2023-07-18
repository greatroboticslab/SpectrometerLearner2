# -*- coding: utf-8 -*-
import pandas as pd
import glob
import os
from sklearn.preprocessing import LabelEncoder

class CSVProcessor:
  def __init__(self, file_directory):
    self.file_directory = file_directory 

  def process_csv_files(self):
    # Directory where your CSV files are located
    file_path = self.file_directory + "/*.csv"
    csv_files = glob.glob(file_path)

    # List to store the combined DataFrames
    combined_dfs = []

    # Iterate over the CSV files in the directory
    for file in csv_files:
        df = pd.read_csv(file)
            
        # Create the new DataFrame with the desired columns
        new_df = pd.DataFrame()
        new_df['Subject_subtype_healthy'] = df['Subject'] + '_' + df['Subtype'] + '_' + df['Healthy']
        new_df['Returned_Intensity'] = df['Returned Intensity']

        # Add a column to represent the column index for each "Returned Intensity" value
        new_df['Column_Index'] = new_df.groupby('Subject_subtype_healthy').cumcount() + 1

        # Pivot the DataFrame to have "Returned Intensity" values in separate columns
        new_df = new_df.pivot(index='Subject_subtype_healthy', columns='Column_Index', values='Returned_Intensity').reset_index()

        # Rename the columns
        new_df.columns = ['Subject_subtype_healthy'] + ['Intensity_' + str(i) for i in new_df.columns[1:]]

        combined_dfs.append(new_df)

    # Combine all the DataFrames into a single DataFrame
    combined_df = pd.concat(combined_dfs)
    combined_df_transpose = combined_df.T
    return combined_df_transpose
    # Append the combined DataFrame to the new CSV file
    
  def split(self, print_mapping=False, vector_array=False):
    df_transposed = self.process_csv_files()
    # Assuming df_transposed is your DataFrame after transposing
    target_col = df_transposed.columns[0]  # Assuming the first column is the target variable

    # Extract the target variable (first row)
    target_variable = df_transposed.iloc[0]

    # Remove the first row (target variable) to get the input features
    input_features = df_transposed.drop(index=df_transposed.index[0])

    label_encoder = LabelEncoder()
    encoded_target = label_encoder.fit_transform(target_variable)
    original_categories = label_encoder.inverse_transform(encoded_target)

    if print_mapping:
      # Create a dictionary to map encoded values to their meanings
      encoded_to_meaning = dict(zip(encoded_target, original_categories))
      # Print the mapping of encoded values to their meanings
      print("Encoded Values to Meanings:")
      for encoded_val, meaning in encoded_to_meaning.items():
          print(f"Encoded Value {encoded_val}: {meaning}")
    if vector_array:
        encoded_target = encoded_target.reshape(-1, 1)
    input_features = input_features.T
    # Return the target variable and input features
    return input_features, encoded_target