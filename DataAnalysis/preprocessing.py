import pandas as pd
import os
import ast
import numpy as np
from utilities.utility_preprocessing import preprocess_data

# here is where we load and preprocess the data
behavioral_list = ['ReactTime', 'Reward', 'BestOption', 'KeyResponse', 'SetSeen ', 'OptionRwdMean', 'Phase']

other_data_list = ['OptionOrder', 'EstOptionA', 'OptionAConfidence', 'EstOptionS', 'OptionSConfidence', 'EstOptionK',
                   'OptionKConfidence', 'EstOptionL', 'OptionLConfidence', 'Gender', 'Ethnicity', 'Race', 'Age',
                   'Big5O', 'Big5C', 'Big5E', 'Big5A', 'Big5N', 'studyResultId', 'BISScore', 'CESDScore',
                   'ESIBF_disinhScore', 'ESIBF_aggreScore', 'ESIBF_sScore', 'PSWQScore', 'STAITScore', 'STAISScore']

numeric_list = ['studyResultId', 'Big5O', 'Big5C', 'Big5E', 'Big5A', 'Big5N', 'BISScore', 'CESDScore',
                'ESIBF_disinhScore', 'ESIBF_aggreScore', 'ESIBF_sScore', 'PSWQScore', 'STAITScore', 'STAISScore']

dict_list = ['Gender', 'Ethnicity', 'Race', 'Age']


# Replace this with the path to the directory containing your 'comp-result_*' folders
main_folder_directory = './Data/jatos_results_data_20240324154855'


# Define a function to safely parse the dictionary-like string and extract the value
def extract_value(dict_like_str):
    try:
        # Safely evaluate the string as a dictionary
        val = ast.literal_eval(dict_like_str)
        # Return the value associated with the key 'Q0'
        return val.get('Q0', None)  # Replace 'Q0' with your actual key
    except (ValueError, SyntaxError):
        # In case the string is not a valid dictionary-like string, return NaN or some default value
        return pd.NA

# This is the function that processes the data for a single participant
def process_participant_data(participant_path):
    # Initialize an empty list to store the DataFrames
    dfs = []

    # Loop through each folder in the base directory
    for folder_name in os.listdir(participant_path):
        # Construct the full path to the folder
        folder_path = os.path.join(participant_path, folder_name)

        # Check if it is indeed a directory
        if os.path.isdir(folder_path):
            # In each folder, list the .txt files
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.txt'):
                    # Construct the full file path
                    file_path = os.path.join(folder_path, file_name)

                    # Read the JSON file into a DataFrame
                    df = pd.read_json(file_path, lines=True)

                    # Append the DataFrame to the list
                    dfs.append(df)

    dfs[0] = dfs[0].apply(lambda x: x.explode() if x.name in behavioral_list else x)
    dfs[0]['Condition'] = 'Baseline'
    dfs[11] = dfs[11].apply(lambda x: x.explode() if x.name in behavioral_list else x)
    dfs[11]['Condition'] = 'Frequency'

    # Combine all the DataFrames into one
    combined_df = pd.concat(dfs, ignore_index=True)

    # Select only the columns we need
    kept_columns = behavioral_list + other_data_list + ['Condition']
    combined_df = combined_df[kept_columns]

    # fill in values and drop NA
    combined_df[other_data_list] = combined_df[other_data_list].fillna(method='bfill').fillna(method='ffill')
    combined_df = combined_df.dropna()

    # change the data type of the columns
    combined_df[numeric_list] = (
        combined_df[numeric_list].applymap(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x))

    # Apply the function to the relevant column
    for col in dict_list:
        combined_df[col] = combined_df[col].apply(extract_value)

    return combined_df

# The below two functions are used only for the knowledge questionnaires
# Function to reorder and rename option and confidence columns based on 'OptionOrder'
def reorder_and_rename(row):
    # Reorder options
    order_to_option_col = {order: col for col, order in zip(options_to_pos.keys(), row['OptionOrder'])}
    reordered_options = [row[order_to_option_col[order]] for order in sorted(row['OptionOrder'])]

    # Reorder confidences
    order_to_confidence_col = {order: col for col, order in zip(confidences_to_pos.keys(), row['OptionOrder'])}
    reordered_confidences = [row[order_to_confidence_col[order]] for order in sorted(row['OptionOrder'])]

    return pd.Series(reordered_options + reordered_confidences, index=new_option_cols + new_confidence_cols)

# Function to clear nonsense data in the knowledge questionnaires
def confidence_data_preprocessing(data):
    # first, convert all "," to "."
    data = data.str.replace(',', '.')
    # if the value contains a mixture of numbers and letters, extract the numbers
    data = data.str.extract('(\d+.\d+|\d+)', expand=False)
    # if the value is still a string, replace it with NaN
    data = pd.to_numeric(data, errors='coerce')
    # then, convert all the values to float
    data = data.astype(float)
    return data

# List to store the DataFrames for all participants
all_participants_dfs = []

# Iterate over each subfolder in the main folder
for participant_folder_name in os.listdir(main_folder_directory):
    participant_folder_path = os.path.join(main_folder_directory, participant_folder_name)

    # Check if this path is indeed a folder
    if os.path.isdir(participant_folder_path):
        # Process the participant folder and collect the DataFrame
        participant_df = process_participant_data(participant_folder_path)
        all_participants_dfs.append(participant_df)

# Combine all participant DataFrames into one
all_data_combined = pd.concat(all_participants_dfs, ignore_index=True)

# move participant id to the first column
# Pop the column
col = all_data_combined.pop('studyResultId')

# Insert it at the start
all_data_combined.insert(0, 'studyResultId', col)

# reset the participant id column
subject_id = len(all_data_combined) // len(participant_df) + 1
ids = np.arange(1, subject_id)
ids = np.repeat(ids, len(participant_df))
all_data_combined['studyResultId'] = ids
all_data_combined = all_data_combined.rename(columns={'studyResultId': 'Subnum'})


confidence_lists = ['Subnum', 'OptionOrder', 'EstOptionA', 'OptionAConfidence', 'EstOptionS', 'OptionSConfidence',
                    'EstOptionK', 'OptionKConfidence', 'EstOptionL', 'OptionLConfidence']

confidence_data = all_data_combined[confidence_lists].groupby('Subnum').first().reset_index()

# Mapping of old columns to their positions (1-based for readability)
options_to_pos = {'EstOptionA': 1, 'EstOptionS': 2, 'EstOptionK': 3, 'EstOptionL': 4}
confidences_to_pos = {'OptionAConfidence': 1, 'OptionSConfidence': 2, 'OptionKConfidence': 3, 'OptionLConfidence': 4}

# New column names after reordering
new_option_cols = ['EstA', 'EstB', 'EstC', 'EstD']
new_confidence_cols = ['A_Confidence', 'B_Confidence', 'C_Confidence', 'D_Confidence']

# Apply the function to reorder and rename columns for each row
reordered_df = confidence_data.apply(reorder_and_rename, axis=1)
reordered_df.insert(0, 'Subnum', confidence_data['Subnum'])

# explode the data
columns_to_explode = new_option_cols + new_confidence_cols
knowledge_df = reordered_df.explode(columns_to_explode)

# add a column to indicate phase from 1 to 7
knowledge_df['Phase'] = np.tile(np.arange(1, 8), len(knowledge_df) // 7)

# clear nonsense data
knowledge_df[new_option_cols] = knowledge_df[new_option_cols].apply(confidence_data_preprocessing)
knowledge_df[new_confidence_cols] = knowledge_df[new_confidence_cols].apply(confidence_data_preprocessing)

# if the estimated value is greater than 10, replace it with NaN
knowledge_df[new_option_cols] = knowledge_df[new_option_cols].apply(lambda x: x.where(x <= 2))
# if the confidence value is not between 1 and 10, replace it with NaN
knowledge_df[new_confidence_cols] = knowledge_df[new_confidence_cols].apply(lambda x: x.where((x >= 1) & (x <= 10)))



# # Call the function to preprocess the data
# data, knowledge = preprocess_data(main_folder_directory, behavioral_list, other_data_list, numeric_list, dict_list,
#                                   estimate=True, cutoff=2)
