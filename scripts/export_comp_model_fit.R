#R_file_path <- "/Users/hugofluhr/phd_local/repositories/Learning-Habits-Behavioral-Analyses/data_RP_MRI_model_RL_CK_reduced_20251218.RData"
R_file_path <- "/Users/hugofluhr/phd_local/data/LearningHabits/dev_sample/modeling_data/data_RP_MRI_model_RL_CK_reduced_20251218.RData"
output_directory <- "/Users/hugofluhr/phd_local/data/LearningHabits/dev_sample/modeling_data/2025-12-18/"

# Load the RData file
load(R_file_path)  # Replace with your actual .RData file path

# Function to split a data frame by the first column and save the splits as CSV
split_and_save <- function(dataframe, dfname, output_directory) {
  # Get the unique values in the first column
  subject_ids <- unique(dataframe[[1]])
  
  # Loop over each unique value, split, and save
  for (subject in subject_ids) {
    # Subset the dataframe based on the first column value
    subset_df <- subset(dataframe, dataframe[[1]] == subject)
    
    # Generate a filename based on the data frame name and the value in the first column
    filename <- tolower(paste0(subject,"_", dfname, ".csv"))
    
    # Save the subset to a CSV file
    write.csv(subset_df, file = file.path(output_directory,filename), row.names = FALSE)
    
    # Print a message to indicate the file has been saved
    cat("Saved:", filename, "\n")
  }
}


# Applying the function to both learning and test data
split_and_save(data_test_RL_CK2_extended, "test_compmodel251218", output_directory)  
split_and_save(data_train_RL_CK2_extended, "learning_compmodel251218", output_directory)