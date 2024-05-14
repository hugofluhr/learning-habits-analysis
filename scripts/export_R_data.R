# Small script to load preproc data from OSF and export relevant data as CSV files

og_file <- '/Users/hugofluhr/phd_local/data/RR/preproc/habits_tasks/01_analyses_RP_task.RData'
dest_dir <- '/Users/hugofluhr/phd_local/data/RR/preproc/habits_tasks/'
# Load the data
load(og_file)

# Get all variable names in the global environment
all_vars <- ls()

# Filter variables starting with "data"
data_vars <- grep("^data", all_vars, value = TRUE)

# Loop through each data variable and save as CSV
for (var_name in data_vars) {
  # Get the data associated with the variable
  data <- get(var_name)
  
  # Generate the file name (e.g., data_variable_name.csv)
  file_name <- file.path(dest_dir, paste0(var_name, ".csv"))
  
  # Save data as CSV file with headers
  write.csv(data, file = file_name, row.names = FALSE)
}
