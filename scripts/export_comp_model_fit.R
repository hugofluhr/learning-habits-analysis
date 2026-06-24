modeling_data_dir <- "/Users/hugofluhr/phd_local/data/LearningHabits/dev_sample/modeling_data"

configs <- list(
  list(
    R_file_path      = file.path(modeling_data_dir, "data_RP_MRI_model_RL_CK_reduced_Q1.RData"),
    output_directory = file.path(modeling_data_dir, "2026-02-09"),
    test_var         = "data_test_RL_CK2_extended",
    train_var        = "data_train_RL_CK2_extended",
    suffix           = "compmodelQ1"
  ),
  list(
    R_file_path      = file.path(modeling_data_dir, "data_RP_MRI_model_RL_CK_reduced_20260206.RData"),
    output_directory = file.path(modeling_data_dir, "2026-02-06"),
    test_var         = "data_test_RL_CK2_extended",
    train_var        = "data_train_RL_CK2_extended",
    suffix           = "compmodel260206"
  ),
  list(
    R_file_path      = file.path(modeling_data_dir, "data_RP_MRI_model_RL_CK_reduced_Q5_Hpretest.RData"),
    output_directory = file.path(modeling_data_dir, "2026-05-reduced"),
    test_var         = "data_test_RL_CK2_extended",
    train_var        = "data_train_RL_CK2_extended",
    suffix           = "compmodel052026reduced"
  ),
  list(
    R_file_path      = file.path(modeling_data_dir, "data_RP_MRI_model_RL_CK_omega_Q5_Hpretest.RData"),
    output_directory = file.path(modeling_data_dir, "2026-05-combined"),
    test_var         = "data_test_RL_CK3_extended",
    train_var        = "data_train_RL_CK3_extended",
    suffix           = "compmodel052026combined"
  )
)

split_and_save <- function(dataframe, dfname, output_directory) {
  subject_ids <- unique(dataframe[[1]])
  for (subject in subject_ids) {
    subset_df <- subset(dataframe, dataframe[[1]] == subject)
    filename <- tolower(paste0(subject, "_", dfname, ".csv"))
    write.csv(subset_df, file = file.path(output_directory, filename), row.names = FALSE)
    cat("Saved:", filename, "\n")
  }
}

for (cfg in configs) {
  cat("\n=== Processing:", cfg$R_file_path, "===\n")
  dir.create(cfg$output_directory, showWarnings = FALSE, recursive = TRUE)
  env <- new.env()
  load(cfg$R_file_path, envir = env)
  split_and_save(env[[cfg$test_var]],  paste0("test_",     cfg$suffix), cfg$output_directory)
  split_and_save(env[[cfg$train_var]], paste0("learning_", cfg$suffix), cfg$output_directory)
}
