# Join trial results to their image IDs, then compute priority.

read_trial_results <- function(experiment, root) {
  design <- experiment$trial_design
  matrix_id <- sprintf(
    "%s__%s__%sitems__%strials__seed%s",
    experiment$imageset, experiment$subset,
    design$n_items_per_trial, design$n_trials, design$sampling_seed
  )

  matrix_path <- file.path(root, "data", "design_matrix", paste0(matrix_id, ".csv"))
  result_path <- file.path(root, "results", experiment$protocol, experiment$id, "results.csv")

  matrix <- data.table::fread(matrix_path, colClasses = list(character = c("Constituent_0", "Constituent_1")))
  results <- data.table::fread(result_path)

  positions <- data.table::melt(
    matrix,
    id.vars = "Trial_id",
    measure.vars = grep("^Constituent_", names(matrix), value = TRUE),
    variable.name = "Constituent",
    value.name = "Image_id"
  )
  positions[, Position_index := as.integer(sub("Constituent_", "", Constituent))]

  results <- merge(
    results,
    positions[, .(Trial_id, Position_index, Image_id)],
    by = c("Trial_id", "Position_index"),
    sort = FALSE
  )
  results[, Image_id := as.character(Image_id)]
  results
}

make_trial_priority <- function(results, mode) {
  trial <- data.table::copy(results)

  if (mode == "absolute") {
    trial[, Priority := Value]
  } else if (mode == "relative") {
    trial[, Total := sum(Value), by = .(Trial_id, Metric, Variant_id)]
    trial[, Priority := data.table::fifelse(Total == 0, NA_real_, Value / Total)]
    trial[, Total := NULL]
  } else {
    stop("Unknown priority mode: ", mode)
  }

  data.table::setcolorder(trial, c("Trial_id", "Position_index", "Image_id", "Metric", "Variant_id", "Value", "Priority"))
  trial
}
