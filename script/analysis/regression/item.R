# Aggregate trial priority by item and attach ratings and visual features.

read_item_covariates <- function(imageset, root) {
  stimulus_dir <- file.path(root, "stimuli", imageset)
  ratings <- data.table::fread(
    file.path(stimulus_dir, "ratings.csv"),
    select = c("Image_id", "Arousal", "Valence"),
    colClasses = list(character = "Image_id")
  )
  visual <- data.table::fread(
    file.path(stimulus_dir, "visual_features.csv"),
    colClasses = list(character = "Image_id")
  )
  list(ratings = ratings, visual = visual)
}

make_item_priority <- function(trial, covariates) {
  item <- trial[, .(
    Priority = mean(Priority, na.rm = TRUE),
    N_presentations = .N,
    N_valid = sum(!is.na(Priority))
  ), by = .(Image_id, Metric, Variant_id)]

  item <- merge(item, covariates$ratings, by = "Image_id", all.x = TRUE, sort = FALSE)
  merge(item, covariates$visual, by = "Image_id", all.x = TRUE, sort = FALSE)
}
