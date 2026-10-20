# Run from project root: Rscript script/analysis/regression/runner.R

library(data.table)
library(yaml)

root <- normalizePath(".", winslash = "/")
source(file.path(root, "script/analysis/regression/trial.R"))
source(file.path(root, "script/analysis/regression/item.R"))
source(file.path(root, "script/analysis/regression/regression.R"))

experiment_config <- yaml::read_yaml(
  file.path(root, "config/experiment.yml"),
  merge.precedence = "override"
)
analysis_config <- yaml::read_yaml(file.path(root, "config/regression.yml"))
modes <- unlist(analysis_config$run$priority_modes, use.names = FALSE)
predictors <- c(
  unlist(analysis_config$regression$affective_predictors, use.names = FALSE),
  unlist(analysis_config$regression$visual_predictors, use.names = FALSE)
)

parse_model_metadata <- function(model_id) {
  parts <- strsplit(model_id, "_", fixed = TRUE)[[1]]

  if (parts[1] == "random") {
    size <- c(vits16 = "small", vitb16 = "base", vitl16 = "large")[[parts[2]]]
    seed <- as.integer(sub("^seed", "", parts[3]))
    return(list(training = "random", size = size, seed = seed))
  }

  list(training = parts[1], size = parts[2], seed = NA_integer_)
}

coefficient_tables <- list()
statistics_tables <- list()
k <- 0L

for (sweep_id in unlist(analysis_config$run$sweeps, use.names = FALSE)) {
  sweep <- experiment_config$experiment_sweeps[[sweep_id]]
  protocol_id <- sweep$protocol
  protocol <- experiment_config$protocol_presets[[protocol_id]]

  for (model_id in sweep$models) {
    model_metadata <- parse_model_metadata(model_id)

    for (imageset in names(sweep$imagesets)) {
      for (subset in sweep$imagesets[[imageset]]) {
        experiment_id <- paste(model_id, imageset, subset, sep = "__")
        result_path <- file.path(root, "results", protocol_id, experiment_id, "results.csv")
        if (!file.exists(result_path)) {
          cat("Skipping unfinished experiment:", protocol_id, experiment_id, "\n")
          next
        }

        cat("Analyzing:", protocol_id, experiment_id, "\n")
        snapshot_path <- file.path(root, "results", protocol_id, experiment_id, "config_snapshot.yml")
        snapshot <- yaml::read_yaml(snapshot_path)
        experiment <- list(
          protocol = protocol_id, id = experiment_id, model = model_id,
          imageset = imageset, subset = subset, trial_design = snapshot$trial_design
        )
        item_dir <- file.path(root, "data", "item_priority", protocol_id, experiment_id)
        dir.create(item_dir, recursive = TRUE, showWarnings = FALSE)

        # Load trial results only if an item-level cache is missing.
        result_data <- NULL
        covariates <- NULL
        item_paths <- file.path(item_dir, paste0(modes, ".csv"))
        if (any(!file.exists(item_paths))) {
          result_data <- read_trial_results(experiment, root)
          covariates <- read_item_covariates(imageset, root)
        }

        for (mode in modes) {
          item_path <- file.path(item_dir, paste0(mode, ".csv"))

          if (file.exists(item_path)) {
            item <- data.table::fread(item_path, colClasses = list(character = "Image_id"))
          } else {
            trial <- make_trial_priority(result_data, mode)
            item <- make_item_priority(trial, covariates)
            data.table::fwrite(item, item_path)
            rm(trial)
          }

          variants <- unique(item[, .(Metric, Variant_id)])
          for (i in seq_len(nrow(variants))) {
            metric <- variants$Metric[i]
            variant_id <- variants$Variant_id[i]
            selected <- item[Metric == metric & Variant_id == variant_id]
            fit <- fit_priority_regression(selected, predictors)

            meta <- data.table::data.table(
              Priority_mode = mode, Layout = protocol$layout,
              Protocol = protocol_id, Sweep = sweep_id,
              Imageset = imageset, Subset = subset,
              Model = model_id,
              Training = model_metadata$training,
              Size = model_metadata$size,
              Seed = model_metadata$seed,
              Metric = metric, Variant_id = variant_id
            )
            k <- k + 1L
            coefficient_tables[[k]] <- cbind(meta[rep(1L, nrow(fit$coefficients))], fit$coefficients)
            statistics_tables[[k]] <- cbind(meta, fit$statistics)
          }
        }
        result_data <- NULL
        covariates <- NULL
      }
    }
  }
}

out_dir <- file.path(root, "analysis_results", "regression")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

if (k == 0L) stop("No completed experiments found in the configured sweeps.")
coefficients <- data.table::rbindlist(coefficient_tables, use.names = TRUE)
statistics <- data.table::rbindlist(statistics_tables, use.names = TRUE)
coefficients <- correct_pvalues(coefficients, analysis_config$regression$correction)

data.table::fwrite(coefficients, file.path(out_dir, "coefficients.csv"))
data.table::fwrite(statistics, file.path(out_dir, "model_statistics.csv"))
yaml::write_yaml(
  list(
    priority_modes = modes,
    affective_predictors = analysis_config$regression$affective_predictors,
    visual_predictors = analysis_config$regression$visual_predictors,
    correction = analysis_config$regression$correction,
    confidence_interval = "95% unadjusted",
    regressions = nrow(statistics)
  ),
  file.path(out_dir, "metadata.yml")
)
cat("Finished:", nrow(statistics), "regressions ->", out_dir, "\n")
