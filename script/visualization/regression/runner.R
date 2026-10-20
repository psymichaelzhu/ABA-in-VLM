# Run from project root: Rscript script/visualization/regression/runner.R

library(data.table)
library(yaml)
library(ggplot2)

root <- normalizePath(".", winslash = "/")
source(file.path(root, "script/visualization/regression/coefficients.R"))

experiment_config <- yaml::read_yaml(
  file.path(root, "config/experiment.yml"),
  merge.precedence = "override"
)
analysis_config <- yaml::read_yaml(file.path(root, "config/regression.yml"))
coefficients <- data.table::fread(file.path(root, "analysis_results/regression/coefficients.csv"))

sweep_ids <- unlist(analysis_config$run$sweeps, use.names = FALSE)
layout_levels <- unique(vapply(sweep_ids, function(id) {
  protocol_id <- experiment_config$experiment_sweeps[[id]]$protocol
  experiment_config$protocol_presets[[protocol_id]]$layout
}, character(1)))
imageset_levels <- unique(unlist(lapply(sweep_ids, function(id) {
  names(experiment_config$experiment_sweeps[[id]]$imagesets)
}), use.names = FALSE))
predictors <- unlist(analysis_config$visualization$predictors, use.names = FALSE)
metric <- analysis_config$visualization$metric
variant_id <- analysis_config$visualization$variant_id

figure_dir <- file.path(root, "analysis_results", "regression", "figures")
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

for (mode in unlist(analysis_config$run$priority_modes, use.names = FALSE)) {
  plot <- plot_coefficients(
    coefficients, sweep_ids, mode, layout_levels, imageset_levels,
    predictors, metric, variant_id
  )
  if (is.null(plot)) next

  name <- paste(metric, variant_id, mode, "coefficients", sep = "__")
  ggplot2::ggsave(file.path(figure_dir, paste0(name, ".pdf")), plot, width = 16, height = 14)
  ggplot2::ggsave(file.path(figure_dir, paste0(name, ".png")), plot, width = 16, height = 14, dpi = 220)
  cat("Saved:", name, "\n")
}
