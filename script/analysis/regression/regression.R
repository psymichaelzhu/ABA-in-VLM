# Fit item-level regressions and correct affective predictor p-values.

fit_priority_regression <- function(item, predictors) {
  needed <- c("Priority", predictors)
  data <- item[stats::complete.cases(item[, ..needed]), ..needed]
  data[, (needed) := lapply(.SD, function(x) as.numeric(scale(x))), .SDcols = needed]

  fit <- stats::lm(
    stats::reformulate(predictors, response = "Priority"),
    data = data
  )
  s <- summary(fit)
  terms <- predictors
  estimates <- s$coefficients[terms, , drop = FALSE]
  intervals <- stats::confint(fit, parm = terms)

  coefficients <- data.table::data.table(
    Predictor = terms,
    Beta = estimates[, "Estimate"],
    SE = estimates[, "Std. Error"],
    CI_low = intervals[, 1],
    CI_high = intervals[, 2],
    P_raw = estimates[, "Pr(>|t|)"]
  )
  statistics <- data.table::data.table(
    N = stats::nobs(fit),
    R2 = s$r.squared,
    Adjusted_R2 = s$adj.r.squared
  )
  list(coefficients = coefficients, statistics = statistics)
}

correct_pvalues <- function(coefficients, config) {
  coefficients[, `:=`(
    P_adjusted = NA_real_,
    Significant = NA,
    Correction_family = NA_character_,
    N_tests_in_family = NA_integer_
  )]

  scope <- unlist(config$scope, use.names = FALSE)
  targets <- unlist(config$predictors, use.names = FALSE)
  indices <- which(coefficients$Predictor %in% targets & !is.na(coefficients$P_raw))

  coefficients[indices, `:=`(
    P_adjusted = stats::p.adjust(P_raw, method = config$method),
    N_tests_in_family = .N,
    Correction_family = paste(unlist(.BY, use.names = FALSE), collapse = "__")
  ), by = scope]

  coefficients[, Significant := !is.na(P_adjusted) & P_adjusted < config$alpha]
  coefficients[is.na(P_adjusted), Significant := NA]
  coefficients[]
}
