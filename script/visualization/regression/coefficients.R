# Coefficient plots: layouts as rows, imagesets as columns.
# Random, SimCLR, SLIP and CLIP are compared on the same axes.

plot_coefficients <- function(coefficients, sweep_ids, priority_mode,
                              layout_levels, imageset_levels, predictors,
                              metric, variant_id) {
  library(ggplot2)

  df <- data.table::copy(coefficients[
    Sweep %in% sweep_ids &
      Priority_mode == priority_mode &
      Predictor %in% predictors &
      Metric == metric & Variant_id == variant_id
  ])
  if (nrow(df) == 0) return(NULL)

  df[, Layout := factor(Layout, levels = layout_levels)]
  df[, Imageset := factor(Imageset, levels = imageset_levels)]
  df[, Predictor := factor(Predictor, levels = predictors)]
  df[, Training := factor(
    Training, levels = c("random", "simclr", "slip", "clip"),
    labels = c("Random", "SimCLR", "SLIP", "CLIP")
  )]
  df[, Size := factor(
    Size, levels = c("small", "base", "large"),
    labels = c("Small", "Base", "Large")
  )]

  # Four training groups, each with three size positions.
  predictor_x <- setNames(seq_along(predictors), predictors)
  model_offset <- c(Random = -0.27, SimCLR = -0.09, SLIP = 0.09, CLIP = 0.27)
  size_offset <- c(Small = -0.055, Base = 0, Large = 0.055)
  df[, X := predictor_x[as.character(Predictor)] +
            model_offset[as.character(Training)] +
            size_offset[as.character(Size)]]

  df[, Fill := ifelse(Significant %in% TRUE, as.character(Training), "n.s.")]
  model_colors <- c(Random = "#777777", SimCLR = "#D55E00",
                    SLIP = "#0072B2", CLIP = "#009E73")

  ggplot(df, aes(x = X, y = Beta, colour = Training, shape = Size, fill = Fill)) +
    geom_hline(yintercept = 0, linetype = "dashed", colour = "grey65", linewidth = 0.35) +
    geom_point(size = 2.5, stroke = 0.8, na.rm = TRUE) +
    facet_grid(Layout ~ Imageset, drop = FALSE) +
    scale_x_continuous(
      breaks = seq_along(predictors), labels = predictors,
      limits = c(0.55, length(predictors) + 0.45)
    ) +
    scale_colour_manual(name = "Model", values = model_colors) +
    scale_shape_manual(name = "Size", values = c(Small = 21, Base = 24, Large = 22)) +
    scale_fill_manual(values = c(model_colors, "n.s." = "white"), guide = "none") +
    guides(
      shape = guide_legend(order = 1, override.aes = list(fill = "white", colour = "black")),
      colour = guide_legend(
        order = 2,
        override.aes = list(
          shape = 21,
          fill = unname(model_colors[levels(droplevels(df$Training))])
        )
      )
    ) +
    labs(
      title = paste(variant_id, "—", priority_mode, "priority"),
      x = NULL, y = "Standardized regression coefficient (β)",
      caption = "Open symbols: n.s.  |  Filled symbols: adjusted p < .05"
    ) +
    theme_bw(base_size = 11) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      strip.background = element_rect(fill = "grey93", colour = "white"),
      strip.text.x = element_text(face = "bold", size = 10),
      strip.text.y = element_text(angle = 0, size = 8),
      axis.text.x = element_text(size = 9),
      legend.position = "bottom",
      legend.box = "horizontal",
      plot.caption = element_text(hjust = 1, size = 9)
    )
}
