# Shared visual language for the Study 1 figures (sourced by the analysis qmds).
# One palette keyed to model family so a model is trackable across every figure;
# a consistent theme; and a save helper that emits both PNG and vector PDF.

suppressMessages(library(ggplot2))

# Model family from the config's model key.
model_family <- function(model) {
  dplyr::case_when(
    grepl("^bert",       model) ~ "BERT",
    grepl("^modernbert", model) ~ "ModernBERT",
    grepl("^gpt2",       model) ~ "GPT-2",
    grepl("^olmo2",      model) ~ "OLMo-2",
    grepl("^qwen",       model) ~ "Qwen2.5",
    grepl("^llama",      model) ~ "Llama-3.1",
    TRUE ~ model
  )
}

FAMILY_LEVELS <- c("BERT", "ModernBERT", "GPT-2", "OLMo-2", "Qwen2.5", "Llama-3.1")

# Okabe-Ito (colourblind-safe), one hue per family. Masked families warm, causal cool.
FAMILY_COLORS <- c(
  "BERT"       = "#E69F00",  # orange
  "ModernBERT" = "#CC79A7",  # pink
  "GPT-2"      = "#009E73",  # green
  "OLMo-2"     = "#56B4E9",  # sky
  "Qwen2.5"    = "#0072B2",  # blue
  "Llama-3.1"  = "#D55E00"   # vermillion
)

# Proficiency ordered low (worst/red) -> high (best/blue).
LEVEL_LEVELS <- c("low", "medium", "high")
LEVEL_COLORS <- c(low = "#b2182b", medium = "#ef8a62", high = "#2166ac")

# Shorten instruct suffix for compact point labels.
short_model <- function(model) sub("-instruct", "-inst", model)

theme_study1 <- function(base_size = 12) {
  theme_minimal(base_size = base_size) +
    theme(
      panel.grid.minor = element_blank(),
      plot.title       = element_text(face = "bold"),
      plot.subtitle    = element_text(colour = "grey30"),
      strip.text       = element_text(face = "bold"),
      panel.spacing    = grid::unit(1, "lines"),
      legend.position  = "right"
    )
}

# Save a vector PDF (the dissertation's preferred form) plus a high-resolution
# PNG for contexts that cannot place vector art. 400 dpi is above the 300 dpi
# floor most print submissions ask for, so the PNG is usable as a fallback
# rather than as a review-only proof.
#
# The PDF goes through cairo_pdf rather than the default pdf() device, which is
# Latin-1 only and silently rewrites the em dashes and arrows in the axis labels
# as hyphens (and drops a Greek delta outright).
FIG_DPI <- 400

save_fig <- function(p, name, width, height, dir = RESULTS, dpi = FIG_DPI) {
  ggplot2::ggsave(file.path(dir, paste0(name, ".png")), p,
                  width = width, height = height, dpi = dpi)
  ggplot2::ggsave(file.path(dir, paste0(name, ".pdf")), p,
                  width = width, height = height, device = grDevices::cairo_pdf)
  invisible(p)
}
