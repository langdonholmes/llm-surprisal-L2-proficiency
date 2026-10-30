# Accuracy network (D54): the approved 15-node network's readouts, in the same
# tables as the corrector arm, so the two can be compared edge by edge. Runs the
# code chunks of _network-body.qmd with nothing added, leaving out the chunks that
# save figures, the MTLD hold-out and the prose summaries. Writes nothing to fig/.
#
#   Rscript src/3-surprisal-decomposition/accuracy-network/approved_readouts.R

grDevices::pdf(NULL)  # the plotting chunks draw to nowhere

drop_var <- NULL
extra_nodes <- NULL
extra_nodes_file <- NULL
run_dir <- "results/study3/ellipse/accuracy_network/approved"

body_lines <- readLines("src/2-nomological-network/_network-body.qmd")
chunk_starts <- which(body_lines == "```{r}")
chunk_ends <- vapply(chunk_starts, function(start) start + which(body_lines[-(1:start)] == "```")[1], numeric(1))
chunk_label <- function(chunk) sub("#\\| label: ", "", grep("^#\\| label: ", chunk, value = TRUE)[1])

skipped_labels <- c("boot-drop-bridge", "gradation-note", "summary-values", "summary-communities",
                    "summary-importance", "summary-importance-caveat", "summary-cfa-table",
                    "summary-network-compare")

for (i in seq_along(chunk_starts)) {
  chunk <- body_lines[(chunk_starts[i] + 1):(chunk_ends[i] - 1)]
  label <- chunk_label(chunk)
  if (label %in% skipped_labels || startsWith(label, "save-")) next
  cat("==", label, format(Sys.time(), "%H:%M:%S"), "\n")
  eval(parse(text = chunk), envir = globalenv())
}
