# Token-level text heatmap: render a passage as flowing text with every token
# tinted by a numeric score (surprisal, attention weight, any per-token value).
#
# Sourced by path from the repo root, which is where Quarto starts every render
# (see `execute-dir: project` in _quarto.yml):
#
#     source("src/util/text_heatmap.R")
#
# The layout is monospaced on purpose. A token's tile has to be exactly as wide
# as the glyphs inside it, and that is only true when every character advances
# by the same amount; with a proportional face the tiles and the text drift
# apart as the line grows. Monospace also reads as an instrument display rather
# than as a quotation, which is the right register for a figure whose subject is
# what a model charged for each word.

suppressMessages(library(ggplot2))

# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #

# Greedy wrap of a token sequence into lines of at most `width` characters.
# Returns one row per token: its line, and its start/end/middle column. A token
# longer than `width` gets its own line rather than being split, since these are
# words and a hyphenated break would misrepresent what was scored.
layout_text_tokens <- function(token, space_after = TRUE, width = 72) {
  n <- length(token)
  space_after <- rep_len(space_after, n)
  w <- nchar(token)

  line <- integer(n)
  x0 <- numeric(n)
  cur_line <- 1L
  cur_x <- 0

  for (i in seq_len(n)) {
    if (cur_x > 0 && cur_x + w[i] > width) {
      cur_line <- cur_line + 1L
      cur_x <- 0
    }
    line[i] <- cur_line
    x0[i] <- cur_x
    cur_x <- cur_x + w[i] + if (space_after[i]) 1 else 0
  }

  data.frame(line = line, x0 = x0, x1 = x0 + w, xmid = x0 + w / 2)
}

# Suggested device size, in inches, for a heatmap of this shape.
#
# ggplot's `size` aesthetic is a font height in mm; a monospace face advances
# roughly 0.6 of its height per character, so `width` columns need
# `width * 0.6 * label_size` mm of panel. Height allows 1.9 line-heights per
# rendered line, which leaves the tiles visually separated rather than abutting.
# Both are estimates -- they put the figure in the right neighbourhood so the
# text neither overflows its tiles nor floats inside them, and the caller can
# nudge from there.
text_heatmap_size <- function(width, n_lines, n_panels = 1, label_size = 2.6,
                              legend_in = 1.35, title_in = 0.75) {
  w_in <- width * 0.6 * label_size / 25.4 + legend_in
  h_in <- n_panels * (n_lines * 1.9 * label_size / 25.4 + title_in)
  c(width = round(w_in, 2), height = round(h_in, 2))
}

# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #

#' Render tokens as tinted flowing text.
#'
#' @param data       one row per token; several panels stack by repeating the
#'                   same token sequence once per panel value.
#' @param token,value,panel  column names (strings). `panel` may be NULL.
#' @param space_after column name flagging whether a space follows the token
#'                   (spaCy's `whitespace_`); NULL treats every token as spaced.
#' @param highlight  regex matched case-insensitively against the token, or a
#'                   character vector of exact token strings. Matching tokens
#'                   get an outline so the eye finds them in both panels.
#' @param limits     fill scale limits. Leave NULL to span the data -- but pass
#'                   an explicit range whenever panels are meant to be compared,
#'                   so a token's colour means the same thing in each.
#' @param label_dark fill quantile above which label text switches to white.
#' @return a ggplot object.
text_heatmap <- function(data,
                         token = "token",
                         value = "value",
                         panel = NULL,
                         space_after = NULL,
                         highlight = NULL,
                         width = 72,
                         limits = NULL,
                         palette = "mako",
                         direction = -1,
                         legend_title = "Surprisal\n(bits)",
                         label_size = 2.6,
                         na_colour = "grey92",
                         highlight_colour = "#D55E00",
                         highlight_linewidth = 0.45,
                         label_dark = 0.55,
                         panel_levels = NULL,
                         base_size = 10) {

  stopifnot(is.data.frame(data), token %in% names(data), value %in% names(data))

  # One canonical token order, taken from the first panel. Every panel must
  # repeat it exactly: the whole point of the figure is that the same word sits
  # in the same place in each panel and only its colour moves.
  if (is.null(panel)) {
    data$.panel <- factor("")
  } else {
    lv <- if (is.null(panel_levels)) unique(as.character(data[[panel]])) else panel_levels
    data$.panel <- factor(as.character(data[[panel]]), levels = lv)
    if (anyNA(data$.panel)) stop("panel values outside `panel_levels`")
  }
  data <- data[order(data$.panel), , drop = FALSE]

  first <- data[data$.panel == levels(data$.panel)[1], , drop = FALSE]
  n_tok <- nrow(first)
  counts <- table(data$.panel)
  if (any(counts != n_tok)) {
    stop("every panel needs the same ", n_tok, " tokens; got ",
         paste(sprintf("%s=%d", names(counts), counts), collapse = ", "))
  }
  by_panel <- split(as.character(data[[token]]), data$.panel)
  if (!all(vapply(by_panel, identical, logical(1), by_panel[[1]]))) {
    stop("panels carry different token sequences")
  }

  sp <- if (is.null(space_after)) TRUE else first[[space_after]]
  lay <- layout_text_tokens(as.character(first[[token]]), sp, width)
  data <- cbind(data, lay[rep(seq_len(n_tok), times = nlevels(data$.panel)), ])

  data$.value <- as.numeric(data[[value]])
  data$.label <- as.character(data[[token]])
  if (is.null(limits)) limits <- range(data$.value, na.rm = TRUE)

  # Label contrast follows the fill, not the raw value, so a clipped scale still
  # gets readable text at both ends.
  scaled <- (data$.value - limits[1]) / diff(limits)
  data$.light_text <- !is.na(scaled) & scaled > label_dark

  if (is.null(highlight)) {
    data$.hit <- FALSE
  } else if (length(highlight) == 1 && !highlight %in% data$.label) {
    data$.hit <- grepl(highlight, data$.label, ignore.case = TRUE)
  } else {
    data$.hit <- tolower(data$.label) %in% tolower(highlight)
  }

  p <- ggplot(data, aes(xmin = x0, xmax = x1, ymin = -line - 0.42, ymax = -line + 0.42)) +
    geom_rect(aes(fill = .value), colour = NA) +
    geom_text(aes(x = xmid, y = -line, label = .label, colour = .light_text),
              size = label_size, family = "mono", show.legend = FALSE) +
    scale_fill_viridis_c(option = palette, direction = direction,
                         limits = limits, oob = scales::squish,
                         na.value = na_colour, name = legend_title) +
    scale_colour_manual(values = c("FALSE" = "grey12", "TRUE" = "white"),
                        guide = "none") +
    scale_x_continuous(expand = expansion(add = 0.5)) +
    scale_y_continuous(expand = expansion(add = 0.5)) +
    labs(x = NULL, y = NULL) +
    theme_minimal(base_size = base_size) +
    theme(
      panel.grid = element_blank(),
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      strip.text = element_text(face = "bold", hjust = 0),
      plot.subtitle = element_text(colour = "grey30"),
      legend.key.height = grid::unit(1.1, "lines")
    )

  if (any(data$.hit)) {
    p <- p + geom_rect(data = data[data$.hit, , drop = FALSE],
                       fill = NA, colour = highlight_colour,
                       linewidth = highlight_linewidth)
  }
  if (!is.null(panel)) {
    p <- p + facet_wrap(~.panel, ncol = 1, scales = "free_y")
  }
  p
}
