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
#
# Fills are lightened and the text stays dark throughout, so the tint reads as a
# highlighter laid over prose rather than as a grid of coloured cells. That keeps
# every word legible at the top of the scale, which an opaque fill does not, and
# it is why there is no light-text-on-dark-fill switch here.
#
# The lightening is baked into the colour ramp rather than applied as an `alpha`
# aesthetic on the rectangles. Drawing the tiles translucent leaves the legend's
# colourbar opaque, so the key stops describing the tiles it is a key for; and
# compositing mako's near-black top end against white turns it grey, which loses
# the deep blue the rest of the project's figures are keyed to. Blending the ramp
# up front fixes both, since the scale and the tiles then carry the same values.

suppressMessages(library(ggplot2))

# Advance width of DejaVu Sans Mono, in em (1233/2048 units). Every width
# calculation below depends on this, so a different monospace face needs its own
# value or the tiles stop matching the glyphs they sit behind.
MONO_ADVANCE <- 0.60205
MONO_FAMILY <- "DejaVu Sans Mono"

# Composite a colour over a background at opacity `alpha`, returning opaque hex.
blend_over <- function(hex, alpha, bg = "#FFFFFF") {
  fg <- grDevices::col2rgb(hex) / 255
  back <- as.vector(grDevices::col2rgb(bg)) / 255
  grDevices::rgb(t(alpha * fg + (1 - alpha) * back))
}

# The project's sequential ramp, lightened and with the near-black top end
# trimmed off. `begin` stops mako short of pure black, which stays blue after
# blending where the true extreme goes neutral grey.
heatmap_ramp <- function(alpha = 0.75, n = 256, option = "mako",
                         begin = 0.28, end = 1, direction = -1,
                         bg = "#FFFFFF") {
  cols <- viridisLite::viridis(n, option = option, begin = begin, end = end,
                               direction = direction)
  blend_over(cols, alpha, bg)
}

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

# Device size, in inches, that makes one character of `label_size` text exactly
# one x-unit wide. Getting this right is the whole of the alignment problem: the
# tiles are laid out in character units, so the panel has to be
# `width * MONO_ADVANCE * label_size` mm across or the tiles and the glyphs
# disagree. `side_in` is everything outside the panel (legend, margins); measure
# it once for a given legend and reuse.
#
# Height allows 1.9 line-heights per rendered line, which leaves the rows
# visually separated rather than abutting.
text_heatmap_size <- function(width, n_lines, n_panels = 1, label_size = 3.2,
                              side_in = 1.35, title_in = 0.75) {
  panel_in <- width * MONO_ADVANCE * label_size / 25.4
  c(width = round(panel_in + side_in, 2),
    height = round(n_panels * (n_lines * 1.9 * label_size / 25.4 + title_in), 2),
    panel = round(panel_in, 2))
}

# Lines the layout will use, so a caller can size the device before plotting.
text_heatmap_lines <- function(token, space_after = TRUE, width = 72) {
  max(layout_text_tokens(token, space_after, width)$line)
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
#'                   are marked so the eye finds them in every panel.
#' @param highlight_style "underline" (a rule beneath the word, which leaves the
#'                   fill undisturbed) or "box".
#' @param limits     fill scale limits. Leave NULL to span the data, but pass an
#'                   explicit range whenever panels are meant to be compared, so
#'                   a token's colour means the same thing in each.
#' @param alpha      how strongly the ramp is tinted, from 0 (white) to 1 (the
#'                   full palette). Lower values keep the text readable at the
#'                   top of the scale at the cost of some dynamic range. Applied
#'                   to the scale, not to the rectangles, so the legend matches.
#' @param pad        horizontal padding added to each tile, in characters.
#' @return a ggplot object.
text_heatmap <- function(data,
                         token = "token",
                         value = "value",
                         panel = NULL,
                         space_after = NULL,
                         highlight = NULL,
                         highlight_style = c("underline", "box"),
                         width = 72,
                         limits = NULL,
                         palette = "mako",
                         direction = -1,
                         alpha = 0.75,
                         ramp_begin = 0.28,
                         legend_title = "Surprisal\n(bits)",
                         label_size = 3.2,
                         label_colour = "grey10",
                         family = MONO_FAMILY,
                         na_colour = "grey92",
                         highlight_colour = "#B2182B",
                         highlight_linewidth = 0.9,
                         pad = 0.12,
                         panel_levels = NULL,
                         base_size = 10) {

  highlight_style <- match.arg(highlight_style)
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

  if (is.null(highlight)) {
    data$.hit <- FALSE
  } else if (length(highlight) == 1 && !highlight %in% data$.label) {
    data$.hit <- grepl(highlight, data$.label, ignore.case = TRUE)
  } else {
    data$.hit <- tolower(data$.label) %in% tolower(highlight)
  }

  p <- ggplot(data) +
    geom_rect(aes(xmin = x0 - pad, xmax = x1 + pad,
                  ymin = -line - 0.42, ymax = -line + 0.42, fill = .value),
              colour = NA) +
    geom_text(aes(x = xmid, y = -line, label = .label),
              size = label_size, family = family, colour = label_colour) +
    scale_fill_gradientn(
      colours = heatmap_ramp(alpha, option = palette, begin = ramp_begin,
                             direction = direction),
      limits = limits, oob = scales::squish,
      na.value = na_colour, name = legend_title) +
    scale_x_continuous(expand = expansion(add = 0.6)) +
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
    hits <- data[data$.hit, , drop = FALSE]
    p <- p + if (highlight_style == "underline") {
      geom_segment(data = hits,
                   aes(x = x0 - pad, xend = x1 + pad,
                       y = -line - 0.46, yend = -line - 0.46),
                   colour = highlight_colour, linewidth = highlight_linewidth,
                   lineend = "round")
    } else {
      geom_rect(data = hits,
                aes(xmin = x0 - pad, xmax = x1 + pad,
                    ymin = -line - 0.42, ymax = -line + 0.42),
                fill = NA, colour = highlight_colour,
                linewidth = highlight_linewidth)
    }
  }
  if (!is.null(panel)) {
    p <- p + facet_wrap(~.panel, ncol = 1, scales = "free_y")
  }
  p
}
