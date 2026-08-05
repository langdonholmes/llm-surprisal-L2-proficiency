from .lexical import lexical_density, log_mean_token_freq, mean_word_length, mtld
from .syntactic import (
    clauses_per_tunit,
    complex_nominals_per_clause,
    coord_phrases_per_clause,
    deps_per_clause,
    deps_per_nominal,
    mean_length_of_clause,
    n_words,
)
from .phraseological import MiCalculator
from .cohesion import connective_density, content_word_overlap, sentence_similarity


def calculate_all_features(doc, token_freq, total_tokens, mi_calculator) -> dict:
    """Calculate all 16 linguistic features for a single spaCy Doc.

    14 are candidate variables for the network analysis (which adds the two
    predictability features from the Study 1 benchmark). The other two,
    deps_per_clause and deps_per_nominal, are computed for reference but not
    entered — see features/syntactic.py.
    """
    features = {}

    # Lexical (4)
    features["mtld"] = mtld(doc)
    features["lexical_density"] = lexical_density(doc)
    features["log_mean_token_freq"] = log_mean_token_freq(doc, token_freq, total_tokens)
    features["mean_word_length"] = mean_word_length(doc)

    # Syntactic (4) -- the L2SCA battery (Lu, 2010, 2011)
    features["clauses_per_tunit"] = clauses_per_tunit(doc)
    features["mean_length_of_clause"] = mean_length_of_clause(doc)
    features["complex_nominals_per_clause"] = complex_nominals_per_clause(doc)
    features["coord_phrases_per_clause"] = coord_phrases_per_clause(doc)
    # Kyle & Crossley (2018) dependency indices. Retained for reference; not
    # entered (deps_per_clause duplicates MLC, deps_per_nominal is an island).
    features["deps_per_clause"] = deps_per_clause(doc)
    features["deps_per_nominal"] = deps_per_nominal(doc)

    # Phraseological (3) -- MI scores
    mi_scores = mi_calculator(doc)
    features.update(mi_scores)

    # Cohesion (3)
    features["content_word_overlap"] = content_word_overlap(doc)
    features["connective_density"] = connective_density(doc)
    features["sentence_similarity"] = sentence_similarity(doc)

    return features
