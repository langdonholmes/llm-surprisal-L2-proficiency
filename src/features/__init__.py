from .lexical import lexical_density, log_mean_token_freq, mean_word_length, mtld
from .syntactic import (
    advcl_per_clause,
    advmod_per_clause,
    deps_per_clause,
    deps_per_nominal,
    words_per_clause,
)
from .phraseological import MiCalculator
from .cohesion import connective_density, content_word_overlap, sentence_similarity


def calculate_all_features(doc, token_freq, total_tokens, mi_calculator) -> dict:
    """Calculate all 15 linguistic features for a single spaCy Doc.

    14 are candidate variables for the network analysis (which adds the two
    predictability features from the Study 1 benchmark); words_per_clause is
    carried as the reserve substitute for the syntactic battery, and is
    computed but not entered (see features/syntactic.py).
    """
    features = {}

    # Lexical (4)
    features["mtld"] = mtld(doc)
    features["lexical_density"] = lexical_density(doc)
    features["log_mean_token_freq"] = log_mean_token_freq(doc, token_freq, total_tokens)
    features["mean_word_length"] = mean_word_length(doc)

    # Syntactic (4) -- fine-grained dependency battery (Kyle & Crossley, 2018)
    features["deps_per_nominal"] = deps_per_nominal(doc)
    features["advmod_per_clause"] = advmod_per_clause(doc)
    features["advcl_per_clause"] = advcl_per_clause(doc)
    features["deps_per_clause"] = deps_per_clause(doc)
    # Reserve substitute index; not part of the battery.
    features["words_per_clause"] = words_per_clause(doc)

    # Phraseological (3) -- MI scores
    mi_scores = mi_calculator(doc)
    features.update(mi_scores)

    # Cohesion (3)
    features["content_word_overlap"] = content_word_overlap(doc)
    features["connective_density"] = connective_density(doc)
    features["sentence_similarity"] = sentence_similarity(doc)

    return features
