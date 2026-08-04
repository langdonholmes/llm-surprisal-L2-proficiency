"""Fine-grained dependency-based syntactic complexity indices.

A four-index battery sampling distinct subconstructs rather than accumulating
redundant omnibus measures (Norris & Ortega, 2009; Bulté & Housen, 2012):

    deps_per_nominal   phrasal (NP elaboration)
    advmod_per_clause  clausal (adverbial modification)
    advcl_per_clause   clausal (finite adverbial subordination)
    deps_per_clause    clausal (aggregate structural density)

All four follow the fine-grained approach of Kyle and Crossley (2018): each
normalizes to a syntactic unit (nominal or clause) rather than to the
orthographic sentence, avoiding the punctuation and length artifacts that
compromise sentence-normalized measures in learner corpora.

``words_per_clause`` is provided as the designated substitute index (a
phrasal-at-clause-level measure in the Norris & Ortega scheme) should the two
clause-level modification indices prove locally dependent with each other. It
is computed and carried in the metrics table but is not part of the battery.

Operationalizations below are written against spaCy's English dependency
scheme (a modified ClearNLP/OntoNotes scheme), NOT Universal Dependencies.
The differences that matter here: copular clauses are headed by ``be`` rather
than by a predicate with a ``cop`` dependent; prepositional phrases attach as
``prep`` -> ``pobj`` rather than ``nmod`` -> ``case``; and negation carries its
own ``neg`` label rather than being folded into ``advmod``.
"""

import numpy as np

# Nominal heads. Pronouns (PRON) are excluded: they are effectively
# non-elaborable, so including them would make the index partly a measure of
# pronoun rate rather than of phrasal elaboration.
NOMINAL_POS = {"NOUN", "PROPN"}

# Auxiliary relations within a verb group. A token bearing one of these is part
# of another clause's verb group, not the head of a clause of its own.
AUX_DEPS = {"aux", "auxpass"}


def _clause_heads(doc):
    """Tokens heading a clause.

    A clause head is a lexical verb, or an auxiliary that is not itself inside
    another verb group. The second case captures copular clauses, which spaCy's
    English scheme heads with ``be`` (tagged AUX) rather than with the
    predicate: in "The dog is happy", ``is`` is the ROOT and ``happy`` an
    ``acomp`` dependent.
    """
    return [
        t
        for t in doc
        if t.pos_ == "VERB" or (t.pos_ == "AUX" and t.dep_ not in AUX_DEPS)
    ]


def _n_dependents(token) -> int:
    """Count a token's dependents, excluding punctuation.

    Punctuation is excluded from every dependent count in this module so that
    the density indices do not absorb the inconsistent punctuation that
    characterizes learner writing.
    """
    return sum(1 for c in token.children if c.dep_ != "punct")


def _is_finite(token) -> bool:
    """Whether a verbal token heads a finite clause.

    Finiteness is read from morphological features: either the token itself is
    a finite verb form, or it carries a finite auxiliary ("If she *is*
    running"). Modals are finite (``must``, ``will`` are VerbForm=Fin).
    Non-finite adverbial clauses -- participial ("Having finished the work,
    ...") and infinitival ("To succeed, ...") -- are excluded.
    """
    if "Fin" in token.morph.get("VerbForm"):
        return True
    return any(
        c.dep_ in AUX_DEPS and "Fin" in c.morph.get("VerbForm")
        for c in token.children
    )


def deps_per_nominal(doc) -> float:
    """Mean number of dependents attached to each nominal head.

    Phrasal (NP-level) complexity: a direct operationalization of the phrasal
    embedding Biber, Gray and Poonpon (2011) identify as the developmental
    endpoint of written academic register, and the strongest predictor class in
    Kyle and Crossley (2018).

    All dependent types count, so determiners, adjectival modifiers, nominal
    compounds, possessives, prepositional modifiers (via their ``prep`` head)
    and postmodifying clauses (``relcl``, ``acl``) all contribute.
    """
    nominals = [t for t in doc if t.pos_ in NOMINAL_POS]
    if not nominals:
        return np.nan
    return sum(_n_dependents(t) for t in nominals) / len(nominals)


def advmod_per_clause(doc) -> float:
    """Mean number of non-clausal adverbial dependents per clause.

    Clausal modification: captures precision, stance and hedging marked
    adverbially, structurally and functionally distinct from nominal
    modification.

    Only the ``advmod`` relation counts. ``neg`` is excluded because spaCy's
    English scheme labels negation separately and negation is a distinct
    grammatical category rather than adverbial modification; ``npadvmod``
    ("they went home every *day*") is excluded because it is a nominal
    structure, and counting it here would blur the boundary with
    :func:`deps_per_nominal`.
    """
    heads = _clause_heads(doc)
    if not heads:
        return np.nan
    n_advmod = sum(
        sum(1 for c in h.children if c.dep_ == "advmod") for h in heads
    )
    return n_advmod / len(heads)


def advcl_per_clause(doc) -> float:
    """Mean number of finite adverbial clause dependents per clause.

    Clausal subordination: the subconstruct a purely elaborative battery would
    omit, developmentally diagnostic at intermediate proficiency (Norris &
    Ortega, 2009) and an early-to-mid stage feature in Biber et al.'s (2011)
    sequence. Restricting to finite adverbial clauses keeps the index on the
    stage the sequence assigns it, since non-finite clauses are a later stage.
    """
    heads = _clause_heads(doc)
    if not heads:
        return np.nan
    n_advcl = sum(
        sum(1 for c in h.children if c.dep_ == "advcl" and _is_finite(c))
        for h in heads
    )
    return n_advcl / len(heads)


def deps_per_clause(doc) -> float:
    """Mean number of dependents attached to each clausal head.

    Aggregate structural density at the clause level; the general clausal
    counterpart to :func:`deps_per_nominal`. Partially subsumes
    :func:`advmod_per_clause` and :func:`advcl_per_clause`, which are both
    subsets of clausal dependents -- this local dependence is screened
    empirically by Unique Variable Analysis in the network analysis rather than
    assumed away.
    """
    heads = _clause_heads(doc)
    if not heads:
        return np.nan
    return sum(_n_dependents(h) for h in heads) / len(heads)


def words_per_clause(doc) -> float:
    """Mean non-punctuation tokens per clause.

    Not part of the battery. Held in reserve as the substitute phrasal
    index at clause level (Norris & Ortega, 2009) in case the two clause-level
    modification indices turn out to be locally dependent with each other.
    """
    heads = _clause_heads(doc)
    if not heads:
        return np.nan
    n_words = sum(1 for t in doc if not t.is_punct)
    return n_words / len(heads)
