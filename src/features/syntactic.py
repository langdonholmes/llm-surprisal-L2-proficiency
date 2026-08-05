"""Syntactic complexity indices.

A battery sampling distinct subconstructs rather than accumulating redundant
omnibus measures (Norris & Ortega, 2009; Bulté & Housen, 2012). Entered into the
network analysis:

    mean_length_of_clause         clausal (length)
    complex_nominals_per_clause   phrasal (NP elaboration)
    coord_phrases_per_clause      phrasal (coordination)

All three are L2SCA indices (Lu, 2010, 2011), so the battery is directly
comparable to the large body of work using that tool.

Three further indices are computed and kept in the metrics table but are NOT
entered. ``clauses_per_tunit`` indexes punctuation rather than subordination on
this corpus -- see its docstring. ``deps_per_clause`` is Kyle and Crossley's
(2018) fine-grained clausal density index; ``mean_length_of_clause`` measures
much the same thing and is far more widely used, so MLC is the one entered.
``deps_per_nominal`` is also Kyle and Crossley's; in the network analysis it
behaved as an island, loading -.098 on its own community.

A NOTE ON THE SHARED DENOMINATOR. Lu's indices and Kyle and Crossley's were
developed against different clause definitions. Mixing them silently would mean
"per clause" denoted two different things across the battery, so a single
clause definition -- Lu's -- is used throughout: a clause is a structure with a
subject and a finite verb. ``deps_per_clause`` is therefore Kyle and Crossley's
numerator over Lu's denominator, and its values are not directly comparable to
published TAASSC figures.

Operationalizations are written against spaCy's English dependency scheme (a
modified ClearNLP/OntoNotes scheme), NOT Universal Dependencies. The
differences that matter here: copular clauses are headed by ``be`` rather than
by a predicate with a ``cop`` dependent; prepositional phrases attach as
``prep`` -> ``pobj`` rather than ``nmod`` -> ``case``; relative clauses attach
as ``relcl``; and clausal subjects (including gerund and infinitive subjects)
attach as ``csubj``.
"""

import numpy as np

# Relations marking a subject. `expl` covers existential "There is a problem",
# where "There" is the grammatical subject and the clause is a clause.
SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "expl"}

# Auxiliary relations within a verb group.
AUX_DEPS = {"aux", "auxpass"}

NOMINAL_POS = {"NOUN", "PROPN"}

# Relations by which a finite clause can legitimately be subordinated to
# another. A clause attached by anything else (`prep`, `dep`, `dobj`, ...) is a
# parse failure, and in this corpus that failure is nearly always an
# unpunctuated run-on -- see `_is_independent`.
SUBORDINATING_DEPS = {
    "advcl", "ccomp", "relcl", "acl", "csubj", "csubjpass", "xcomp", "pcomp",
    "conj",
}

# Modifiers that make a noun a complex nominal, following Lu's (2010) adoption
# of Cooper (1976): "nouns plus adjective, possessive, prepositional phrase,
# relative clause, participle, or appositive". Determiners are deliberately
# absent -- they are not in Cooper's list, and article use in L2 writing is an
# accuracy phenomenon rather than a complexity one.
CN_MODIFIERS = {
    "amod",      # adjectival modifier
    "compound",  # noun adjunct ("distance learning students")
    "poss",      # possessive
    "prep",      # prepositional phrase (the PP attaches via its preposition)
    "relcl",     # relative clause
    "acl",       # participial / clausal modifier
    "appos",     # appositive
    "nmod",      # nominal modifier
    "nummod",    # numeric modifier
}

# Positions in which a finite verb heads a clause even with no overt subject.
#
# Lu's prose definition asks for "a subject and a finite verb", but L2SCA's
# actual Tregex pattern keys on finiteness alone:
#
#     S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD
#                | < (VP <# MD|VBP|VBZ|VBD)]
#
# The two come apart on learner text with a dropped subject ("Is very important
# to study"), which the constituency parse still wraps in an S and L2SCA still
# counts. They also come apart the other way, and there the subject requirement
# is what saves us: verb-phrase coordination ("I ate and slept") puts the second
# conjunct inside one S, so L2SCA counts one clause, and requiring a subject is
# how a dependency parse reproduces that. Hence the split -- a subject is
# required in general, but not in the positions that get their own S node
# regardless. `conj` and `xcomp` are deliberately absent.
SUBJECTLESS_CLAUSE_DEPS = {"ROOT", "ccomp", "advcl", "relcl", "acl", "pcomp"}

# Nominal clauses: the second limb of Lu's complex-nominal definition. `csubj`
# also covers the third limb (gerunds and infinitives in subject position),
# which spaCy attaches as a clausal subject rather than as a verbal nsubj.
NOMINAL_CLAUSE_DEPS = {"ccomp", "csubj", "csubjpass"}

# Phrase types whose coordination Lu (2010) counts: adjective, adverb, noun and
# verb phrases. PRON is included because a coordinated pronoun ("he and I") is
# an NP conjunct.
COORD_PHRASE_POS = {"ADJ", "ADV", "NOUN", "PROPN", "PRON", "VERB"}


def _is_finite(token) -> bool:
    """Whether a verbal token is finite.

    Read from morphological features: either the token itself is a finite verb
    form, or it carries a finite auxiliary ("If she *is* running"). Modals are
    finite.

    This correctly excludes the exceptional-case-marking structures that are
    common in this corpus -- "let students *use* their phones", "allow students
    to *bring*" -- where a non-finite verb has its own subject. Lu's definition
    requires a finite verb, so these are not clauses. The residual error is
    spaCy occasionally tagging a past-tense verb VBN rather than VBD, which
    costs a genuine clause; this affects well under 1% of subject-bearing
    verbal tokens in ELLIPSE.
    """
    if "Fin" in token.morph.get("VerbForm"):
        return True
    return any(
        c.dep_ in AUX_DEPS and "Fin" in c.morph.get("VerbForm")
        for c in token.children
    )


def _clause_heads(doc):
    """Tokens heading a clause, in Lu's (2010) sense: a subject and a finite verb.

    Both verbs and auxiliaries qualify as the verbal element, because spaCy's
    English scheme heads copular clauses with ``be`` rather than with the
    predicate: in "The dog is happy", ``is`` is the ROOT and ``happy`` an
    ``acomp`` dependent. Auxiliaries *within* a verb group are excluded --
    "should" in "she should go" is part of one clause, not a second one.

    Two departures from the bare prose definition, both to track what L2SCA's
    Tregex pattern actually does: a base-form verb heading its own sentence is
    an imperative and counts as a clause, and a finite verb in one of the
    :data:`SUBJECTLESS_CLAUSE_DEPS` positions counts even with no overt subject.
    """
    return [
        t
        for t in doc
        if t.pos_ in {"VERB", "AUX"}
        and t.dep_ not in AUX_DEPS
        and (
            # Imperative: base-form verb heading its own sentence, no subject.
            (
                t.tag_ == "VB"
                and t.dep_ == "ROOT"
                and not any(c.dep_ in SUBJECT_DEPS for c in t.children)
            )
            or (
                _is_finite(t)
                and (
                    any(c.dep_ in SUBJECT_DEPS for c in t.children)
                    or t.dep_ in SUBJECTLESS_CLAUSE_DEPS
                )
            )
        )
    ]


def _is_independent(token) -> bool:
    """Whether a non-ROOT clause is really an independent clause.

    The T-unit was devised (Hunt, 1965) to be immune to the writer's
    punctuation, which is exactly what makes it attractive for learner writing.
    A dependency operationalization does not inherit that immunity for free:
    the parser's sentence segmentation is punctuation-driven, so an
    unpunctuated run-on arrives as one "sentence" and its independent clauses
    get attached to each other as if subordinate. Left uncorrected, C/T then
    measures punctuation rather than subordination -- in ELLIPSE it correlates
    .81 with words per terminal punctuation mark.

    Two signatures recover the independent clauses:

    1. The clause is attached by a relation that cannot subordinate a clause at
       all (`prep`, `dep`, `dobj`, ...). A finite verb hanging off a
       preposition is a parse failure, and here it is nearly always a run-on:
       "I like dogs, I like cats" attaches the second `like` as `prep`.

    2. The clause is a `ccomp` with no complementizer that *precedes* its head.
       A genuine zero-complementizer complement follows its matrix verb ("He
       said the book was good"), so order separates the two cases: "School is
       fun, students learn a lot" attaches `is` as a `ccomp` of the later
       `learn`.
    """
    if token.dep_ not in SUBORDINATING_DEPS:
        return True
    return (
        token.dep_ == "ccomp"
        and token.i < token.head.i
        and not any(c.dep_ == "mark" for c in token.children)
    )


def _n_tunits(doc, clause_heads=None) -> int:
    """Count T-units: each one main clause plus whatever attaches to it.

    A T-unit head is a clause that is not subordinate to another clause --
    operationally, a clause that is its sentence's ROOT, or one coordinated
    (possibly transitively) with such a clause and carrying its own subject.

    This is what distinguishes clause coordination from verb-phrase
    coordination: "I ate and I slept" is two T-units, "I ate and slept" is one,
    because the second conjunct has no subject of its own and so is not a
    clause. A conjunct hanging off a *subordinate* clause is not promoted --
    in "He said that I ate and I slept", both conjuncts sit inside the single
    T-unit headed by "said".

    Independent clauses that the parser buried inside a run-on are recovered by
    :func:`_is_independent`, without which C/T would largely be measuring the
    writer's punctuation. Sentences containing no clause at all -- bare
    noun-phrase fragments -- contribute nothing.
    """
    if clause_heads is None:
        clause_heads = _clause_heads(doc)
    clause_set = set(clause_heads)

    n = 0
    for sent in doc.sents:
        sent_clauses = {t for t in clause_set if sent.start <= t.i < sent.end}
        if not sent_clauses:
            continue
        heads = {t for t in sent_clauses if t is sent.root or _is_independent(t)}

        # Promote clauses coordinated with an independent clause, transitively.
        changed = True
        while changed:
            changed = False
            for t in sent_clauses - heads:
                if t.dep_ == "conj" and t.head in heads:
                    heads.add(t)
                    changed = True

        # Fallback: the sentence has clauses but none of them reads as
        # independent, which happens when a grammatical error leaves the main
        # verb non-finite ("they should not all be require to do community
        # service if they don't want to..."). The writer plainly produced a
        # terminable unit, so count one.
        n += len(heads) if heads else 1
    return n


def _n_dependents(token) -> int:
    """Count a token's dependents, excluding punctuation.

    Punctuation is excluded from every dependent count in this module so that
    the density indices do not absorb the inconsistent punctuation that
    characterizes learner writing.
    """
    return sum(1 for c in token.children if c.dep_ != "punct")


def clauses_per_tunit(doc) -> float:
    """Clauses per T-unit (Lu, 2010, 2011).

    Sentential-level subordination: the classic C/T ratio, developmentally
    diagnostic at intermediate proficiency (Norris & Ortega, 2009), which is
    where the ELLIPSE writers sit.

    NOT entered into the network analysis. On ELLIPSE this index measures
    punctuation rather than subordination. The T-unit is the only denominator
    in the battery that depends on sentence segmentation, and spaCy segments on
    punctuation, so the immunity Hunt (1965) designed the T-unit for does not
    survive a dependency operationalization. :func:`_is_independent` recovers
    much of it but not enough:

    - C/T correlates +.726 with words per terminal punctuation mark; the other
      three L2SCA indices sit at +.078 to +.121.
    - 45% of the corpus exceeds 25 words per terminal punctuation mark.
    - C/T correlates -.205 with Overall proficiency, but -.014 among
      well-punctuated essays alone (8-25 words per mark, n = 3,590).
    - Entered into the network it joined the *cohesion* community (loading
      .478) rather than the syntactic one, and its strongest local dependence
      was with connective density (wTO .348).

    Restoring subordination to the battery would need an index normalized per
    clause rather than per T-unit, e.g. adverbial clauses per clause.
    """
    clauses = _clause_heads(doc)
    n_tunits = _n_tunits(doc, clauses)
    if n_tunits == 0:
        return np.nan
    return len(clauses) / n_tunits


def _n_words(doc) -> int:
    """Count words, excluding punctuation and whitespace.

    L2SCA's word count is a count of terminal nodes in the parse minus
    punctuation, which is what this reproduces. Note that spaCy splits
    contractions and possessive clitics ("don't" -> "do" + "n't", "student's"
    -> "student" + "'s"), so word counts run slightly above a whitespace
    tokenizer's; the same tokenization underlies every index here, so the
    battery stays internally consistent.
    """
    return sum(1 for t in doc if not t.is_punct and not t.is_space)


def mean_length_of_clause(doc) -> float:
    """Mean length of clause in words (Lu, 2010, 2011).

    Clausal-level length: total words over total clauses. Closely related to
    :func:`deps_per_clause` -- both index how much material a clause carries --
    but MLC is the standard L2SCA measure and by far the more widely reported,
    so it is the one entered into the network.

    Words not inside any clause (noun-phrase fragments, headings) still count in
    the numerator, as in L2SCA, where the numerator is simply the text's word
    count.
    """
    clauses = _clause_heads(doc)
    if not clauses:
        return np.nan
    return _n_words(doc) / len(clauses)


def complex_nominals_per_clause(doc) -> float:
    """Complex nominals per clause (Lu, 2010, 2011).

    Phrasal (NP) elaboration. A complex nominal is (i) a noun with an
    adjectival, possessive, prepositional, relative-clause, participial or
    appositive modifier, (ii) a nominal clause, or (iii) a gerund or infinitive
    in subject position -- the last two being ``ccomp`` / ``csubj`` here.

    Unlike :func:`deps_per_nominal`, this counts elaborated nominals rather
    than averaging dependents over all nominals, so a text is not penalized for
    also containing many bare nouns, and determiners play no part.
    """
    clauses = _clause_heads(doc)
    if not clauses:
        return np.nan
    n_cn = sum(
        1
        for t in doc
        if (
            t.pos_ in NOMINAL_POS
            and any(c.dep_ in CN_MODIFIERS for c in t.children)
        )
        or t.dep_ in NOMINAL_CLAUSE_DEPS
    )
    return n_cn / len(clauses)


def coord_phrases_per_clause(doc) -> float:
    """Coordinate phrases per clause (Lu, 2010, 2011).

    Phrasal coordination: adjective, adverb, noun and verb phrases joined by a
    coordinating conjunction. In Norris and Ortega's (2009) developmental
    sequence coordination precedes subordination, which precedes phrasal
    elaboration, so this index is the early-stage anchor of the battery.

    Coordinated *clauses* are excluded -- a conjunct with its own subject and
    finite verb is clause coordination, counted by
    :func:`clauses_per_tunit` instead. "He ate and slept" contributes a
    coordinate phrase; "He ate and he slept" does not.
    """
    clauses = _clause_heads(doc)
    if not clauses:
        return np.nan
    clause_set = set(clauses)
    n_cp = sum(
        1
        for t in doc
        if t.dep_ == "conj"
        and t.pos_ in COORD_PHRASE_POS
        and t not in clause_set
    )
    return n_cp / len(clauses)


def deps_per_clause(doc) -> float:
    """Mean number of dependents attached to each clausal head.

    Aggregate structural density at the clause level (Kyle & Crossley, 2018),
    over Lu's clause definition -- see the module docstring on the shared
    denominator.

    NOT part of the battery: :func:`mean_length_of_clause` indexes the same
    clausal-length subconstruct and is the standard, widely reported measure.
    Retained for reference and for comparison against published TAASSC work.
    """
    clauses = _clause_heads(doc)
    if not clauses:
        return np.nan
    return sum(_n_dependents(t) for t in clauses) / len(clauses)


def deps_per_nominal(doc) -> float:
    """Mean number of dependents attached to each nominal head.

    NOT part of the battery -- computed and retained for reference only. In the
    nomological network it loaded -.098 on its own community and correlated
    ~.00 with every ELLIPSE subscore, i.e. it behaved as an island. Determiners
    are its largest single component (31.8% of counted dependents), and
    removing them does not rescue the correlations, so the null is a property
    of the construct at this developmental level rather than an artifact.
    :func:`complex_nominals_per_clause` is the battery's phrasal index instead.

    Pronouns are excluded as nominal heads: they are effectively
    non-elaborable, so including them would make the index partly a measure of
    pronoun rate.
    """
    nominals = [t for t in doc if t.pos_ in NOMINAL_POS]
    if not nominals:
        return np.nan
    return sum(_n_dependents(t) for t in nominals) / len(nominals)
