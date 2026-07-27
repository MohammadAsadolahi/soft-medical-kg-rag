"""Biomedical entity-type schema for the soft knowledge graph.

Every node in the graph carries exactly one type from this closed vocabulary. The schema is
deliberately small and retrieval-oriented rather than ontologically complete: each type names a
category a clinician might *ask about* ("what FOOD lowers cholesterol", "what DISEASE_DISORDER
causes cold feet"), which is what makes a type usable as a query-side wildcard at search time.

This is the key difference from linking entities to a large external ontology such as UMLS. A
1M-concept ontology gives precision but no useful abstraction to query with -- there is no way to
write "any concept that could answer this question". Twelve coarse, answer-shaped types give
exactly that, at the cost of granularity.
"""
from __future__ import annotations

# The prose form is injected verbatim into the extraction and search prompts, so the LLM sees the
# same definitions the graph is built with. Keep the wording and the numbering stable: changing it
# changes extraction behaviour.
ENTITY_SCHEMA_PROSE = """1_ ACTIVITY : an activity or behavior. E.g. smoking, exercise, overeat, following a procedure, stop doing something
2_ BODY_PART : organ part or tissue of human body. E.g. heart, feet, tooth, skin, eye, hair
3_ CHEMICALS : chemicals found in the human body. E.g. sugar, cholesterol, iron
4_ DISCRIMINATIVES : entities that discriminate between different patients, diseases, conditions. E.g. age, gender, ethnicity, a group with a particular predisposition e.g pregnant people, kids, olders
5_ DISEASE_DISORDER : diseases, disorders or syndromes. E.g. Diabetes mellitus, Hypertension, CVD, obesity
6_ DRUGS : any chemical or supplement used as a drug. E.g. vaccine, vitamin B6, painkiller
7_ BIO_MARKER : any measurable quantity or property that relates to health state, disorders, symptoms. E.g. blood pressure, body temperature, fever
8_ FOOD : any material that is considered as food. E.g. fruit, lamb meat, pork, apples, high protein meals
9_ GENES : any gene, mutation or genetical substance
10_ HEALTH_PROCEDURES : or therapies used to treat diseases, or a procedure to recognize a health situation. E.g. blood test, sugar measurement
11_ SIGN_OR_SYMPTOM : Symptoms or clinical signs associated with a disease or syndrome. E.g. Frequent urination, Increased thirst, high blood pressure, body heat
12_ RISK_FACTOR : any factor that increase likelihood of a situation. E.g. higher cancer risk, heart attack likelihood
13_ NONE : none of the provided categories"""

# Types the extractor may assign. NONE is included so the structured-output schema can always be
# satisfied, but the extraction prompt instructs the model never to use it -- an entity that fits
# nothing after shortening should make the whole triplet get dropped instead.
EXTRACTION_TYPES: tuple[str, ...] = (
    "ACTIVITY", "BODY_PART", "CHEMICALS", "DISCRIMINATIVES", "DISEASE_DISORDER",
    "DRUGS", "BIO_MARKER", "FOOD", "GENES", "HEALTH_PROCEDURES", "SIGN_OR_SYMPTOM",
    "RISK_FACTOR", "NONE",
)

# Types usable as a query-side wildcard. NONE is excluded: "match anything untypeable" is not an
# information need, and allowing it would make every typed slot trivially satisfiable.
QUERY_TYPES: tuple[str, ...] = tuple(t for t in EXTRACTION_TYPES if t != "NONE")

SCHEMA_BLOCK = f"""This is the schema representation of the Neo4j database.
Node properties are the following:
{ENTITY_SCHEMA_PROSE}
Relationships are saved as verbs."""

# Typos and near-misses seen in real extractor output. Normalising them at load time rather than
# discarding the triplet keeps recall, and the mapping is auditable.
TYPE_ALIASES: dict[str, str] = {
    "DIISEASE_DISORDER": "DISEASE_DISORDER",
    "DISEASE-DISORDER": "DISEASE_DISORDER",
    "DISEASE": "DISEASE_DISORDER",
    "CHEMICAL": "CHEMICALS",
    "DRUG": "DRUGS",
    "GENE": "GENES",
    "BIOMARKER": "BIO_MARKER",
    "SYMPTOM": "SIGN_OR_SYMPTOM",
    "SIGN": "SIGN_OR_SYMPTOM",
    "HEALTH_PROCEDURE": "HEALTH_PROCEDURES",
    "CELL_LINE": "BODY_PART",
}


def normalize_type(raw: str | None) -> str:
    """Canonicalise an extractor-supplied type string.

    Unrecognised values collapse to NONE rather than raising: extraction runs are long and
    expensive, and one malformed type should degrade a single node's typing, not abort a build.
    """
    if not raw:
        return "NONE"
    t = raw.strip().upper().replace(" ", "_")
    t = TYPE_ALIASES.get(t, t)
    return t if t in EXTRACTION_TYPES else "NONE"


def is_type_token(token: str) -> bool:
    """True if a query slot is a schema TYPE rather than a concrete entity mention.

    Case-sensitive on purpose. The search prompt requires types in UPPERCASE, which lets the parser
    distinguish the type FOOD from a document that literally discusses the word "food".
    """
    return token.strip() in QUERY_TYPES
