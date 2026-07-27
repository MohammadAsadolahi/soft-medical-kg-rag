"""Prompts for the extraction and retrieval halves of the pipeline.

These strings are the substance of the method. The graph's quality is bounded almost entirely by
DECONTEXTUALIZE and EXTRACT below, and the retrieval quality by the two SEARCH prompts -- the
surrounding Python is comparatively mechanical.

Each prompt was developed by inspecting failure cases on biomedical abstracts and adding a rule for
each recurring error mode. That history is why the extraction prompt is long: nearly every line
suppresses a specific class of junk triplet (policy statements, study-methodology framing, generic
nutrition-inventory facts) or fixes a specific typing mistake. Shortening it costs precision.

The prompts are deliberately kept as data, separate from the code that calls them, so they can be
diffed, versioned, and swapped without touching pipeline logic.
"""
from __future__ import annotations

from .schema import (ENTITY_SCHEMA_PROSE, EXTRACTION_TYPES, QUERY_TYPES,
                     SCHEMA_BLOCK)

# ---------------------------------------------------------------------------
# Stage 1 -- DECONTEXTUALIZE: pronoun resolution
#
# Why this exists: a triplet is torn out of its sentence and indexed on its own, so any sentence
# whose meaning depends on the paragraph around it yields a useless node. "It reduces LDL" becomes
# the entity "it". Resolving references BEFORE extraction is what makes standalone triplets
# meaningful, and it is the single largest quality lever in the pipeline.
#
# The response contract at the top is load-bearing. Reasoning models asked to rewrite a passage
# will occasionally return a summary, a title, or an empty string; each of those silently destroys
# a document. The contract plus the explicit no-change examples suppress that.
# ---------------------------------------------------------------------------
DECONTEXTUALIZE_PRONOUNS = """RESPONSE CONTRACT: return the full passage exactly once. Never omit sentences, never answer with an empty string, and never output only a title or wrapper.

Replace every pronoun (he, she, it, they, them, their, his, her, its, we, our, him) with the specific entity it refers to based on context. Preserve the original meaning, structure, and punctuation.

Do NOT expand abbreviations or acronyms (keep PBDEs, IBS, CVD, etc. as-is). Only resolve actual pronouns.
Do NOT rewrite relative pronouns or clause markers such as which, that, who, whom, where, or when.
Do NOT resolve demonstratives such as this, that, these, or those in this step; step 2 handles those.
Do NOT copy wrapper text such as "here is the text:" into the output.
The user message may begin with a wrapper line exactly equal to "here is the text:". Ignore that wrapper line and process the remaining passage only.
If a pronoun does not have a clear explicit referent, leave that pronoun unchanged.

If the text contains no listed pronouns to resolve, return the entire input exactly as-is, including any title line.

Example with no change:
Input: "Vitamin B12 deficiency is common in older adults."
Output: "Vitamin B12 deficiency is common in older adults."

Output ONLY the final passage text with no explanations, prefixes, or apologies. You MUST always return the complete passage and never a blank response."""


# ---------------------------------------------------------------------------
# Stage 2 -- DECONTEXTUALIZE: indirect reference transfer
#
# Handles what stage 1 deliberately leaves alone: demonstratives and discourse anaphora ("this",
# "such", "the former", "to achieve so"). Split into a separate call because combining both in one
# prompt measurably increased the rate of unrequested paraphrasing -- the model started rewriting
# rather than substituting.
# ---------------------------------------------------------------------------
DECONTEXTUALIZE_REFERENCES = """RESPONSE CONTRACT: return the full passage exactly once, including any title line. Never omit sentences and never return a blank response.

Replace indirect references (this, that, these, those, such, the former, the latter, doing so, to achieve so, the target, etc.) with the full concept or phrase they refer to from preceding sentences. This makes each sentence independently understandable.

Always output the full passage. If nothing needs to be changed, copy the input passage exactly character-for-character. Never return a blank response or an empty string.
Do NOT copy wrapper text such as "here is the text:" into the output.
The user message may begin with a wrapper line exactly equal to "here is the text:". Ignore that wrapper line and process the remaining passage only.
Do NOT rewrite ordinary relative clauses with which/that/who unless they are genuinely functioning as an indirect reference to an earlier concept.

Example:
Input: "Losing weight is my target. To achieve that, I need to exercise regularly."
Output: "Losing weight is my target. To achieve weight loss, I need to exercise regularly."

Only replace references you are certain about. Do not paraphrase, summarize, or alter any other parts of the text.
If a reference cannot be resolved with confidence, keep that phrase unchanged rather than dropping or rewriting the sentence.
If replacing a reference would force you to rewrite sentence grammar or produce awkward repetition, leave the original phrase unchanged.

If the text contains no indirect references, return it exactly as-is, including the title line if present.

Example with no change:
Input: "Vitamin B12 deficiency is common in older adults."
Output: "Vitamin B12 deficiency is common in older adults."

Output ONLY the final text with no explanations, prefixes, or apologies. You MUST always return the complete text and never a blank response."""


# ---------------------------------------------------------------------------
# Stage 4 -- EXTRACT: typed relation extraction
#
# Emits (e1, e1_type, verb, e2, e2_type) in one pass. Entity typing is fused into extraction rather
# than run as a second classification call: the extractor already holds the sentence context that
# disambiguates "cholesterol" the CHEMICALS node from "cholesterol" the BIO_MARKER reading, and a
# separate classifier sees only the bare string and has to guess.
#
# Delivered via a forced tool call (see llm.py) so the JSON is grammar-constrained rather than
# parsed out of prose.
# ---------------------------------------------------------------------------
EXTRACT_RELATIONS = f"""Extract medical knowledge as (entity_1, verb, entity_2) triplets from the text.

ENTITY CONSTRAINTS:
- Short noun phrases only, max 5-6 words. Extract the core concept.
- If a phrase is longer, split into separate triplets (e.g., "colorectal diseases associated with faecal retention" → triplet 1: entity "colorectal diseases", triplet 2 linking it to "faecal retention").
- Never use clauses, sentence fragments, or single adjectives as entities.
- If a candidate entity would be generic or non-medical (e.g., authors, study, public, health authorities, time, years, information, protocol, recommendation form), shorten it to a medical concept or skip the triplet.
- Generic actor entities such as physicians, medical students, reviewers, hospitals, agencies, stakeholders, or institutions are usually not medically retrievable entities. Skip those triplets unless the text is about occupational exposure, provider health, or a direct patient-level clinical effect.
- Prefer the most specific non-redundant entity available in the sentence. Do not extract both a broad and a narrower restatement of the same fact unless they add distinct medical information.

VERB CONSTRAINTS:
- 1-4 words capturing the core relationship (e.g., "treats", "causes", "reduces risk of", "is associated with").
- HARD LIMIT: never exceed 4 words. If a candidate verb is 5+ words, shorten it.
  Shorten examples: "has no reported adverse events" → "is safe in"; "is involved in pathogenesis of" → "contributes to"; "have not been proven to transmit to" → "does not transmit to"; "is associated with lower risk of" → "reduces risk of"; "have weaker oestrogenic activity than" → "weaker than".
- For comparisons between two entities, use exactly "higher than", "lower than", or "more effective than" as the verb, not longer phrases like "lower urinary estriol excretion than" — split that into a direct effect triplet instead.
- Condense verbose phrases: "is effective and safe as a therapeutic agent in" → "treats".
- No quantitative data or qualifying clauses in verbs.
- Avoid weak scaffold verbs such as "is a", "has", "had", or "was" unless they express a core biomedical relation like "is associated with" or "is adverse effect of".

WHAT TO EXTRACT:
- Medical findings, results, conclusions, and established facts.
- Keep triplets only when their payload is medically useful for later retrieval: diseases, symptoms, biomarkers, body systems, chemicals, drugs, foods, procedures, exposures, mechanisms, risks, protective factors, or clinical outcomes.
- If both sides of a candidate triplet are healthcare operations or policy/process concepts rather than patient-level biomedical concepts, skip it as retrieval noise.
- Skip pure methodology (study design, data collection, randomization procedures).
- If a sentence frames a finding in study language ("the study showed X reduces Y"), extract the finding (X reduces Y), skip the framing.
- Review, meta-analysis, and hypothesis papers can still contain extractable biomedical facts. Extract summarized mechanistic, associative, or clinical findings when they are stated as evidence or a supported conclusion.
- Fundamental biomedical mechanisms are high-value. Extract nutrient sensing, cell response, protein synthesis constraints, immune signaling, toxicologic mechanisms, microbial invasiveness, exposure reservoirs, tissue localization, biomagnification, and disease-trigger relations when the text states them.
- Review and hypothesis papers can still contain extractable biomedical facts. Extract mechanistic, associative, exposure-chain, tissue-presence, toxicologic, and etiologic findings when they are presented as evidence, a supported conclusion, or a motivated disease hypothesis.
- Exposure-burden and nutritional epidemiology findings are high-value when they state medically relevant contaminant intake, toxic exposure routes, tolerable-intake exceedance, major exposure sources, or positive, inverse, or null associations between a nutrient/food/exposure and disease risk.
- Exposure-burden and nutritional epidemiology findings are high-value when they state medically relevant contaminant intake, toxic exposure routes, tolerable-intake exceedance, major exposure sources, or positive, inverse, or null associations between a nutrient/food/exposure and disease risk. Do not skip these just because the paper is observational or population-level.
- Comparative review and meta-analysis results are high-value facts. If a study concludes that treatment X has higher or lower odds/risk than treatment Y for outcome Z, extract those comparative outcome relations.
- Dose-response and physiologic response findings are high-value facts. If a treatment or compound changes an organ function or biomarker at a stated dose, extract the direct effect even when the paper centers on dosage comparisons.
- Risk scores, diagnostic scores, and severity scores are high-value biomedical entities when they are associated with lesion burden, disease severity, prevalence, progression, or outcomes. Extract those score-to-disease or score-to-lesion relations.
- If the text states that compound X is produced by organism Y, accumulates in tissue Z, biomagnifies through a food chain, is present in diseased tissue, or may trigger disease D, extract those relations.
- If the text states that nutrient insufficiency, extracellular nutrient availability, or amino-acid abundance is monitored and elicits a cellular response, extract that mechanism even if the sentence is not tied to a named disease.
- Toxicology and poisoning papers are high-value even when descriptive. Extract toxin identity, route of exposure, affected organ system, poisoning outcome, carcinogenicity, lack of antidote, and medically relevant outbreak or exposure-burden findings when the text states them.
- If a sentence is mainly about policy, administration, stakeholder behavior, adoption of systems, implementation pace, incentives, documentation, institutional goals, or national programs, skip it unless it directly states a biomedical risk, patient harm, symptom, diagnosis, treatment effect, or clinical outcome.
- General healthcare quality/process phrases such as electronic health records, stakeholder engagement, implementation pace, safe practices, national goals, incentives, disclosure programs, or hospital performance are usually noise unless the sentence also states a concrete injury, adverse event, disease, symptom, biomarker, or treatment outcome.
- Editorial or opinion statements about physician behavior, medical education, evidence hierarchies, research culture, or preference for one evidence source over another are usually noise unless they also state a concrete biomedical exposure, intervention effect, diagnosis, symptom, biomarker, or patient outcome.
- Statements about physician bias, medical education gaps, evidence-hierarchy preferences, or research-culture criticism are usually noise. Keep only the embedded biomedical relation itself, not the commentary about who prefers or ignores it.
- Skip low-value meta statements, communication advice, institutional recommendations, adjustment-variable lists, or tautological measurement scaffolding when they do not state a medical fact.
- Skip purely legal, warning-label, or regulatory compliance facts unless they directly state a biomedical risk or clinical consequence.
- General food marketing claims or broad nutritional inventory text are low value. Do not exhaustively enumerate every nutrient, vitamin, mineral, fatty acid, or export/production fact for a food unless a constituent is explicitly tied to a medical effect, toxic exposure, deficiency relevance, protective effect, or clinically meaningful comparison.
- For food composition reviews, prefer at most a small number of medically relevant composition triplets. Keep toxic constituents, protective constituents, or constituents linked to a stated medical effect. Skip generic composition filler like total sugars, fat percentage, generic vitamins, generic minerals, seed weight, export growth, or broad "contains nutrients" statements.
- For food composition reviews, prefer at most 3-5 medically relevant composition triplets for the whole passage. Keep toxic constituents, protective constituents, or constituents linked to a stated medical effect. Skip generic composition filler like total sugars, fat percentage, generic vitamins, generic minerals, seed weight, export growth, broad "contains nutrients" statements, and bare percentage/composition descriptions.

ENTITY TYPING — classify each entity:
{ENTITY_SCHEMA_PROSE}

Typing guidance:
- Lab measurements and in-vitro outcomes (cell growth rates, apoptosis levels, serum levels) → BIO_MARKER
- Health outcomes or status measures such as survival, mortality, recurrence, prognosis, disease progression, weight loss, weight control, aging, health span, quality of life, cognition, executive control, inhibitory control, reward response, food intake, inflammation, and oxidative stress → BIO_MARKER
- Risk scores, diagnostic scores, lesion burden, lesion prevalence, plaque burden, fatty streaks, lesion severity, and physiologic contraction responses → BIO_MARKER
- Clinical symptoms experienced by patients → SIGN_OR_SYMPTOM
- Dietary patterns (low-fat diet, high-protein diet) → FOOD
- Gases, ions, metabolites, and other small chemical substances → CHEMICALS
- Cell types, tissues, organ systems, immune compartments, barriers, and lymphocyte populations → BODY_PART
- Environmental surfaces, fomites, reservoirs, contaminated sites, and exposure locations default to RISK_FACTOR when no better category exists in this schema.
- Environmental surfaces or locations are worth keeping only when linked to a concrete pathogen, contaminant, exposure, or intervention effect. Do not keep bare site-ranking or site-comparison facts by themselves.
- Pathogens, commensal bacteria, microbiota, allergens, food antigens, toxins, and contaminants default to RISK_FACTOR when no better category exists in this schema.
- Proteins, receptors, ligands, cytokines, enzymes, caspases, adapter proteins, transcription factors, and named signaling molecules default to GENES when no better category exists in this schema.
- "placebo" → DRUGS
- "neural pathways" → BODY_PART
- "laser photocoagulation", "pattern scan laser" → HEALTH_PROCEDURES
- NEVER use NONE as an entity type. Every entity must be assigned a real type from the list above. If an entity does not fit any type after shortening, skip the triplet entirely. Common remappings: "operated patients" → DISCRIMINATIVES, "antimicrobial agent" → DRUGS, "skeletal health" → BIO_MARKER, diets/foods always → FOOD, signaling pathways → GENES.

SKIP THESE LOW-VALUE CASES:
- Do not extract model-adjustment or confounder relations such as "BMI confounds X" or lists of covariates adjusted for in regression models.
- Do not create triplets whose main content is that an article, database search, reviewer, or institution reported or recommended something.
- Do not create tautological measurement triplets like (procedure, has effective dose, effective dose).
- Do not create triplets where either entity is only a time label, study phase, arm label, control value, institution, stakeholder group, policy instrument, incentive, or implementation program.
- Do not create triplets whose main meaning is operational improvement, administrative coordination, hospital adoption, payment policy, or documentation workflow.
- Do not create triplets whose main meaning is clinician preference, physician bias, medical education bias, or preference for one evidence source over another.
- Do not create pure location-ranking triplets such as one room or surface being more or less contaminated than another unless the triplet also includes a pathogen, contaminant, or intervention effect.
- Do not create triplets about clinicians, physicians, medical students, training programs, or evidence-based medicine preferring, ignoring, favoring, disfavoring, or being biased toward a treatment or evidence source.
- Do not create triplets of the form (site, more contaminated than, site) or (site, less contaminated than, site). Keep the pathogen- or intervention-specific contamination facts instead.
- Do not explode a food composition table or review paragraph into a long list of generic (food, contains, nutrient) triplets when the text is only describing composition and does not connect those constituents to a medical implication.
- Do not create generic inventory triplets like (food, contains, vitamin), (food, contains, fat), (food, contains, minerals), (seed, weighs, percentage), or export/production statistics unless the passage explicitly connects them to a medical outcome, toxic exposure, or protective effect.
- Do not create bare composition verbs such as contain, consists of, weighs, has percentage of, or includes when they only report ordinary nutritional inventory without a medical implication.

DEDUPLICATION:
- Skip truly identical (e1, verb, e2) triplets. But related-but-distinct entities (e.g., "diet and exercise" vs "exercise alone") are separate — extract both if the text reports separate findings for each.
- If one sentence states the same fact at both a general and a more specific level, keep the more informative triplet and drop the redundant weaker one.

COMPARISON HANDLING:
- If the text says X is more effective than Y at reducing Z, extract (X, more effective than, Y) and the direct outcome triplet(s) such as (X, reduces, Z). Do not merge comparison and outcome into one verbose verb.

EXAMPLES:
Text: "Statins reduce the risk of cardiovascular disease."
✓ (statins : DRUGS) --[reduce risk of]--> (cardiovascular disease : DISEASE_DISORDER)

Text: "Sexual dysfunction is an adverse effect of antidepressant drugs."
✓ (sexual dysfunction : SIGN_OR_SYMPTOM) --[is adverse effect of]--> (antidepressant drugs : DRUGS)

Text: "Krill oil was more effective than fish oil for reducing LDL."
✓ (krill oil : DRUGS) --[more effective than]--> (fish oil : DRUGS)
✓ (krill oil : DRUGS) --[reduces]--> (LDL : BIO_MARKER)

Text: "Palatable foods activate somatosensory regions and may be hyperresponsive in obese individuals."
✓ (palatable foods : FOOD) --[activate]--> (somatosensory regions : BODY_PART)
✓ (obesity : DISEASE_DISORDER) --[is associated with]--> (hyperresponsive somatosensory regions : BIO_MARKER)

Text: "Rosiglitazone was associated with higher odds of myocardial infarction than pioglitazone."
✓ (rosiglitazone : DRUGS) --[higher risk than]--> (pioglitazone : DRUGS)
✓ (rosiglitazone : DRUGS) --[increases risk of]--> (myocardial infarction : DISEASE_DISORDER)

Text: "Forty mg curcumin produced a 50% gall bladder contraction."
✓ (40 mg curcumin : DRUGS) --[induces]--> (gall bladder contraction : BIO_MARKER)
✓ (curcumin : DRUGS) --[affects]--> (gall bladder : BODY_PART)

Text: "Risk scores were associated with early atherosclerotic lesions."
✓ (atherosclerosis risk scores : BIO_MARKER) --[are associated with]--> (early atherosclerotic lesions : BIO_MARKER)

Text: "Amino acid insufficiency for protein synthesis is actively monitored by eukaryotic cells, eliciting cellular responses."
✓ (eukaryotic cells : BODY_PART) --[monitor]--> (amino acid insufficiency : RISK_FACTOR)
✓ (amino acid insufficiency : RISK_FACTOR) --[elicits]--> (cellular responses : BIO_MARKER)

Text: "Cereal products and potatoes contribute more than 60% to dietary cadmium intake, and 2% of adults exceed the tolerable weekly intake."
✓ (cereal products : FOOD) --[contribute to]--> (dietary cadmium intake : CHEMICALS)
✓ (potatoes : FOOD) --[contribute to]--> (dietary cadmium intake : CHEMICALS)
✓ (dietary cadmium intake : CHEMICALS) --[exceeds]--> (tolerable weekly intake : RISK_FACTOR)

Text: "Isoflavone intake was associated with lower risk of ovarian cancer, whereas isothiocyanate intake was not associated with risk."
✓ (isoflavone intake : FOOD) --[reduces risk of]--> (ovarian cancer : DISEASE_DISORDER)
✓ (isothiocyanate intake : FOOD) --[is not associated with]--> (ovarian cancer : DISEASE_DISORDER)

Text: "Dates contain sugars, fat, vitamins, and minerals; elemental fluorine may protect against tooth decay."
✓ (elemental fluorine : CHEMICALS) --[protects against]--> (tooth decay : DISEASE_DISORDER)
Do NOT extract generic inventory triplets like (dates, contains, sugars) or (dates, contains, vitamins) from this passage.

Text: "Dates contain sugars, fat, vitamins, minerals, and fiber; selenium supports immune function and fluorine protects against tooth decay."
✓ (selenium : CHEMICALS) --[supports]--> (immune function : BIO_MARKER)
✓ (elemental fluorine : CHEMICALS) --[protects against]--> (tooth decay : DISEASE_DISORDER)
Do NOT extract generic inventory triplets such as (dates, contains, sugars), (dates, contains, fat), (dates, contains, vitamins), or (dates, contains, minerals).

Extract every factual medical relationship. Multiple triplets per sentence are expected. Do NOT add information not in the text."""


# The tool is never executed. It exists so the provider will constrain decoding to this JSON shape;
# we read the arguments the model was forced to produce and throw the call away.
EXTRACT_RELATIONS_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "store_extracted_relations",
        "description": (
            "Store all extracted entity-relationship-entity triplets from the medical text "
            "with their entity types."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "triplets": {
                    "type": "array",
                    "description": "Array of all extracted triplets from the text.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "e1": {
                                "type": "string",
                                "description": "First entity -- a short noun phrase (max 6 words) as mentioned in the text.",
                            },
                            "e1_type": {
                                "type": "string",
                                "enum": list(EXTRACTION_TYPES),
                                "description": "The category of entity_1 based on the medical schema.",
                            },
                            "verb": {
                                "type": "string",
                                "description": "The core relationship verb in 1-4 words (e.g. 'treats', 'causes', 'reduces risk of').",
                            },
                            "e2": {
                                "type": "string",
                                "description": "Second entity -- a short noun phrase (max 6 words) as mentioned in the text.",
                            },
                            "e2_type": {
                                "type": "string",
                                "enum": list(EXTRACTION_TYPES),
                                "description": "The category of entity_2 based on the medical schema.",
                            },
                        },
                        "required": ["e1", "e1_type", "verb", "e2", "e2_type"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["triplets"],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------------
# RETRIEVAL -- soft pattern generation
#
# The query side of the method. A user question is rewritten into one or more soft graph patterns
#
#     MEM( entity ^ relation ^ entity )
#
# where any of the three slots may be (a) a concrete mention, (b) a schema TYPE, or (c) "?", a
# wildcard. The pattern is not a database query: nothing in it has to match a stored string. Slots
# are resolved by vector similarity against the corresponding subspace of the graph, and TYPE slots
# resolve by exact match against a node's assigned type. That combination -- fuzzy on text, exact on
# type -- is what "soft" means here, and it is why the graph tolerates the vocabulary mismatch that
# makes strict Cypher generation over an extracted KG so brittle.
#
# Two variants are provided because the shape of the generated pattern turns out to matter more than
# the wording:
#
#   SEARCH_PATTERNS        the permissive original. It *offers* schema types ("if intent of the user
#                          can be queried by a category ... we can use the category names as
#                          entity") and leaves the count open. Strong instruction-following models
#                          take the cheapest legal option: a single MEM( concrete ^ ? ^ ? ), which
#                          degenerates to plain dense retrieval on one entity string and leaves the
#                          typed machinery idle.
#
#   SEARCH_PATTERNS_TYPED  makes a TYPE mandatory in at least one slot and demands 4-8 patterns
#                          covering both directions. Same schema, same output format, same task --
#                          only the instruction about types changes, which keeps the two comparable
#                          as an A/B over pattern shape alone.
#
# Keeping both in the repo is intentional: which prompt is used determines whether the mechanism
# under study actually fires, so an experiment that does not pin it down is not interpretable.
# ---------------------------------------------------------------------------
SEARCH_PATTERNS = f"""Task: Generate appropriate memory api calls to neo4j in the "MEM( entity ^ relation ^ entity )" format
Instructions:
According to the schema of neo4j graph you should call the api to invoke related information to the provided query.
First its necessary to recognize entities that can be categorized accordin to the schema Node properties.
The recognized entities can be directly be used in api call in entity sections. if intnent of the user can be queried by a category of schema then we can use the category names as entity.
Relation part in the api call is the most possible verb that can potentially represent the relashionship.
if the information can be queried in different styles return all possible memory calls in a line.
Example 1: "what disease cause cold feet" answer: "MEM(cold feet ^ caused by ^ DISEASE_DISORDER ) \n MEM( DISEASE_DISORDER ^ causes ^ cold feet )"
Example 2: "how should i know if i have diabetes? answer: "MEM( diabetes ^ leads to ^ SIGN_OR_SYMPTOM ) \n MEM( SIGN_OR_SYMPTOM ^ is symptom of ^ diabetes )"
Example 3: "what is the relationship betweeen CVA and blood sugare" answer: "MEM( CVA ^ ? ^ blood sugare ) \n MEM( blood sugare ^ ? ^ CVA)"
Example 4: "how to stay healthy?" answer: "MEM( ? ^ is good for ^ health) \n MEM( ? ^ should be avoided for ^ health)"
Example 5: "is avoiding trans fats beneficial" answer: "MEM( avoid trans fats ^ results in ^ ? ) \n MEM( avoid trans fats ^ can improve ^ ? )"
Example 6: "heart-healthy diet" answer: "MEM( heart-healthy diet ^ ? ^ ? )"
Example 7: "how to control cholesterol" answer: "MEM( ? ^ helps ^ cholesterol control) \n MEM( ? ^ help for control ^ cholesterol)"
Example 8: "avoid heart diseases for me" answer: "MEM( ? ^ prevents ^ heart disease) \n MEM( ? ^ reduce chance of ^ heart disease)"
Example 9: "how long excersice should take for cardio vascular fitness" answer: "MEM( vascular fitness ^ should take ^ ?)"
Schema:
{SCHEMA_BLOCK}
Note: Do not include any prefix, explanations or apologies in your responses."""


SEARCH_PATTERNS_TYPED = f"""Task: Generate memory api calls to neo4j in the "MEM( entity ^ relation ^ entity )" format.

HARD REQUIREMENTS (a response violating any of these is invalid):
1. Output BETWEEN 4 AND 8 calls, one per line. Nothing else -- no prose, no numbering, no blank lines.
2. In EVERY call, AT LEAST ONE of the two entity slots MUST be a schema TYPE written exactly in
   UPPERCASE from this list: {', '.join(QUERY_TYPES)}.
3. Use the concrete entities from the user's query in the OTHER slot. Never put a bare "?" in both
   entity slots.
4. The relation slot must be a plausible verb phrase. Use "?" there only if no verb is implied.
5. Cover BOTH directions where it makes sense: (concrete ^ verb ^ TYPE) and (TYPE ^ verb ^ concrete).
6. Prefer the TYPE that answers the user's actual information need (e.g. a "what food helps X"
   question needs FOOD; a "what disease causes X" question needs DISEASE_DISORDER).

Example 1: "what disease cause cold feet"
MEM( cold feet ^ caused by ^ DISEASE_DISORDER )
MEM( DISEASE_DISORDER ^ causes ^ cold feet )
MEM( cold feet ^ is symptom of ^ DISEASE_DISORDER )
MEM( RISK_FACTOR ^ increases ^ cold feet )
MEM( cold feet ^ affects ^ BODY_PART )

Example 2: "Breast Cancer Cells Feed on Cholesterol"
MEM( breast cancer ^ fed by ^ CHEMICALS )
MEM( CHEMICALS ^ promotes ^ breast cancer )
MEM( cholesterol ^ increases risk of ^ DISEASE_DISORDER )
MEM( FOOD ^ raises ^ cholesterol )
MEM( DRUGS ^ lowers ^ cholesterol )
MEM( breast cancer ^ measured by ^ BIO_MARKER )

Example 3: "how to control cholesterol"
MEM( FOOD ^ lowers ^ cholesterol )
MEM( DRUGS ^ lowers ^ cholesterol )
MEM( ACTIVITY ^ reduces ^ cholesterol )
MEM( cholesterol ^ controlled by ^ HEALTH_PROCEDURES )
MEM( cholesterol ^ is ^ BIO_MARKER )

Schema:
{SCHEMA_BLOCK}
Note: Do not include any prefix, explanations or apologies in your responses."""


SEARCH_PROMPTS: dict[str, str] = {
    "permissive": SEARCH_PATTERNS,
    "typed": SEARCH_PATTERNS_TYPED,
}
