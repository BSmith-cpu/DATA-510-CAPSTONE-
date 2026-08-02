# <Project Name>

> One-sentence description of what this project is about. Replace this entire block before week 3.

## Quick reference

| Field | Value |
|-------|-------|
| Owner team | Brandon Smith |
| Owner Product Lead | Brandon Smith |
| Peer Stakeholder POs | Manish Reddy Kallu, Alex Jackson, Jackson Garro |
| Studio Session | 1 |
| GitHub repo | (https://github.com/BSmith-cpu/DATA-510-CAPSTONE-) |
| GitHub Projects board | (https://github.com/users/BSmith-cpu/projects/4) |
| Discord category | `#<project>-*` |
| Instructor / Sponsor | Lucas Cordova (`LucasCordova` on GitHub) |

## What this repo contains

| Path | Purpose |
|------|---------|
| [`CHARTER.md`](CHARTER.md) | Studio Charter: vision, mission, context, success criteria, working agreements, SLAs, DoR / DoD. Committed at the end of the week 3 Studio Charter session. |
| [`BACKLOG.md`](BACKLOG.md) | Human-readable mirror of the GitHub Projects board. |
| [`studio/briefs/`](studio/briefs/) | Weekly Studio Briefs from peer POs (`W<NN>-<peer>.md`). |
| [`studio/critiques/`](studio/critiques/) | Weekly Studio Critiques from peer POs (`W<NN>-<peer>.md`). |
| [`src/`](src/) | Working code (scripts, modules). |
| [`notebooks/`](notebooks/) | Exploratory and reporting notebooks. |
| [`data/`](data/) | Project data. Raw inputs are `.gitignored` by default; see `data/README.md`. |
| [`deliverables/`](deliverables/) | Milestone deliverables: proposal, data summary, poster, write-up. |

## How this project runs (DS3 in one paragraph)

This project is run as a **DS3 studio**: the owner team is paired with two or three **peer Stakeholder POs** drawn from adjacent capstone projects. Every week the peer POs file a **Studio Brief** for the next iteration and a **Studio Critique** of the last iteration. The owner team commits an **Iteration Review** here in `README.md` before each class. See the [Studio Session weekly ritual](https://courses.lpcordova.phd/data510/project-framework/weekly-ritual.html) for the cadence and [Studio Charter](https://courses.lpcordova.phd/data510/project-framework/charter-inception.html) for the inception session.

---

# Iteration Reviews

One subsection per class week. The owner team commits the new section **before each class** so peer POs can read it before filing the next Brief and Critique. Use the template at the bottom of this file for any extra weeks you add.

## Week 4 -- Proposal milestone (M1)

**Iteration ending:** 06/01/2026
**Milestone tag in focus:** `M1-proposal`

**Completed PBIs**
- PBI-001: Acquire and document
- PBI-002: Draft research question and frame as a testable claim

**In-flight (carrying across the boundary)**
- Early data inventory and source mapping for the housing and economics datasets
- Early drafting of the metro-level EDA plan.
- Narrowing the afforadbility-collapse target defintion 

**Stakeholder response log**
- N/A

**Plan for next iteration**
- Top PBIs (with milestone tags): PBI-003: Building metro level EDA Profile (m2-data-summary)

**Risks and impediments**
- The affordabilty-collapse target stil needs to be fully operationalized
- Data coverage may vary across metros and years.
- Scope creep could weaken the propsal if too many feature are added to soon. 

## Week 5

**Iteration ending:** 06/08/2026
**Milestone tag in focus:** `M1-proposal` / `M2-data-summary`

**Completed PBIs**
- None
- WIP: Build metro-levle EDA Profile

**Stakeholder response log**
- N/A

**Plan for next iteration**
- Finalize the target definition and EDA profiles for each metro
- Work on feature engineering setup

**Risks and impediments**
- Some series may need harmonization before modeling
- Missingness patterns could distort comparisions if not handled carefully
- The next phase depends on whether the EDA has exposed enough signal to justify the features. 

## Week 6

**Iteration ending:** 6/15/2026
**Milestone tag in focus:** `M2-data-summary`

**Completed PBIs**
- Compeleted metro-level EDA profiles for Tampa, Austin, Boise
- Resolved missing years in FHFA HPI series for Tampa
- Finalized affordability-collapse target definition: price-to-income ratio >= 5.0 sustained over 3 consecutive years

**Stakeholder response log**
- N/A

**Plan for next iteration**
- Begin feature engineering: price-to-income ratio trends, affordability momentum score

**Risks and impediments**
- ACS income daat lags by 1-2 years
- Tampa metro boundary changed mid-series

## Week 7 -- Data summary milestone (M2)

**Iteration ending:** 6/22/2026
**Milestone tag in focus:** `M2-data-summary`

**Completed PBIs**
- PBI-003: metro-leve EDA profiles finzlized for all three training cities
- EDA summary notboook commited to repo with ggplot and ploty visulizations
- Affordability-collapse target fully operationlized and applied to training cities
- All five data sources joined through rmd via CBSA metro codes; merged_full.csv was exported

**Stakeholder response log**
- N/A

**Plan for next iteration**
- Begin PBI-004: Feature enineering pipline in python
- Being identify candidate "at-risk" cities for model inference

**Risks and impediments**
- Merged dataset has 14% missingness in permit data cross smaller metros
- Feature leakage risk if collapse-year data bleeds into preditor window; need strict temporal cutoff enforcement

**Retrospective (milestone boundary)**
- What worked: Joining function to CBSA codes were clean (expect for Tampa which needed a crosswalk)
- What did not: initial affordability target purley on HPI growth was too blunt
- One change for next iteration: Enforea strict train/test temporal split from the start of feature enineering, not at modeling time

## Week 8

**Iteration ending:** 06/29/2026
**Milestone tag in focus:** `M3-poster-draft`

**Completed PBIs**
- PBI-004: Featyre engineering pipline built - price-to-income ratio, momentum score, population velocity
- Population velocity features drafted and added to feature matrix

**Stakeholder response log**
- N/A

**Plan for next iteration**
- Train baseline XGBoost classifier on training cities
- Run intial SHAP anaylsis on baseline model to verify feature signal
- Begin outlining poster structure

**Risks and impediments**
- Only three training cities limit generlizability; will document this as a known scopre constraint
- Some permit data still sprase for Boise pre-2010 - using husing supply pressure proxy where needed

## Week 9

**Iteration ending:** 07/06/2026
**Milestone tag in focus:** `M3-poster-draft`

**Completed PBIs**
- Baseline XGBoost model trainied on Tampa, Austin, Boise with leave-one-city cross-validation
- SHAP summary plot generated - top feature: affordability momentum score, price-to-income ratio trend, population velocity
- Candidate at-risk cities identifed for inference: Bozeman, MT

**Stakeholder response log**
- N/A

**Plan for next iteration**
- Finalize poster draft (intro, methods, results, etc)
- Run model infrence on candidate at-risk cities

**Risks and impediments**
- Poster layout needs to balance techincal depth with acccessibility for non-techincal stakeholders

## Week 10 -- Poster rough-draft milestone (M3)

**Iteration ending:** 07/13/2026
**Milestone tag in focus:** `M3-poster-draft`

**Completed PBIs**
- Created a poster rough drafted centered at the problem definition, research question, stakeholder relevance, and initial EDA
- Included Tampa, Austin, Boise as the project's training-city example.
- Documented the proposed approach

**Stakeholder response log**
- N/A

**Plan for next iteration**
- Revise the poster based on instructor feedback
- Continue target validation and begin only the feature-engineering work that is feasible within the reamining course timeline.
- Organize notebooks, data documentation, and the repository so completed work is reproducible.

**Risks and impediments**
- A complete predictie model may not be feasible after week 10
- the affordabilty-collapase threshold requires transparent justification before it can support model training
- Any results beyond desripitive analysis should be lableded premiliary until validated

**Retrospective (milestone boundary)**
- What worked: Keep Tampa, Austin, and Boise as the consistent training-city examples gave the rpoject a clear analytical narrative
- What did not: The orignial plan assumed modeling could begin before data harmozation and target definition were compelete.
- One change for next iteration: Prioritize a polished, reporoducible EDA and transparent methodologyover inomplete model results.

## Week 11

**Iteration ending:** <date>
**Milestone tag in focus:** `M4-writeup-draft`

**Completed PBIs**
- ...

**Stakeholder response log**
- ...

**Plan for next iteration**
- ...

**Risks and impediments**
- ...

## Week 12 -- Write-up rough-draft milestone (M4)

**Iteration ending:** <date>
**Milestone tag in focus:** `M4-writeup-draft`

**Completed PBIs**
- ...

**Stakeholder response log**
- ...

**Plan for next iteration**
- ...

**Risks and impediments**
- ...

**Retrospective (milestone boundary)**
- What worked: ...
- What did not: ...
- One change for next iteration: ...

## Week 13

**Iteration ending:** <date>
**Milestone tag in focus:** `M5-final`

**Completed PBIs**
- ...

**Stakeholder response log**
- ...

**Plan for next iteration**
- ...

**Risks and impediments**
- ...

## Week 14 -- Final write-up and poster (M5)

**Iteration ending:** <date>
**Milestone tag in focus:** `M5-final`

**Completed PBIs**
- ...

**Stakeholder response log**
- ...

**Final retrospective**
- What worked: ...
- What did not: ...
- What we would change if we ran this project again: ...

---

## Iteration Review template (copy for any extra week)

```markdown
## Week <NN>

**Iteration ending:** <date>
**Milestone tag in focus:** <M1-proposal | M2-data-summary | M3-poster-draft | M4-writeup-draft | M5-final | infra | ethics>

**Completed PBIs**
- ...

**In-flight (carrying across the boundary)**
- ...

**Stakeholder response log**
- Studio Brief from <peer PO>: adopted = ..., deferred = ..., declined (with reason) = ...

**Plan for next iteration**
- Top PBIs (with milestone tags): ...

**Risks and impediments**
- ...
```
