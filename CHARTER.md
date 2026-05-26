# Studio Charter: <project name>

> Filled in live during the **Studio Charter** session in week 3. Every section below is committed in the same commit at the end of that class block. See [Studio Charter (single-session inception)](https://courses.lpcordova.phd/data510/project-framework/charter-inception.html) for the script and time-boxes.

* **Owner team:** Brandon Smith
* **Owner Product Lead:** Brandon Smith
* **Peer Stakeholder POs:** Manish Reddy Kallu, Alex Jackson, Jackson Garro
* **Instructor / Sponsor:** Lucas Cordova (`LucasCordova` on GitHub)
* **GitHub repo:** [<link to this repo>](https://github.com/BSmith-cpu/DATA-510-CAPSTONE-)
* **GitHub Projects board:** [<link>](https://github.com/users/BSmith-cpu/projects/4)
* **Discord category:** `#<project>-*`
* **Studio Session:** 1
* **Studio formed:** 5/25/2026

## Vision


We want to build a reproductive model that ranks U.S. metros by affordability risk using publicly available federal data sources. When the project succeeds, the majority of middle-class Americans will be able to identify cities they should avoid based on key indicators of prior cities that have priced out the middle class. 


## Mission

The mission is to predict which afforable U.S. mid-sized cities are next to tip into a housing crisis and pricing out the middle class. 

## Context

- **Users / affected parties:** **Who Benefits** city/county housing departments, mortgage lenders, state housing financial planner | **Who is at Risk**: working/middle class who are either renting/in process of buying | **Who might use the results** Real Estate agents trying identify an emerging markets. 
- **Data sources (proposed):** Zillow Research, Free/Public Download, NO PII | FHAH Housing Price Index, Free/public API key, U.S. government open data | Census ACS, Free via Census API, Public Domain | FRED (Federal Reserves), Free API, Open/No Restrictions | Census Population Estimates, Free/Public Download, Public Domain
- **Constraints:** **Time**: I only have 1 semesmeter (~15 weeks) which limits the amount of metro cities can be modelded | **Scope**: Limited to metros with sufficent historic data (2000 to present); this excluded rural/smaller cities. | **Access**: Even though most of the data was able to be downloaded locally, there are rate limiters on the API keys. 
- **Ethics risks:** **PII/Consent**: All of the data is public, aggregate, and anonymized at the metro level | **Fairness**: The model might to try to flag lower-income or majority middle-class cities, this might be something that we have to address. 

## Success criteria by milestone

- **M1, proposal (W4):** All 5 data sources accessed and a SQL schema drafted with CBSA as primary key
- **M2, data summary (W7):** A joined, clean panel dataset existing more than 75 metros with EDA complete.  
- **M3, poster rough draft (W10):** A baseline model is trained and a rough draft poster exists for prelimiary results. 
- **M4, write-up rough draft (W12):** Final model is tuned, SHAP values computed, and a full draft write-up is complete. 
- **M5, final write-up and poster (W14):** Polished write-up, working dashboard, and public GITHUB repo are all submitted. 

## Working agreements (internal to owner team) - SOLO

- **Sync rhythm:** Monday standup, to review what needs to be done by EOW (end of week); this just keeps myself accountable. 
- **Code review:** I will be writing the code, let it run, wait 24 hours/let another classmate/peer stakeholder review it before pushing it. 
- **Decision rule:** Solo project, but if I have confliciting opinons consult others opinon/judgement. 

## Working agreements (triad with peer POs)

- **Studio Brief due:** <example: by 5 pm the day before class, committed to `studio/briefs/W<NN>-<peer>.md` and linked in `#<project>-studio` on Discord>. If the owner team needs the peer POs to read or review something specific *before* the Studio Session (a data preview, model results, a draft figure), file the Brief earlier so the peer POs actually have time to do that homework. Otherwise the default is "before the Studio Session starts."
- **Studio Critique due:** <example: by the end of class for the in-person discussion, or at an agreed-upon time within one day after class (e.g., 5 pm the next day) if the peer PO needs extra time to draft a thoughtful write-up>.
- **Priority conflict resolution:** owner team integrates briefs in good faith; the instructor arbitrates (as Process Expert) if peer POs and owner team disagree.

## Response SLAs (Service Level Agreements)

A **Service Level Agreement** is a written promise the triad makes about *how fast* each side responds when a specific signal arrives. Every row must have an answer before this Charter is committed. See [Response SLAs](https://courses.lpcordova.phd/data510/project-framework/charter-inception.html#response-slas-service-level-agreements) for the full definition.

| When this signal arrives... | Who responds | By when |
|-----------------------------|--------------|---------|
| Peer PO files a **Studio Brief** (commits to `studio/briefs/...`, links in `#<project>-studio`) | Owner team | <e.g., acknowledge in `#<project>-studio` within 24 hours, with a first-pass adopt / defer / decline call for each item> |
| Peer PO files a **Studio Critique** | Owner team | <e.g., respond in `#<project>-studio` within 24 hours and capture follow-up items into the backlog> |
| Owner team posts an **Iteration Review** in `README.md` | Both peer POs | <e.g., read before filing the next Brief and Critique> |
| Owner team flags a **blocker** in `#<project>-blockers` | Instructor, plus any tagged peer PO | <e.g., responds by the next Studio Session at the latest; faster if online> |
| Anyone asks a clarifying question in `#<project>-general` | Whoever is tagged (default: owner team) | <e.g., reply within 48 hours, even if the reply is "we will look at this next iteration"> |

## Definition of Ready (PBI)

A PBI is ready to be pulled out of `Backlog` and moved into `Create` when it has:

- A one-sentence hypothesis or user story.
- A named **Create**, **Observe**, **Analyze** triple.
- A milestone tag (`M1-proposal`, `M2-data-summary`, `M3-poster-draft`, `M4-writeup-draft`, `M5-final`, `infra`, `ethics`).
- A T-shirt size estimate (S, M, L, XL).
- WIP slack on the board: `Create + Observe + Analyze` is below the team's WIP cap (owners + 1).

## Definition of Done (PBI)

A PBI is done, and may be moved from `Analyze` into `Done`, when:

- The Create artifact is in the repo or linked from the issue.
- The Observe results are recorded somewhere referenceable (notebook output, processed dataset, draft results section).
- The Analyze writeup names a next step (continue, pivot, kill, or decompose into new PBIs).
- A peer PO has either signed off in `#<project>-studio` or filed a Studio Critique covering it.
- The card is linked under *Completed PBIs* in the next Iteration Review in `README.md`.

## Context map

> Optional. Replace this block with a Mermaid `flowchart LR` showing how users, data, constraints, and ethics risks flow into the owner team and out to the capstone outcome. See the [`charter-inception.qmd` template](https://courses.lpcordova.phd/data510/project-framework/charter-inception.html) for a starting Mermaid diagram.

## Stakeholder alignment memo (one-page summary)

### Why we exist
<two sentences from Vision and Mission>

### What we will deliver to peer POs every week
- An Iteration Review in this `README.md` by <day / time>
- A summary of which Studio Brief items we adopted, deferred, or declined and why

### What we need from peer POs every week
- A Studio Brief by <day / time> next class (next iteration's requirements, questions, risks)
- A Studio Critique by <day / time> next class (assessment of last week's delivery)

### How to reach us
- Discord category: `#<project>-general` (day-to-day), `#<project>-studio` (Briefs and Critiques), `#<project>-blockers` (impediments)
- GitHub repo: <link>
- GitHub Projects board: <link>
