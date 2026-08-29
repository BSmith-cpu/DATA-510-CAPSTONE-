# Where Next? Predicting the Next Unaffordable U.S. City

Predicting which affordable, mid-sized U.S. metros are at risk of tipping into a housing affordability crisis, using public housing and economic indicators (Zillow, FHFA HPI, Census ACS, FRED, Census population estimates).

**Author:** Brandon Smith

## Repo contents

| Path | Purpose |
|------|---------|
| [`notebooks/`](notebooks/) | Exploration, data summary, feature/model, and figure notebooks. |
| [`data/`](data/) | Raw and merged datasets, plus generated plots. |
| [`deliverables/`](deliverables/) | Milestone write-ups: proposal, posters, final write-up. |

## Project summary

The model ranks U.S. metros by affordability risk, using price-to-income ratio, affordability momentum, and population velocity as key features. Training cities: Tampa, Austin, and Boise. Target definition: price-to-income ratio >= 5.0 sustained over 3+ consecutive years.

See [`deliverables/M15-Final Writeup/writeup.pdf`](deliverables/M15-Final%20Writeup/writeup.pdf) for the full methodology and results.
