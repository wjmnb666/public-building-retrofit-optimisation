# Data provenance and reuse

The repository contains aggregate research inputs, code and declared assumptions. It excludes individual building addresses, certificate identifiers, participant identifiers, response-level survey data and restricted operational records.

## Public input sources

- UK Energy Performance of Buildings data service, non-domestic EPC and DEC. Aggregate archetype and Manchester prototype statistics from the publication archives. https://epc.opendatacommunities.org/
- Department for Energy Security and Net Zero, Public Sector Decarbonisation Scheme publications. Technology-level project counts and spending aggregates. https://www.gov.uk/government/collections/public-sector-decarbonisation-scheme
- DESNZ, Non-domestic National Energy Efficiency Data-Framework 2025. Office size-band energy factors.
- DESNZ, Green Book energy and greenhouse-gas data tables, November 2023. Electricity factor series used in the retained model.
- DESNZ, Quarterly Energy Prices, June 2026 update. Q1 2026 non-domestic energy prices including the Climate Change Levy.
- UK Government greenhouse-gas conversion factors 2026, version 1.1. Gross-calorific-value natural gas factor.

Exact table, sheet, cell and SHA-256 provenance for the ENB factors appears in `enb/data/external_parameter_provenance.csv`. These are frozen research inputs. Re-running with a newer source edition creates a different analysis and should retain a new provenance record.

The F-AHP file contains published aggregate weights and consistency information. It contains no participant-level judgements. The LLM files contain public task passages, de-identified review decisions and condition/task-level aggregates. They contain no individual reviewer identity or confidential source passage.

The original providers' terms continue to govern reused data. Public availability does not override attribution or reuse conditions. Code and documentation remain copyright of their respective authors. No new blanket licence over third-party data is asserted by this package.
