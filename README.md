# Public Building Retrofit Whole-Life Cost and Carbon Optimisation

This repository models public-building retrofit choices through whole-life cost, carbon and service criteria. It provides scenario prediction, constrained intervention sequencing, exact pathway comparison and source-linked parameter extraction for the following studies:

- Tao, Z., Baguley, P., Maqbool, R. and Ejohwomu, O. (2026). AI-driven Whole-Life Costing for Decision-Ready UK Public Building Retrofits. *Complex & Intelligent Systems*. https://doi.org/10.1007/s40747-026-02516-9
- Tao, Z., Baguley, P., Maqbool, R. and Ejohwomu, O. (2026). Balancing cost and carbon: prioritising office retrofit pathways in Manchester. *Energy & Buildings*, 371, 118230. https://doi.org/10.1016/j.enbuild.2026.118230

This package runs from public-source aggregates and declared scenario parameters. No confidential organisational rows, interview transcripts, individual survey responses, signed consent forms or contact details are included.

## Run

Repository: https://github.com/wjmnb666/public-building-retrofit-optimisation

[Open the reproduction notebook in Google Colab](https://colab.research.google.com/github/wjmnb666/public-building-retrofit-optimisation/blob/main/reproduce.ipynb)

```bash
git clone https://github.com/wjmnb666/public-building-retrofit-optimisation.git
cd public-building-retrofit-optimisation
```

Use Python 3.11 or later in a clean environment.

```bash
python -m pip install -r requirements.txt
python -m unittest -v
python run_reproduction.py
```

The default run enumerates **5,456** monotonic upgrade schedules for each of three Manchester office prototypes under the central scenario. It saves feasible paths, Pareto membership, preferred schedules and 30-year ledgers in `outputs/`. It also recalculates the retained LLM human-review totals without calling a model service.

```bash
python run_reproduction.py --all-scenarios
python cais/scripts/run_bcnn_lstm_revision.py
python cais/scripts/run_dqn_revision.py
python verify_outputs.py
```

The all-scenarios run evaluates **278,256** candidate prototype–scenario–path combinations across 17 registered scenarios. The other commands regenerate public-building scenario trajectories, surrogate metrics, interval estimates and constrained action sequences. Outputs are calculated scenarios, not observed post-retrofit outcomes. Open `reproduce.ipynb` from this folder in Jupyter, or use the Colab link above. The notebook provides the same commands.

## Evidence and version map

| Route | Input and implementation | Scope |
|---|---|---|
| CAIS public-building scenario | DEC archetype aggregates, PSDS technology aggregates and published aggregate F-AHP weights | Re-runs the archived scenario surrogate and intervention-sequencing scripts. Preserves the publication implementation and seed 1702. |
| ENB Manchester extension | Three aggregate prototypes, frozen public energy/carbon factors and registered package assumptions | Exact 30-year ledger and enumeration for hold and three cumulative upgrade levels. Source code is copied from the retained publication archive, with a portable aggregate-data interface. |
| LLM evaluation | Exact prompts, schemas, model configuration, public task passages and 612 de-identified review decisions | Recalculates review totals for the 612-record gpt-5.6-sol experiment. Does not make live LLM calls or claim identical regenerated responses. |
| Earlier thesis development summaries | Historical reported results | These have different data populations and target definitions. The result map does not equate a new public-data run with every historical figure or table. |

The CAIS encoder uses fixed seeded convolutional/recurrent representations and a regularised fitted output layer. Its archived configuration contains manuscript epoch and batch labels; those labels do not represent end-to-end gradient training in this script. The DQN uses its own action-value network and direct scenario reward calculation. These runnable implementations are documented as they execute.

The ENB route retains planning-level carbon boundaries, diagnostic EPC scaling, package lifetimes, replacement timing and residual-value rules in `enb/config/accepted_assumptions.yaml`. Its parameter settings are distinct from early fixed-carbon-price illustrations in the thesis. `enb/data/external_parameter_provenance.csv` records public-source editions, cells, units and hashes. Government source terms remain applicable to derived data.

The earlier 60-item LLM extraction check lacks a complete retained model/prompt record. Chapter 4 Table 4.25 instead reports the documented 612-record evaluation. Its human-review labels distinguish accept, accept with edit and reject; they are not raw extraction-accuracy labels. Generic narrative outputs and parameter extraction are separate task types. The retained review has one author-rater.

## Validation on 16 November 2026

All three commands above and the tests run successfully. The 153 preferred prototype–scenario–preference selections agree with the retained ENB archive. Maximum cost difference is £0 and maximum carbon difference is below 1.1e-8 kgCO2e. Selected annual ledgers reconcile within 1e-6. `reference/enb/` contains the original preferred-result snapshot and a comparison report. `reference/cais/` retains publication snapshots for checking generated results.

`provenance.json` records hashes of copied source files. `MANIFEST.sha256` records hashes of the public package. Source-specific reuse conditions are described in `DATA_AND_REUSE.md`.

## Files

- `cais/scripts/` — portable archived public-building calculations.
- `cais/tables/` — public and published aggregate inputs.
- `enb/src/` — annual transition kernel, ledger and exact enumeration.
- `enb/config/` — explicit package, appraisal and scenario assumptions.
- `enb/data/` — aggregate prototypes and public-source factor series.
- `llm/` — prompt text, schemas, configuration, public task passages and de-identified review decisions.
- `reference/` — retained result snapshots and validation report.

Case-study instruments and ethical approval details are documented in the thesis appendices. Confidential case data require their existing institutional access arrangements and are not distributed through this repository.
