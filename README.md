# Rivu

## Overview
Rivu is a Rust reimplementation of incremental learning ideas popularized by the [Massive Online Analysis (MOA)](https://github.com/Waikato/moa/tree/master/moa/src/main/java/moa) framework. It focuses on prequential evaluation (test-then-train) for streaming classification with learners such as Naive Bayes and Hoeffding Trees, while providing an interactive command line wizard and real-time console output.

## Features
- **Prequential evaluation runner** – Interleaves prediction and training while honoring optional limits on processed instances and wall-clock time. Periodically samples performance metrics and RAM-hours usage so you can track drift and resource consumption during execution.
- **Interactive CLI wizard** – Guides you through picking a task, configuring a stream, evaluator, and learner. Each prompt includes contextual help, default values, and validation (including `.arff` path checks) to keep configuration friction low.
- **Streaming data sources** – Supports `.arff` file streams and synthetic generators for SEA, Agrawal, and Asset Negotiation concepts. Generators expose knobs for seeds, class balancing, noise, drift, and concept duration.
- **Incremental learners** – Ships with a classic Naive Bayes classifier and a configurable Hoeffding Tree (VFDT) that lets you choose the numeric estimator, split criterion, and leaf prediction strategy.
- **Online metrics** – Basic classification evaluator emits accuracy, Cohen's kappa, optional precision/recall/F1 aggregates, and per-class statistics. Snapshots feed the live console renderer to display throughput, accuracy, kappa variants, elapsed time, and RAM-hours.

## Getting Started

### Prerequisites
- Rust 1.76 or newer (Edition 2024) with Cargo.

### Download a release
Grab the latest build from the project's GitHub Releases page. On macOS you can install the signed package and then launch the CLI with `rivu`. For Linux and Windows download the corresponding archive and run the `rivu` executable directly; these builds are not signed, so follow your platform's guidelines for running unsigned binaries.

### Launch the wizard
```bash
cargo run
```
Select the prequential evaluation task and answer the wizard prompts for stream, evaluator, and learner. The runner prints a header describing the session and refreshes a live status line with metrics, throughput, and progress bars.

### Run a task non-interactively
If you already know the configuration you want, the `rivu run` subcommand lets you provide everything up front without stepping through the wizard. The flags mirror the wizard prompts and accept additional `--*-param key=value` overrides for nested settings.

For example, the following command mirrors MOA's `EvaluatePrequential` invocation shown above, training a Hoeffding Tree on the `covtypeNorm.arff` stream while reporting basic classification metrics:

```bash
rivu run \
  --task evaluate-prequential \
  --learner hoeffding-tree \
  --stream arff-file \
  --evaluator basic-classification \
  --stream-param path=/Users/rafaelvenetikides/Developer/rivu/data/covtypeNorm.arff
```

The `split_criterion` in the Hoeffding Tree defaults to Gini (matching MOA's `-s GiniSplitCriterion` flag), so no extra learner override is required. If a value contains spaces, wrap it in quotes so the shell passes it through as a single argument. Any unspecified flags fall back to the same defaults used by the interactive wizard.

### Run the test suite
```bash
cargo test
```
The tests cover the prequential evaluator's guards, curve updates, UI schema helpers, and utility modules that support the CLI and evaluation pipeline.

## Sample Data
Example `.arff` files are available under `data/` (`airlines`, `covtypeNorm`, and `giveMeLoanKaggle`). Use the "Arff File Stream" option in the wizard and supply one of these paths along with the zero-based class index to get started quickly.

## Project Structure
```
src/
├── classifiers/        # Naive Bayes and Hoeffding Tree implementations
├── core/               # Shared instance headers and type utilities
├── evaluation/         # Online metrics, snapshots, and evaluators
├── streams/            # ARFF reader and synthetic stream generators
├── tasks/              # Prequential evaluator orchestration
├── ui/                 # CLI wizard, prompt drivers, and schema builders
├── utils/              # Math, parsing, and system helpers
├── testing/            # Test doubles for learners, streams, evaluators
└── main.rs             # CLI entry point with live console renderer
```

## Development Workflow
1. Format the codebase with `cargo fmt`.
2. Run lint checks via `cargo clippy`.
3. Execute the automated tests with `cargo test` before opening a pull request.

To add a new learner, stream, or evaluator:
- Implement the component under `src/classifiers`, `src/streams`, or `src/evaluation`.
- Expose it in the CLI by extending the corresponding enums in `src/ui/types/choices/`.
- Update the builders in `src/ui/types/build/` so the wizard can construct the new option from user selections.

## License
Rivu is distributed under the AGPL-3.0 license.
