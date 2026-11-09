# TRACE

Code for the paper *TRACE: Recycling Transient Search State for Correlated
Vector Query Streams* (under review).

TRACE is a runtime layer that reuses search state across correlated queries
on top of unmodified graph-based ANN indexes (hnswlib and DiskANN).

## Build

Requires Linux, a C++17 compiler, CMake 3.16+, libaio, and Python 3.

```bash
sudo apt-get install -y build-essential cmake libaio-dev python3-dev
pip install -r requirements.txt
./build.sh
```

## Data

The scripts in `scripts/` prepare the workloads (`prep_*.py`), build the
indexes and baseline models (`build_*.py`, `fit_qvcache.py`), and compute
ground truth (`ground_truth.py`, `make_streams.py`). Run any script with
`--help` for its arguments. Paths are set in `configs/aws.yaml`.

## Run

```bash
scripts/run_all.sh memory   # in-memory experiments
scripts/run_all.sh disk     # disk experiments and QVCache fidelity (sudo, for memory caps)
scripts/run_all.sh scale    # concurrency, embedding width, and CatapultDB fidelity
scripts/run_all.sh e2e      # end-to-end RAG (GPU)
scripts/run_all.sh tables   # prints the result tables
```

Results are written to `results/<experiment>/<tier>.json`.

## License

Apache-2.0
