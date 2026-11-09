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

## License

Apache-2.0
