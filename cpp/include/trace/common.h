// Common types and utilities for the TRACE runtime.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <time.h>

namespace trace {

using id_t = uint32_t;
constexpr id_t kInvalidId = 0xFFFFFFFFu;
constexpr float kInf = std::numeric_limits<float>::infinity();

// Execution path taken by a query (Sec. III-D).
enum class Path : uint8_t { Cold = 0, Resolve = 1, Seeded = 2, CacheHit = 3 };

// A candidate with its executor-computed squared distance s(q, x).
struct Neighbor {
  float s;   // squared L2 distance as computed by the executor kernel
  id_t id;   // internal id
  bool operator<(const Neighbor& o) const {
    return s < o.s || (s == o.s && id < o.id);
  }
};

inline uint64_t now_ns() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

inline uint64_t thread_cpu_ns() {
  timespec ts;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull +
         static_cast<uint64_t>(ts.tv_nsec);
}

// Per-query work counters. Every method reports the same counters so that
// tables compare like with like.
struct SearchStats {
  uint64_t n_dist = 0;        // full-vector float32 distance evaluations
  uint64_t n_pq = 0;          // PQ-code distance evaluations (disk tier)
  uint64_t n_aux = 0;         // auxiliary work: cache probes, LSH dot products
  uint64_t n_reads = 0;       // node-page reads issued to storage
  uint64_t n_cache_hits = 0;  // node-cache hits (disk tier)
  uint64_t n_pruned = 0;      // nodes skipped by a bound test
  uint64_t n_hops = 0;        // expansions
  uint64_t io_ns = 0;         // time spent waiting for page reads
  bool early_stopped = false;
  void add(const SearchStats& o) {
    n_dist += o.n_dist; n_pq += o.n_pq; n_aux += o.n_aux;
    n_reads += o.n_reads; n_cache_hits += o.n_cache_hits;
    n_pruned += o.n_pruned; n_hops += o.n_hops; io_ns += o.io_ns;
    early_stopped = early_stopped || o.early_stopped;
  }
};

// Nodes whose full-vector distance was evaluated during a search, with the
// executor-computed squared distance. Used to build residues.
struct VisitRecord {
  std::vector<id_t> ids;
  std::vector<float> s;
  void clear() { ids.clear(); s.clear(); }
  void push(id_t id, float sv) { ids.push_back(id); s.push_back(sv); }
};

[[noreturn]] inline void fail(const std::string& msg) {
  throw std::runtime_error("trace: " + msg);
}

#define TRACE_CHECK(cond, msg)                                  \
  do {                                                          \
    if (!(cond)) ::trace::fail(std::string(msg) + " [" #cond "]"); \
  } while (0)

}  // namespace trace
