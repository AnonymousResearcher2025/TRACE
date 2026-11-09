// Executor interface shared by the in-memory (HNSW) and disk (DiskANN-layout)
// tiers, together with the two hooks TRACE and the baselines plug into:
//   * BoundProvider: a sound lower bound lbS(x) <= s(q, x) used by the prune
//     test that runs immediately before a node's full-vector evaluation.
//   * EarlyStop: a periodic termination check (DARTH-style baseline).
#pragma once

#include <memory>
#include <vector>

#include "trace/common.h"

namespace trace {

class BoundProvider {
 public:
  virtual ~BoundProvider() = default;
  // Returns true and sets *lbS if a sound bound exists for internal id.
  virtual bool lbS(id_t id, float* out) const = 0;
  // Called when the executor skips id because of the bound.
  virtual void on_pruned(id_t /*id*/) const {}
};

// Search-progress features passed to an EarlyStop policy.
struct ProgressFeatures {
  uint64_t n_dist = 0;      // full-vector evaluations so far
  uint64_t n_hops = 0;      // expansions so far
  float s_first = 0.f;      // best squared distance found so far
  float s_kth = 0.f;        // k-th best squared distance found so far
  float s_bound = 0.f;      // admission threshold (L-th best)
  float s_entry = 0.f;      // squared distance of the first evaluated node
  uint64_t n_admit = 0;     // admissions since the previous check
  uint64_t n_eval = 0;      // evaluations since the previous check
};

class EarlyStop {
 public:
  virtual ~EarlyStop() = default;
  virtual uint64_t check_every() const = 0;  // in full-vector evaluations
  // topk_now (internal ids, best first) is provided only if wants_topk().
  virtual bool should_stop(const ProgressFeatures& f,
                           const std::vector<Neighbor>* topk_now) const = 0;
  virtual bool wants_topk() const { return false; }
};

struct SearchParams {
  int k = 10;
  int L = 100;              // candidate-list capacity L_s
  int beam_width = 4;       // disk tier only
  bool collect_visited = true;
  bool want_vectors = false;  // return float32 vectors of the k results
  // With non-empty seeds: also run the cold entry procedure (HNSW upper-layer
  // descent / disk medoid) and add its entry node to the seeds. Used by
  // CatapultDB, whose start set is {entry} plus the catapults.
  bool add_entry_to_seeds = false;
};

struct SearchResult {
  std::vector<Neighbor> topk;     // sorted by (s, id), internal ids
  std::vector<float> vectors;     // k x dim if want_vectors
};

class Executor {
 public:
  virtual ~Executor() = default;
  virtual size_t dim() const = 0;
  virtual size_t size() const = 0;
  virtual const char* tier() const = 0;

  // The executor's own squared-distance kernel s(a, b).
  virtual float dist(const float* a, const float* b) const = 0;

  // Fetches float32 vectors for ids (in memory: copy; on disk: node cache or
  // page reads, which are counted in st).
  virtual void fetch_vectors(const id_t* ids, size_t n, float* out,
                             SearchStats* st) const = 0;

  // Beam search for q. If seeds is empty the search starts from the index
  // entry point (cold). Seeds are internal ids.
  virtual void search(const float* q, const std::vector<id_t>& seeds,
                      const SearchParams& p, const BoundProvider* prune,
                      const EarlyStop* es, SearchResult* out,
                      VisitRecord* visited, SearchStats* st) const = 0;

  // Entry point used by a cold search (HNSW: result of the upper-layer
  // descent is computed per query; this returns the top-level entry point).
  virtual id_t entry_point() const = 0;

  // Internal id -> external label (row index in the corpus file).
  virtual id_t label(id_t internal) const = 0;

  // Index epoch; residues from another epoch are discarded (R3).
  uint64_t epoch() const { return epoch_; }
  void bump_epoch() { ++epoch_; }

 protected:
  uint64_t epoch_ = 0;
};

}  // namespace trace
