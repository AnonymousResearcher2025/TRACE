// Baselines (Sec. V-A). All run through the same executors, kernel, and
// counters as TRACE.
//   Catapult      CatapultDB entry points (LSH buckets of recent results).
//   DarthStop     DARTH-style learned early termination (GBDT recall model).
//   LogStop       never stops; logs features + current top-k for training.
//   DCacheEngine  D-Cache adapted to graph search (recent queries as pivots).
//   SnakeEngine   Snake Table adapted to graph search (per-node pivot table).
//   QVCacheEngine QVCache answer cache with learned per-region thresholds.
#pragma once

#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "trace/engine.h"

namespace trace {

// --------------------------------------------------------- CatapultDB
class Catapult {
 public:
  Catapult(size_t dim, int n_bits = 8, int bucket_cap = 40, uint64_t seed = 42);
  uint32_t hash(const float* q, SearchStats* st) const;
  std::vector<id_t> seeds(uint32_t h) const;
  void update(uint32_t h, id_t best);
  void reset();
  size_t bytes() const;

 private:
  size_t dim_;
  int bits_;
  size_t cap_;
  std::vector<float> planes_;               // bits x dim
  std::vector<std::vector<id_t>> buckets_;  // front = most recently used
  mutable std::mutex mu_;
};

// ------------------------------------------------------ DARTH-style stop
struct GbdtModel {
  struct Tree {
    std::vector<int> feat, left, right;
    std::vector<float> thr, value;
  };
  std::vector<Tree> trees;
  float base = 0.f;
  void load(const std::string& path);
  float predict(const float* x) const;
};

// Feature vector shared by training (Python) and inference.
constexpr int kDarthFeatures = 9;
void darth_features(const ProgressFeatures& f, float* x);

class DarthStop : public EarlyStop {
 public:
  DarthStop(const std::string& model_path, double target_recall,
            uint64_t interval);
  uint64_t check_every() const override { return interval_; }
  bool should_stop(const ProgressFeatures& f,
                   const std::vector<Neighbor>* topk_now) const override;

 private:
  GbdtModel model_;
  double target_;
  uint64_t interval_;
};

class LogStop : public EarlyStop {
 public:
  explicit LogStop(uint64_t interval) : interval_(interval) {}
  uint64_t check_every() const override { return interval_; }
  bool wants_topk() const override { return true; }
  bool should_stop(const ProgressFeatures& f,
                   const std::vector<Neighbor>* topk_now) const override;
  void set_query(int64_t qi) { cur_query_ = qi; }
  void clear() { feats.clear(); topk.clear(); query.clear(); }
  int k = 10;
  mutable std::vector<float> feats;   // n x kDarthFeatures
  mutable std::vector<int64_t> topk;  // n x k internal ids (-1 pad)
  mutable std::vector<int64_t> query; // n
 private:
  uint64_t interval_;
  int64_t cur_query_ = -1;
};

// ------------------------------------------------------------ D-Cache
struct DCacheConfig {
  int k = 10, L = 100, beam_width = 4;
  int pivots = 4;                 // most recent cached queries used as pivots
  size_t budget_bytes = 120000;   // cache size (8 bytes per cached distance)
  double max_norm = 1.001;
};

class DCacheEngine : public Engine {
 public:
  DCacheEngine(const Executor& ex, const DCacheConfig& cfg);
  void query(int64_t session, const float* q, QueryOut* out) override;
  void reset() override { entries_.clear(); total_ = 0; }
  size_t state_bytes() const override { return total_ * 8 + entries_.size() * ex_.dim() * 4; }
  const Executor& executor() const override { return ex_; }

 private:
  struct Entry {
    std::vector<float> anchor;
    std::vector<id_t> ids;
    std::vector<float> e;  // float32 executor distances
  };
  const Executor& ex_;
  DCacheConfig cfg_;
  Envelope env_;
  std::deque<Entry> entries_;  // newest first; global across sessions
  size_t total_ = 0;
  Bounds bounds_;
  VisitRecord vr_;
  IdMap dedupe_{1024};
};

// -------------------------------------------------------- Snake Table
struct SnakeConfig {
  int k = 10, L = 100, beam_width = 4;
  int slots = 4;  // pivot slots per row (most recent queries)
  double max_norm = 1.001;
};

class SnakeEngine : public Engine {
 public:
  SnakeEngine(const Executor& ex, const SnakeConfig& cfg);
  void query(int64_t session, const float* q, QueryOut* out) override;
  void end_session(int64_t session) override;
  void reset() override { tables_.clear(); }
  size_t state_bytes() const override;
  const Executor& executor() const override { return ex_; }

 private:
  struct Table {
    std::vector<float> e;          // n x slots
    std::vector<uint32_t> stamp;   // n x slots (query stamp, 0 = empty)
    std::vector<float> qvec;       // slots x d
    std::vector<uint32_t> qstamp;  // slots
    uint32_t t = 1;                // next query stamp
  };
  class SnakeBounds;
  const Executor& ex_;
  SnakeConfig cfg_;
  Envelope env_;
  std::unordered_map<int64_t, Table> tables_;
  VisitRecord vr_;
};

// ------------------------------------------------------------ QVCache
// Follows the QVCache paper (Göçer et al., arXiv:2602.02057): backend top-k
// vectors are cached in n_mini FreshVamana mini-indexes, ordered by
// temperature (most recently used first) and evicted whole (LRU). A lookup
// is a hit if the k-th cached distance is <= (1 + D) * theta[region], where
// the region is the query's PCA bucket key and theta is learned from backend
// k-th distances: theta <- (1 - alpha) * theta + alpha * d_backend[k], set to
// d_backend[k] on a region's first query. ADAPTIVE search is EAGER (stop at
// the first mini-index that passes) while the hit ratio exceeds a threshold,
// EXHAUSTIVE (rerank the union of all mini-indexes) otherwise.
enum class QVStrategy : int { Adaptive = 0, Eager = 1, Exhaustive = 2 };

struct QVCacheConfig {
  int k = 10, L = 100, beam_width = 4;  // backend search
  size_t n_mini = 4;                    // paper default: 4 x 25,000 vectors
  size_t mini_capacity = 25000;
  // Mini-index graph parameters, as in QVCache's public artifact
  // (qvcache/qvcache-vldb26: R=64, memory_L=128, alpha=1.2).
  uint32_t mini_R = 64, mini_L_build = 128, mini_L_search = 128;
  float mini_alpha = 1.2f;
  float deviation = 0.075f;      // D (paper: 0.075, 0.25 for SIFT/SpaceV)
  float alpha = 0.9f;            // threshold adaptivity (paper: 0.9)
  QVStrategy strategy = QVStrategy::Adaptive;
  float eager_hit_ratio = 0.9f;  // ADAPTIVE switch point (paper example: 0.9)
  // Cache insertion runs after the answer is returned (asynchronously in a
  // deployment); its time is excluded from query latency.
  bool exclude_insert_time = true;
  std::string region_model;      // PCA + bucket boundaries (fit_qvcache.py)
};

class VamanaGraph;

class QVCacheEngine : public Engine {
 public:
  QVCacheEngine(const Executor& ex, const QVCacheConfig& cfg);
  ~QVCacheEngine() override;
  void query(int64_t session, const float* q, QueryOut* out) override;
  void reset() override;
  size_t state_bytes() const override;
  const Executor& executor() const override { return ex_; }
  uint64_t region_of(const float* q) const;
  // theta of q's region (NaN if the region has not been seen).
  float theta_of(const float* q) const;
  size_t cached_vectors() const;
  size_t mini_size(size_t m) const;
  std::vector<size_t> order() const;  // mini-index ids, hottest first
  uint64_t hits() const { return hits_; }
  uint64_t lookups() const { return lookups_; }

 private:
  struct Mini;
  void touch(size_t m);
  void insert_results(const std::vector<Neighbor>& topk, const float* vecs);
  const Executor& ex_;
  QVCacheConfig cfg_;
  int pdim_ = 0, nbuckets_ = 0;
  std::vector<float> mean_, comps_, bounds_;  // comps: pdim x d; bounds: pdim x (nbuckets-1)
  std::unordered_map<uint64_t, float> theta_;
  std::vector<std::unique_ptr<Mini>> minis_;
  std::list<size_t> order_;  // front = hottest
  uint64_t hits_ = 0, lookups_ = 0;
};

}  // namespace trace
