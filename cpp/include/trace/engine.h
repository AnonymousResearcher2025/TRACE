// Query engines. Every method (TRACE, its ablations, and the baselines) is an
// Engine, so the replay harness measures all of them the same way.
#pragma once

#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

#include "trace/executor.h"
#include "trace/fp.h"
#include "trace/idmap.h"

namespace trace {

struct QueryOut {
  std::vector<id_t> labels;  // external ids, best first
  std::vector<float> s;      // executor squared distances
  Path path = Path::Cold;
  SearchStats st;
  uint32_t n_seeds = 0;  // seeds given to the executor (catapults, TRACE seeds)
  uint64_t wall_ns = 0, cpu_ns = 0;
  // execution record (Sec. III-D)
  uint32_t R = 0;        // |R|, retained nodes with transferred bounds
  uint32_t S = 0;        // straddlers counted (capped at s_max + 1)
  float min_delta = -1;  // min_i delta_i (-1: no anchors)
  float sigma = 0;       // accumulated displacement since last traversal
  uint32_t ell = 0;      // resolved chain length since last traversal
  float mu = std::numeric_limits<float>::quiet_NaN();  // separation margin
  // audits (computed outside the timed region, never counted)
  uint32_t audit_checked = 0, audit_viol = 0;
  int8_t resolve_exact = -1;  // 1/0 on resolution path when audited
  int8_t prune_equiv = -1;    // 1/0 on seeded path when verified
  // The same query with the seeded search rerun from the same seeds without
  // pruning (verify_prune mode, seeded path only; 0 otherwise).
  uint64_t noprune_n_dist = 0, noprune_wall_ns = 0;
  std::vector<id_t> retained;  // labels of R (audit mode)
  void reset() { *this = QueryOut(); }
};

class Engine {
 public:
  virtual ~Engine() = default;
  virtual void query(int64_t session, const float* q, QueryOut* out) = 0;
  virtual void end_session(int64_t /*session*/) {}
  virtual void reset() {}
  virtual size_t state_bytes() const { return 0; }
  virtual const Executor& executor() const = 0;
};

class Catapult;    // baselines.h

// ------------------------------------------------------------- residues
struct Residue {
  std::vector<float> anchor;      // float32 anchor (the query)
  std::vector<id_t> ids;          // retained nodes (pinned answers included)
  std::vector<uint16_t> d16;      // float16 e(anchor, x)
  std::vector<id_t> ans;          // pinned answer ids
  std::vector<float> ans_vecs;    // |ans| x d, kept only while newest
  bool thin = false;
  bool compacted = false;
  uint64_t epoch = 0;
  double utility = 0.0;
  size_t bytes() const {
    return anchor.size() * 4 + ids.size() * 6 + ans.size() * 4 +
           ans_vecs.size() * 4;
  }
};

// Transferred intervals for the retained set R (one pass, no vector reads).
class Bounds : public BoundProvider {
 public:
  void clear() {
    map.clear();
    ids.clear();
    lb.clear();
    ub.clear();
    lbS_.clear();
    lb_arg.clear();
    ub_arg.clear();
  }
  bool lbS(id_t id, float* out) const override {
    const uint32_t slot = map.find(id);
    if (slot == kInvalidId) return false;
    *out = lbS_[slot];
    return true;
  }
  void on_pruned(id_t id) const override {
    const uint32_t slot = map.find(id);
    if (slot != kInvalidId && credit) (*credit)[lb_arg[slot]] += 1.0;
  }
  size_t size() const { return ids.size(); }
  float lbS_at(size_t slot) const { return lbS_[slot]; }

  IdMap map{1024};
  std::vector<id_t> ids;
  std::vector<float> lb, ub, lbS_;
  std::vector<uint8_t> lb_arg, ub_arg;  // anchor supplying the tightest bound
  mutable std::vector<double>* credit = nullptr;
};

// Builds intervals from residues (newest first) and displacements delta_i.
void transfer_bounds(const std::vector<const Residue*>& ring,
                     const std::vector<float>& deltas, const Envelope& env,
                     Bounds* out);

// ------------------------------------------------------------ TRACE
struct TraceConfig {
  bool enabled = true;  // false: cold search only (+ catapult / early stop)
  int k = 10;
  int L = 100;          // candidate-list capacity L_s
  int beam_width = 4;   // disk tier
  int m = 4;            // ring size
  float tau = 0.6f;     // jump threshold, in units of scale_k estimate
  int w = 32;           // scale-estimator window and utility horizon
  int s_max = 32;       // straddler cap for resolution
  float gamma_s = std::numeric_limits<float>::infinity();  // staleness budget
  int L_max = std::numeric_limits<int>::max();             // chain cap
  int B_r = 2300;       // entries kept per older residue
  enum class Retention { Recency, Hybrid } retention = Retention::Recency;
  enum class Seeds { Bound, PrevAnswer } seeds = Seeds::Bound;
  bool transfer = true;   // false with Seeds::PrevAnswer = warm-seed baseline
  bool prune = true;
  bool resolve = true;
  bool jump = true;
  bool share_ring = false;  // all sessions share one ring (Table VII)
  double max_norm = 1.001;  // bound on vector norms (envelope)
  bool check_norms = true;
  // diagnostics (outside the timed region)
  bool audit = false;         // interval audit + Theorem 1 check + export R
  bool verify_prune = false;  // Theorem 2 check: rerun without pruning
};

class TraceEngine : public Engine {
 public:
  TraceEngine(const Executor& ex, const TraceConfig& cfg,
              Catapult* catapult = nullptr, const EarlyStop* es = nullptr);
  void query(int64_t session, const float* q, QueryOut* out) override;
  void end_session(int64_t session) override;
  void reset() override { sessions_.clear(); }
  size_t state_bytes() const override;
  const Executor& executor() const override { return ex_; }
  const TraceConfig& config() const { return cfg_; }

 private:
  struct Session {
    std::deque<std::unique_ptr<Residue>> ring;  // newest first
    std::deque<float> scale_hist;               // last w k-th distances
    float sigma = 0.f;
    uint32_t ell = 0;
    std::vector<id_t> prev_answer;  // internal ids
    uint64_t epoch = 0;
  };

  Session& session(int64_t sid);
  bool scale_ready(const Session& S) const { return !S.scale_hist.empty(); }
  float scale(const Session& S) const;
  void record_scale(Session& S, const std::vector<Neighbor>& topk);
  std::unique_ptr<Residue> make_full_residue(const float* q,
                                             const VisitRecord& vr,
                                             const SearchResult& res) const;
  void compact(Residue* r) const;
  void push_residue(Session& S, std::unique_ptr<Residue> r);
  void finish(Session& S, const float* q, const std::vector<Neighbor>& topk,
              QueryOut* out) const;

  const Executor& ex_;
  TraceConfig cfg_;
  Envelope env_;
  Catapult* catapult_;
  const EarlyStop* es_;
  std::unordered_map<int64_t, Session> sessions_;
  // per-query scratch
  Bounds bounds_;
  VisitRecord vr_;
  mutable IdMap dedupe_{1024};
  std::vector<double> credit_;
};

}  // namespace trace
