#include "trace/baselines.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>

#include "trace/vamana.h"

namespace trace {

namespace {
// Shared bound construction for float32-storage caches (D-Cache).
void bounds_fp32(const std::vector<const float*>& e_lists,
                 const std::vector<const std::vector<id_t>*>& id_lists,
                 const std::vector<size_t>& sizes, const std::vector<float>& deltas,
                 const Envelope& env, Bounds* B) {
  B->clear();
  size_t total = 0;
  for (size_t s : sizes) total += s;
  B->map.reserve(total);
  for (size_t i = 0; i < id_lists.size(); ++i) {
    const auto& ids = *id_lists[i];
    for (size_t j = 0; j < sizes[i]; ++j) {
      const float lbi = lb_from_anchor(e_lists[i][j], deltas[i], env.eps_plus);
      const float ubi = ub_from_anchor(e_lists[i][j], deltas[i], env.eps_plus);
      auto ins = B->map.insert(ids[j], static_cast<uint32_t>(B->ids.size()));
      if (ins.second) {
        B->ids.push_back(ids[j]);
        B->lb.push_back(lbi);
        B->ub.push_back(ubi);
        B->lb_arg.push_back(static_cast<uint8_t>(i));
        B->ub_arg.push_back(static_cast<uint8_t>(i));
      } else {
        B->lb[ins.first] = std::max(B->lb[ins.first], lbi);
        B->ub[ins.first] = std::min(B->ub[ins.first], ubi);
      }
    }
  }
  B->lbS_.resize(B->ids.size());
  for (size_t s = 0; s < B->ids.size(); ++s) B->lbS_[s] = lbS_from_lb(B->lb[s], env);
}

void fill_out(const Executor& ex, const std::vector<Neighbor>& topk, QueryOut* out) {
  out->labels.clear();
  out->s.clear();
  for (const Neighbor& nb : topk) {
    out->labels.push_back(ex.label(nb.id));
    out->s.push_back(nb.s);
  }
}
}  // namespace

// ============================================================ CatapultDB
Catapult::Catapult(size_t dim, int n_bits, int bucket_cap, uint64_t seed)
    : dim_(dim), bits_(n_bits), cap_(static_cast<size_t>(bucket_cap)) {
  TRACE_CHECK(n_bits >= 1 && n_bits <= 20, "n_bits must be in [1, 20]");
  TRACE_CHECK(bucket_cap >= 1, "bucket_cap must be >= 1");
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  planes_.resize(static_cast<size_t>(bits_) * dim_);
  for (float& v : planes_) v = nd(rng);
  buckets_.assign(size_t{1} << bits_, {});
}

uint32_t Catapult::hash(const float* q, SearchStats* st) const {
  uint32_t h = 0;
  for (int b = 0; b < bits_; ++b) {
    const float* pl = planes_.data() + static_cast<size_t>(b) * dim_;
    float dot = 0.f;
    for (size_t j = 0; j < dim_; ++j) dot += pl[j] * q[j];
    if (dot >= 0.f) h |= (1u << b);
  }
  if (st) st->n_aux += static_cast<uint64_t>(bits_);
  return h;
}

std::vector<id_t> Catapult::seeds(uint32_t h) const {
  std::lock_guard<std::mutex> g(mu_);
  return buckets_[h];
}

void Catapult::update(uint32_t h, id_t best) {
  std::lock_guard<std::mutex> g(mu_);
  auto& b = buckets_[h];
  auto it = std::find(b.begin(), b.end(), best);
  if (it != b.end()) b.erase(it);
  b.insert(b.begin(), best);
  if (b.size() > cap_) b.pop_back();
}

void Catapult::reset() {
  std::lock_guard<std::mutex> g(mu_);
  for (auto& b : buckets_) b.clear();
}

size_t Catapult::bytes() const {
  std::lock_guard<std::mutex> g(mu_);
  size_t n = planes_.size() * 4;
  for (const auto& b : buckets_) n += b.size() * 4;
  return n;
}

// ================================================================ DARTH
void GbdtModel::load(const std::string& path) {
  std::ifstream in(path);
  TRACE_CHECK(in.good(), "cannot open GBDT model " + path);
  std::string tag;
  int version = 0;
  in >> tag >> version;
  TRACE_CHECK(tag == "GBDT" && version == 1, "bad GBDT model header");
  in >> tag >> base;
  TRACE_CHECK(tag == "base", "bad GBDT model (base)");
  size_t nt = 0;
  in >> tag >> nt;
  TRACE_CHECK(tag == "trees", "bad GBDT model (trees)");
  trees.resize(nt);
  for (auto& t : trees) {
    size_t nn = 0;
    in >> tag >> nn;
    TRACE_CHECK(tag == "tree" && nn > 0, "bad GBDT tree header");
    t.feat.resize(nn);
    t.thr.resize(nn);
    t.left.resize(nn);
    t.right.resize(nn);
    t.value.resize(nn);
    for (size_t i = 0; i < nn; ++i)
      in >> t.feat[i] >> t.thr[i] >> t.left[i] >> t.right[i] >> t.value[i];
    TRACE_CHECK(in.good(), "truncated GBDT tree");
    for (size_t i = 0; i < nn; ++i)
      if (t.feat[i] >= 0)
        TRACE_CHECK(t.feat[i] < kDarthFeatures && t.left[i] > 0 &&
                        t.right[i] > 0 && static_cast<size_t>(t.left[i]) < nn &&
                        static_cast<size_t>(t.right[i]) < nn,
                    "bad GBDT node");
  }
}

float GbdtModel::predict(const float* x) const {
  float s = base;
  for (const auto& t : trees) {
    int n = 0;
    while (t.feat[static_cast<size_t>(n)] >= 0) {
      const size_t i = static_cast<size_t>(n);
      n = (x[t.feat[i]] <= t.thr[i]) ? t.left[i] : t.right[i];
    }
    s += t.value[static_cast<size_t>(n)];
  }
  return s;
}

void darth_features(const ProgressFeatures& f, float* x) {
  const float tiny = 1e-12f;
  x[0] = static_cast<float>(f.n_dist);
  x[1] = static_cast<float>(f.n_hops);
  x[2] = f.s_first;
  x[3] = f.s_kth;
  x[4] = std::isfinite(f.s_bound) ? f.s_bound : -1.f;
  x[5] = f.s_entry;
  x[6] = f.s_kth / std::max(f.s_first, tiny);
  x[7] = f.s_kth / std::max(f.s_entry, tiny);
  x[8] = static_cast<float>(f.n_admit) / static_cast<float>(std::max<uint64_t>(f.n_eval, 1));
}

DarthStop::DarthStop(const std::string& model_path, double target_recall,
                     uint64_t interval)
    : target_(target_recall), interval_(interval) {
  TRACE_CHECK(interval >= 1, "interval must be >= 1");
  model_.load(model_path);
}

bool DarthStop::should_stop(const ProgressFeatures& f,
                            const std::vector<Neighbor>* /*topk_now*/) const {
  float x[kDarthFeatures];
  darth_features(f, x);
  return model_.predict(x) >= static_cast<float>(target_);
}

bool LogStop::should_stop(const ProgressFeatures& f,
                          const std::vector<Neighbor>* topk_now) const {
  float x[kDarthFeatures];
  darth_features(f, x);
  feats.insert(feats.end(), x, x + kDarthFeatures);
  for (int i = 0; i < k; ++i) {
    const bool have = topk_now && static_cast<size_t>(i) < topk_now->size();
    topk.push_back(have ? static_cast<int64_t>((*topk_now)[static_cast<size_t>(i)].id) : -1);
  }
  query.push_back(cur_query_);
  return false;
}

// ============================================================== D-Cache
DCacheEngine::DCacheEngine(const Executor& ex, const DCacheConfig& cfg)
    : ex_(ex), cfg_(cfg) {
  TRACE_CHECK(cfg_.pivots >= 1 && cfg_.pivots <= 255, "pivots must be in [1,255]");
  env_ = Envelope::make(ex.dim(), cfg_.max_norm, /*fp16_storage=*/false);
}

void DCacheEngine::query(int64_t /*session*/, const float* q, QueryOut* out) {
  out->reset();
  const uint64_t t0 = now_ns(), c0 = thread_cpu_ns();
  SearchStats& st = out->st;
  const size_t np = std::min(static_cast<size_t>(cfg_.pivots), entries_.size());
  std::vector<float> deltas(np);
  std::vector<const float*> el;
  std::vector<const std::vector<id_t>*> il;
  std::vector<size_t> sz;
  for (size_t i = 0; i < np; ++i) {
    deltas[i] = e_from_s(ex_.dist(entries_[i].anchor.data(), q));
    st.n_dist++;
    el.push_back(entries_[i].e.data());
    il.push_back(&entries_[i].ids);
    sz.push_back(entries_[i].ids.size());
  }
  bounds_fp32(el, il, sz, deltas, env_, &bounds_);
  out->R = static_cast<uint32_t>(bounds_.size());
  SearchParams p;
  p.k = cfg_.k;
  p.L = cfg_.L;
  p.beam_width = cfg_.beam_width;
  SearchResult res;
  vr_.clear();
  ex_.search(q, {}, p, np > 0 ? &bounds_ : nullptr, nullptr, &res, &vr_, &st);
  out->path = Path::Cold;

  // Cache this query's distances (global, newest first).
  Entry en;
  en.anchor.assign(q, q + ex_.dim());
  IdMap& seen = dedupe_;
  seen.clear();
  seen.reserve(vr_.ids.size());
  for (size_t i = 0; i < vr_.ids.size(); ++i) {
    if (!seen.insert(vr_.ids[i], 0).second) continue;
    en.ids.push_back(vr_.ids[i]);
    en.e.push_back(e_from_s(vr_.s[i]));
  }
  const size_t budget = cfg_.budget_bytes / 8;
  if (en.ids.size() > budget) {  // keep the closest entries of the newest
    std::vector<uint32_t> idx(en.ids.size());
    for (uint32_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::nth_element(idx.begin(), idx.begin() + budget, idx.end(),
                     [&](uint32_t a, uint32_t b) { return en.e[a] < en.e[b]; });
    idx.resize(budget);
    Entry t;
    t.anchor = en.anchor;
    for (uint32_t i : idx) {
      t.ids.push_back(en.ids[i]);
      t.e.push_back(en.e[i]);
    }
    en = std::move(t);
  }
  total_ += en.ids.size();
  entries_.push_front(std::move(en));
  while (entries_.size() > 1 && total_ > budget) {
    total_ -= entries_.back().ids.size();
    entries_.pop_back();
  }
  fill_out(ex_, res.topk, out);
  out->wall_ns = now_ns() - t0;
  out->cpu_ns = thread_cpu_ns() - c0;
}

// =========================================================== Snake Table
class SnakeEngine::SnakeBounds : public BoundProvider {
 public:
  SnakeBounds(const Table& T, const std::vector<float>& deltas,
              const std::vector<char>& valid, int slots, const Envelope& env)
      : T_(T), deltas_(deltas), valid_(valid), slots_(slots), env_(env) {}
  bool lbS(id_t id, float* out) const override {
    const size_t base = static_cast<size_t>(id) * static_cast<size_t>(slots_);
    float best = 0.f;
    bool any = false;
    for (int c = 0; c < slots_; ++c) {
      if (!valid_[static_cast<size_t>(c)]) continue;
      const uint32_t s = T_.stamp[base + static_cast<size_t>(c)];
      if (s == 0 || s != T_.qstamp[static_cast<size_t>(c)]) continue;
      const float lb = lb_from_anchor(T_.e[base + static_cast<size_t>(c)],
                                      deltas_[static_cast<size_t>(c)], env_.eps_plus);
      best = std::max(best, lb);
      any = true;
    }
    if (!any) return false;
    *out = lbS_from_lb(best, env_);
    return true;
  }

 private:
  const Table& T_;
  const std::vector<float>& deltas_;
  const std::vector<char>& valid_;
  int slots_;
  const Envelope& env_;
};

SnakeEngine::SnakeEngine(const Executor& ex, const SnakeConfig& cfg)
    : ex_(ex), cfg_(cfg) {
  TRACE_CHECK(cfg_.slots >= 1, "slots must be >= 1");
  env_ = Envelope::make(ex.dim(), cfg_.max_norm, /*fp16_storage=*/false);
}

void SnakeEngine::end_session(int64_t session) { tables_.erase(session); }

size_t SnakeEngine::state_bytes() const {
  size_t b = 0;
  for (const auto& kv : tables_)
    b += kv.second.e.size() * 4 + kv.second.stamp.size() * 4 + kv.second.qvec.size() * 4;
  return b;
}

void SnakeEngine::query(int64_t session, const float* q, QueryOut* out) {
  out->reset();
  const size_t n = ex_.size(), d = ex_.dim();
  const size_t P = static_cast<size_t>(cfg_.slots);
  auto it = tables_.find(session);
  if (it == tables_.end()) {
    Table T;
    T.e.assign(n * P, 0.f);
    T.stamp.assign(n * P, 0u);
    T.qvec.assign(P * d, 0.f);
    T.qstamp.assign(P, 0u);
    it = tables_.emplace(session, std::move(T)).first;
  }
  Table& T = it->second;
  const uint64_t t0 = now_ns(), c0 = thread_cpu_ns();
  SearchStats& st = out->st;
  std::vector<float> deltas(P, 0.f);
  std::vector<char> valid(P, 0);
  bool any = false;
  for (size_t c = 0; c < P; ++c) {
    if (T.qstamp[c] == 0) continue;
    deltas[c] = e_from_s(ex_.dist(T.qvec.data() + c * d, q));
    st.n_dist++;
    valid[c] = 1;
    any = true;
  }
  SnakeBounds sb(T, deltas, valid, cfg_.slots, env_);
  SearchParams p;
  p.k = cfg_.k;
  p.L = cfg_.L;
  p.beam_width = cfg_.beam_width;
  SearchResult res;
  vr_.clear();
  ex_.search(q, {}, p, any ? &sb : nullptr, nullptr, &res, &vr_, &st);
  out->path = Path::Cold;
  // Round-robin replacement: query t overwrites pivot slot t mod slots.
  const uint32_t t = T.t++;
  const size_t c = static_cast<size_t>(t % P);
  for (size_t i = 0; i < vr_.ids.size(); ++i) {
    const size_t cell = static_cast<size_t>(vr_.ids[i]) * P + c;
    T.e[cell] = e_from_s(vr_.s[i]);
    T.stamp[cell] = t;
  }
  std::memcpy(T.qvec.data() + c * d, q, d * sizeof(float));
  T.qstamp[c] = t;
  fill_out(ex_, res.topk, out);
  out->wall_ns = now_ns() - t0;
  out->cpu_ns = thread_cpu_ns() - c0;
}

// =============================================================== QVCache
struct QVCacheEngine::Mini {
  Mini(size_t dim, size_t cap, const VamanaParams& p) : g(dim, cap, p) {}
  VamanaGraph g;
  std::vector<id_t> ids;              // local id -> executor internal id
  std::unordered_map<id_t, id_t> at;  // executor internal id -> local id
  void clear() {
    g.clear();
    ids.clear();
    at.clear();
  }
};

QVCacheEngine::QVCacheEngine(const Executor& ex, const QVCacheConfig& cfg)
    : ex_(ex), cfg_(cfg) {
  std::ifstream in(cfg_.region_model, std::ios::binary);
  TRACE_CHECK(in.good(), "cannot open QVCache region model " + cfg_.region_model);
  char magic[4];
  in.read(magic, 4);
  TRACE_CHECK(std::memcmp(magic, "QVRM", 4) == 0, "bad region model magic");
  int32_t d = 0, pdim = 0, nb = 0;
  in.read(reinterpret_cast<char*>(&d), 4);
  in.read(reinterpret_cast<char*>(&pdim), 4);
  in.read(reinterpret_cast<char*>(&nb), 4);
  TRACE_CHECK(static_cast<size_t>(d) == ex.dim() && pdim >= 1 && nb >= 2,
              "region model shape mismatch");
  double keyspace = std::pow(static_cast<double>(nb), pdim);
  TRACE_CHECK(keyspace < 1.8e19, "region key space exceeds 64 bits");
  pdim_ = pdim;
  nbuckets_ = nb;
  mean_.resize(static_cast<size_t>(d));
  comps_.resize(static_cast<size_t>(pdim) * d);
  bounds_.resize(static_cast<size_t>(pdim) * (nb - 1));
  in.read(reinterpret_cast<char*>(mean_.data()), mean_.size() * 4);
  in.read(reinterpret_cast<char*>(comps_.data()), comps_.size() * 4);
  in.read(reinterpret_cast<char*>(bounds_.data()), bounds_.size() * 4);
  TRACE_CHECK(in.good(), "truncated region model");
  TRACE_CHECK(cfg_.n_mini >= 1, "n_mini must be >= 1");
  TRACE_CHECK(cfg_.mini_capacity >= static_cast<size_t>(cfg_.k), "mini_capacity < k");
  TRACE_CHECK(cfg_.mini_L_search >= static_cast<uint32_t>(cfg_.k), "mini_L_search < k");
  VamanaParams vp;
  vp.R = cfg_.mini_R;
  vp.L_build = cfg_.mini_L_build;
  vp.alpha = cfg_.mini_alpha;
  vp.maxc = std::max<uint32_t>(vp.maxc, vp.R);
  for (size_t m = 0; m < cfg_.n_mini; ++m) {
    minis_.push_back(std::make_unique<Mini>(ex.dim(), cfg_.mini_capacity, vp));
    order_.push_back(m);
  }
}

QVCacheEngine::~QVCacheEngine() = default;

void QVCacheEngine::reset() {
  theta_.clear();
  for (auto& m : minis_) m->clear();
  order_.clear();
  for (size_t m = 0; m < minis_.size(); ++m) order_.push_back(m);
  hits_ = lookups_ = 0;
}

size_t QVCacheEngine::state_bytes() const {
  size_t b = theta_.size() * (sizeof(uint64_t) + sizeof(float));
  for (const auto& m : minis_) b += m->g.bytes() + m->ids.size() * sizeof(id_t);
  return b;
}

size_t QVCacheEngine::cached_vectors() const {
  size_t n = 0;
  for (const auto& m : minis_) n += m->g.size();
  return n;
}

size_t QVCacheEngine::mini_size(size_t m) const {
  TRACE_CHECK(m < minis_.size(), "mini-index out of range");
  return minis_[m]->g.size();
}

std::vector<size_t> QVCacheEngine::order() const { return {order_.begin(), order_.end()}; }

uint64_t QVCacheEngine::region_of(const float* q) const {
  const size_t d = ex_.dim();
  uint64_t key = 0;
  for (int p = 0; p < pdim_; ++p) {
    const float* c = comps_.data() + static_cast<size_t>(p) * d;
    float z = 0.f;
    for (size_t j = 0; j < d; ++j) z += c[j] * (q[j] - mean_[j]);
    const float* b = bounds_.data() + static_cast<size_t>(p) * (nbuckets_ - 1);
    const uint64_t digit =
        static_cast<uint64_t>(std::upper_bound(b, b + (nbuckets_ - 1), z) - b);
    key = key * static_cast<uint64_t>(nbuckets_) + digit;
  }
  return key;
}

float QVCacheEngine::theta_of(const float* q) const {
  auto it = theta_.find(region_of(q));
  return it == theta_.end() ? std::numeric_limits<float>::quiet_NaN() : it->second;
}

void QVCacheEngine::touch(size_t m) {
  auto it = std::find(order_.begin(), order_.end(), m);
  TRACE_CHECK(it != order_.end(), "mini-index missing from order");
  order_.splice(order_.begin(), order_, it);
}

// Inserts into the hottest mini-index with room for all vectors; if none has
// room, the coldest is evicted (cleared) and reused. The target becomes the
// hottest. A vector already present in the target is not inserted twice.
void QVCacheEngine::insert_results(const std::vector<Neighbor>& topk, const float* vecs) {
  const size_t need = topk.size();
  if (need == 0) return;
  size_t target = order_.back();
  bool found = false;
  for (size_t m : order_) {
    if (minis_[m]->g.size() + need <= cfg_.mini_capacity) {
      target = m;
      found = true;
      break;
    }
  }
  Mini& M = *minis_[target];
  if (!found) M.clear();
  touch(target);
  const size_t d = ex_.dim();
  for (size_t i = 0; i < need; ++i) {
    const id_t id = topk[i].id;
    if (M.at.count(id)) continue;
    const id_t local = M.g.insert(vecs + i * d);
    M.ids.push_back(id);
    M.at.emplace(id, local);
  }
}

void QVCacheEngine::query(int64_t /*session*/, const float* q, QueryOut* out) {
  out->reset();
  const uint64_t t0 = now_ns(), c0 = thread_cpu_ns();
  uint64_t excl_wall = 0, excl_cpu = 0;
  SearchStats& st = out->st;
  const size_t k = static_cast<size_t>(cfg_.k);
  const uint64_t key = region_of(q);
  st.n_aux += static_cast<uint64_t>(pdim_);
  std::vector<Neighbor> topk;
  bool hit = false;
  auto th = theta_.find(key);
  if (th != theta_.end()) {
    const float limit = (1.f + cfg_.deviation) * th->second;
    bool eager = cfg_.strategy == QVStrategy::Eager;
    if (cfg_.strategy == QVStrategy::Adaptive)
      eager = lookups_ > 0 &&
              static_cast<double>(hits_) / static_cast<double>(lookups_) > cfg_.eager_hit_ratio;
    std::vector<Neighbor> cand;
    std::vector<std::pair<Neighbor, size_t>> pool;  // (executor id, s), mini-index
    const std::vector<size_t> ord(order_.begin(), order_.end());
    for (size_t m : ord) {
      const Mini& M = *minis_[m];
      if (M.g.size() < k) continue;
      SearchStats cs;
      M.g.search(q, k, cfg_.mini_L_search, &cand, &cs);
      st.n_dist += cs.n_dist;  // distances to cached copies of corpus vectors
      if (eager) {
        if (cand.size() >= k && e_from_s(cand[k - 1].s) <= limit) {
          hit = true;
          for (const Neighbor& nb : cand) topk.push_back({nb.s, M.ids[nb.id]});
          touch(m);
          break;
        }
      } else {
        for (const Neighbor& nb : cand) pool.push_back({{nb.s, M.ids[nb.id]}, m});
      }
    }
    if (!eager && pool.size() >= k) {
      std::sort(pool.begin(), pool.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
      std::vector<Neighbor> uni;
      std::vector<size_t> used;
      for (const auto& pe : pool) {
        bool dup = false;  // the same vector cached in several mini-indexes
        for (const Neighbor& u : uni) dup = dup || u.id == pe.first.id;
        if (dup) continue;
        uni.push_back(pe.first);
        if (std::find(used.begin(), used.end(), pe.second) == used.end()) used.push_back(pe.second);
        if (uni.size() == k) break;
      }
      if (uni.size() == k && e_from_s(uni[k - 1].s) <= limit) {
        hit = true;
        topk = uni;
        // contributing mini-indexes become hottest, keeping their relative order
        std::vector<size_t> by_heat;
        for (size_t m : ord)
          if (std::find(used.begin(), used.end(), m) != used.end()) by_heat.push_back(m);
        for (auto it = by_heat.rbegin(); it != by_heat.rend(); ++it) touch(*it);
      }
    }
  }
  ++lookups_;
  if (hit) {
    ++hits_;
    out->path = Path::CacheHit;
  } else {
    SearchParams p;
    p.k = cfg_.k;
    p.L = cfg_.L;
    p.beam_width = cfg_.beam_width;
    p.collect_visited = false;
    p.want_vectors = true;
    SearchResult res;
    ex_.search(q, {}, p, nullptr, nullptr, &res, nullptr, &st);
    out->path = Path::Cold;
    topk = res.topk;
    if (topk.size() >= k) {
      const float kth = e_from_s(topk[k - 1].s);
      if (th == theta_.end()) theta_[key] = kth;
      else th->second = (1.f - cfg_.alpha) * th->second + cfg_.alpha * kth;
    }
    const uint64_t tw = now_ns(), tc = thread_cpu_ns();
    insert_results(topk, res.vectors.data());
    if (cfg_.exclude_insert_time) {
      excl_wall += now_ns() - tw;
      excl_cpu += thread_cpu_ns() - tc;
    }
  }
  fill_out(ex_, topk, out);
  out->wall_ns = now_ns() - t0 - excl_wall;
  out->cpu_ns = thread_cpu_ns() - c0 - excl_cpu;
}

}  // namespace trace
