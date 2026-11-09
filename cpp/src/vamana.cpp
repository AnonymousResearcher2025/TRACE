#include "trace/vamana.h"

#include <algorithm>
#include <cfloat>
#include <cstring>
#include <fstream>

namespace trace {

// ------------------------------------------------------- CandidateList
void CandidateList::insert(const Neighbor& nb) {
  if (size_ == cap_ && data_[size_ - 1].nb < nb) return;
  size_t lo = 0, hi = size_;
  while (lo < hi) {
    const size_t mid = (lo + hi) >> 1;
    if (nb < data_[mid].nb) {
      hi = mid;
    } else if (data_[mid].nb.id == nb.id) {
      return;  // already present
    } else {
      lo = mid + 1;
    }
  }
  if (lo < cap_)
    std::memmove(&data_[lo + 1], &data_[lo], (size_ - lo) * sizeof(Entry));
  data_[lo] = {nb, false};
  if (size_ < cap_) ++size_;
  if (lo < cur_) cur_ = lo;
}

Neighbor CandidateList::closest_unexpanded() {
  data_[cur_].expanded = true;
  const size_t pre = cur_;
  while (cur_ < size_ && data_[cur_].expanded) ++cur_;
  return data_[pre].nb;
}

// --------------------------------------------------------- VamanaGraph
VamanaGraph::VamanaGraph(size_t dim, size_t capacity, const VamanaParams& p)
    : dim_(dim), cap_(capacity), p_(p) {
  TRACE_CHECK(dim >= 1 && capacity >= 1, "VamanaGraph: empty shape");
  TRACE_CHECK(p.R >= 1 && p.L_build >= 1 && p.alpha >= 1.f && p.maxc >= p.R,
              "VamanaGraph: invalid parameters");
  data_.resize(cap_ * dim_);
  adj_.resize(cap_);
  space_ = std::make_unique<hnswlib::L2Space>(dim_);
  fdist_ = space_->get_dist_func();
  dparam_ = space_->get_dist_func_param();
  pool_ = std::make_unique<hnswlib::VisitedListPool>(1, static_cast<int>(cap_));
}

size_t VamanaGraph::bytes() const {
  size_t b = n_ * dim_ * sizeof(float);
  for (size_t i = 0; i < n_; ++i) b += adj_[i].size() * sizeof(id_t);
  return b;
}

void VamanaGraph::clear() {
  for (size_t i = 0; i < n_; ++i) adj_[i].clear();
  n_ = 0;
  start_ = 0;
}

void VamanaGraph::greedy(const float* q, size_t L, const std::vector<id_t>& init,
                         CandidateList* cand, const BoundProvider* prune,
                         VisitRecord* visited, std::vector<Neighbor>* expanded,
                         SearchStats* st) const {
  TRACE_CHECK(L >= 1, "greedy: L must be >= 1");
  cand->reset(L);
  if (n_ == 0) return;
  hnswlib::VisitedList* vl = pool_->getFreeVisitedList();
  const hnswlib::vl_type tag = vl->curV;
  hnswlib::vl_type* mass = vl->mass;
  auto eval = [&](id_t id) {
    if (prune != nullptr && cand->full()) {
      float lb;
      if (prune->lbS(id, &lb) && lb > cand->worst().s) {
        prune->on_pruned(id);
        st->n_pruned++;
        return;
      }
    }
    const float s = fdist_(q, vec(id), dparam_);
    st->n_dist++;
    if (visited) visited->push(id, s);
    cand->insert({s, id});
  };
  for (id_t id : init) {
    TRACE_CHECK(id < n_, "greedy: seed out of range");
    if (mass[id] == tag) continue;
    mass[id] = tag;
    eval(id);
  }
  while (cand->has_unexpanded()) {
    const Neighbor nb = cand->closest_unexpanded();
    st->n_hops++;
    if (expanded) expanded->push_back(nb);
    for (id_t j : adj_[nb.id]) {
      if (mass[j] == tag) continue;
      mass[j] = tag;
      eval(j);
    }
  }
  pool_->releaseVisitedList(vl);
}

void VamanaGraph::search(const float* q, size_t k, size_t L, std::vector<Neighbor>* out,
                         SearchStats* st) const {
  TRACE_CHECK(L >= k, "search: L must be >= k");
  out->clear();
  if (n_ == 0) return;
  CandidateList cand;
  greedy(q, L, {start_}, &cand, nullptr, nullptr, nullptr, st);
  for (size_t i = 0; i < cand.size() && i < k; ++i) out->push_back(cand.at(i));
}

void VamanaGraph::prune_list(id_t loc, std::vector<Neighbor>* pool, std::vector<id_t>* out,
                             SearchStats* st) const {
  out->clear();
  if (pool->empty()) return;
  std::sort(pool->begin(), pool->end());
  if (pool->size() > p_.maxc) pool->resize(p_.maxc);
  std::vector<float> occ(pool->size(), 0.f);
  const auto& P = *pool;
  float cur = 1.f;
  while (cur <= p_.alpha && out->size() < p_.R) {
    for (size_t i = 0; out->size() < p_.R && i < P.size(); ++i) {
      if (occ[i] > cur) continue;
      occ[i] = FLT_MAX;
      if (P[i].id != loc) out->push_back(P[i].id);
      for (size_t t = i + 1; t < P.size(); ++t) {
        if (occ[t] > p_.alpha) continue;
        const float djk = fdist_(vec(P[t].id), vec(P[i].id), dparam_);
        if (st) st->n_aux++;
        occ[t] = (djk == 0.f) ? FLT_MAX : std::max(occ[t], P[t].s / djk);
      }
    }
    cur *= 1.2f;
  }
}

id_t VamanaGraph::insert(const float* v, SearchStats* st) {
  TRACE_CHECK(n_ < cap_, "VamanaGraph: capacity exceeded");
  const id_t id = static_cast<id_t>(n_);
  std::memcpy(mvec(id), v, dim_ * sizeof(float));
  adj_[id].clear();
  if (n_ == 0) {
    start_ = id;
    n_ = 1;
    return id;
  }
  // search_for_point_and_prune over the points inserted so far
  SearchStats local;
  CandidateList cand;
  std::vector<Neighbor> pool;
  greedy(v, p_.L_build, {start_}, &cand, nullptr, nullptr, &pool, &local);
  if (st) st->n_aux += local.n_dist;
  std::vector<id_t> pruned;
  prune_list(id, &pool, &pruned, st);
  ++n_;
  adj_[id] = pruned;
  // inter_insert: back edges, re-pruning a list once it exceeds slack * R
  const size_t slack_cap = static_cast<size_t>(p_.slack * static_cast<float>(p_.R));
  for (id_t des : pruned) {
    auto& dp = adj_[des];
    if (std::find(dp.begin(), dp.end(), id) != dp.end()) continue;
    if (dp.size() < slack_cap) {
      dp.push_back(id);
      continue;
    }
    std::vector<Neighbor> cands;
    cands.reserve(dp.size() + 1);
    for (id_t c : dp) cands.push_back({fdist_(vec(des), vec(c), dparam_), c});
    cands.push_back({fdist_(vec(des), vec(id), dparam_), id});
    if (st) st->n_aux += cands.size();
    std::vector<id_t> nout;
    prune_list(des, &cands, &nout, st);
    dp = std::move(nout);
  }
  return id;
}

// DiskANN graph file: u64 file size, u32 max degree, u32 start,
// u64 frozen points, then per node: u32 degree, degree x u32 neighbors.
void VamanaGraph::save_diskann(const std::string& path) const {
  std::ofstream out(path, std::ios::binary);
  TRACE_CHECK(out.good(), "cannot write " + path);
  uint64_t size = 24;
  uint32_t maxdeg = 0;
  for (size_t i = 0; i < n_; ++i) {
    size += 4 + 4 * adj_[i].size();
    maxdeg = std::max<uint32_t>(maxdeg, static_cast<uint32_t>(adj_[i].size()));
  }
  const uint32_t start = static_cast<uint32_t>(start_);
  const uint64_t frozen = 0;
  out.write(reinterpret_cast<const char*>(&size), 8);
  out.write(reinterpret_cast<const char*>(&maxdeg), 4);
  out.write(reinterpret_cast<const char*>(&start), 4);
  out.write(reinterpret_cast<const char*>(&frozen), 8);
  for (size_t i = 0; i < n_; ++i) {
    const uint32_t k = static_cast<uint32_t>(adj_[i].size());
    out.write(reinterpret_cast<const char*>(&k), 4);
    static_assert(sizeof(id_t) == 4, "id_t must be 32-bit");
    out.write(reinterpret_cast<const char*>(adj_[i].data()), 4 * k);
  }
  TRACE_CHECK(out.good(), "write failed " + path);
}

std::unique_ptr<VamanaGraph> VamanaGraph::load_diskann(const std::string& graph_path,
                                                       const std::string& vectors_path,
                                                       const VamanaParams& p) {
  std::ifstream vin(vectors_path, std::ios::binary);
  TRACE_CHECK(vin.good(), "cannot open " + vectors_path);
  int32_t n = 0, d = 0;
  vin.read(reinterpret_cast<char*>(&n), 4);
  vin.read(reinterpret_cast<char*>(&d), 4);
  TRACE_CHECK(vin.good() && n > 0 && d > 0, "bad vector file header " + vectors_path);
  auto g = std::make_unique<VamanaGraph>(static_cast<size_t>(d), static_cast<size_t>(n), p);
  vin.read(reinterpret_cast<char*>(g->data_.data()),
           static_cast<std::streamsize>(g->data_.size() * sizeof(float)));
  TRACE_CHECK(vin.good(), "truncated vector file " + vectors_path);

  std::ifstream gin(graph_path, std::ios::binary);
  TRACE_CHECK(gin.good(), "cannot open " + graph_path);
  uint64_t fsize = 0, frozen = 0;
  uint32_t maxdeg = 0, start = 0;
  gin.read(reinterpret_cast<char*>(&fsize), 8);
  gin.read(reinterpret_cast<char*>(&maxdeg), 4);
  gin.read(reinterpret_cast<char*>(&start), 4);
  gin.read(reinterpret_cast<char*>(&frozen), 8);
  TRACE_CHECK(gin.good(), "bad graph header " + graph_path);
  TRACE_CHECK(frozen == 0, "graphs with frozen points are not supported");
  uint64_t pos = 24;
  size_t node = 0;
  while (pos < fsize) {
    TRACE_CHECK(node < static_cast<size_t>(n), "graph has more nodes than vectors");
    uint32_t k = 0;
    gin.read(reinterpret_cast<char*>(&k), 4);
    TRACE_CHECK(gin.good() && k <= maxdeg, "corrupt graph record");
    auto& a = g->adj_[node];
    a.resize(k);
    gin.read(reinterpret_cast<char*>(a.data()), 4 * static_cast<std::streamsize>(k));
    TRACE_CHECK(gin.good(), "truncated graph " + graph_path);
    for (id_t j : a) TRACE_CHECK(j < static_cast<id_t>(n), "neighbor id out of range");
    pos += 4 + 4 * static_cast<uint64_t>(k);
    ++node;
  }
  TRACE_CHECK(pos == fsize, "graph size mismatch");
  TRACE_CHECK(node == static_cast<size_t>(n), "graph and vector counts differ");
  TRACE_CHECK(start < static_cast<uint32_t>(n), "start node out of range");
  g->n_ = node;
  g->start_ = start;
  return g;
}

// ------------------------------------------------------ VamanaExecutor
VamanaExecutor::VamanaExecutor(const std::string& graph_path, const std::string& vectors_path)
    : g_(VamanaGraph::load_diskann(graph_path, vectors_path)) {}

void VamanaExecutor::fetch_vectors(const id_t* ids, size_t n, float* out,
                                   SearchStats* /*st*/) const {
  const size_t d = g_->dim();
  for (size_t i = 0; i < n; ++i) {
    TRACE_CHECK(ids[i] < g_->size(), "fetch_vectors: id out of range");
    std::memcpy(out + i * d, g_->vec(ids[i]), d * sizeof(float));
  }
}

void VamanaExecutor::search(const float* q, const std::vector<id_t>& seeds,
                            const SearchParams& p, const BoundProvider* prune,
                            const EarlyStop* es, SearchResult* out, VisitRecord* visited,
                            SearchStats* st) const {
  TRACE_CHECK(es == nullptr, "VamanaExecutor does not support early termination");
  TRACE_CHECK(p.k >= 1 && p.L >= p.k, "VamanaExecutor: need L >= k >= 1");
  std::vector<id_t> init;
  if (seeds.empty()) {
    init.push_back(g_->start());
  } else {
    init = seeds;
    if (p.add_entry_to_seeds) init.push_back(g_->start());
  }
  CandidateList cand;
  g_->greedy(q, static_cast<size_t>(p.L), init, &cand, prune,
             p.collect_visited ? visited : nullptr, nullptr, st);
  out->topk.clear();
  const size_t k = std::min(static_cast<size_t>(p.k), cand.size());
  for (size_t i = 0; i < k; ++i) out->topk.push_back(cand.at(i));
  out->vectors.clear();
  if (p.want_vectors) {
    const size_t d = g_->dim();
    out->vectors.resize(k * d);
    for (size_t i = 0; i < k; ++i)
      std::memcpy(out->vectors.data() + i * d, g_->vec(out->topk[i].id), d * sizeof(float));
  }
}

}  // namespace trace
