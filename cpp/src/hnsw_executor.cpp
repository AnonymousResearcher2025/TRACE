#include "trace/hnsw_executor.h"

#include <algorithm>
#include <cstring>

namespace trace {

namespace {
using HPair = std::pair<float, hnswlib::tableint>;
using HCmp = hnswlib::HierarchicalNSW<float>::CompareByFirst;

// std::priority_queue is exactly push_heap/pop_heap over a vector with the
// same comparator; using the primitives directly gives identical behaviour
// and lets the early-stop features read the heap.
inline void heap_push(std::vector<HPair>& h, float d, hnswlib::tableint id) {
  h.emplace_back(d, id);
  std::push_heap(h.begin(), h.end(), HCmp());
}
inline void heap_pop(std::vector<HPair>& h) {
  std::pop_heap(h.begin(), h.end(), HCmp());
  h.pop_back();
}

void current_topk(const std::vector<HPair>& top, int k, std::vector<Neighbor>* out) {
  out->clear();
  for (const auto& e : top) out->push_back({e.first, static_cast<id_t>(e.second)});
  std::sort(out->begin(), out->end());
  if (out->size() > static_cast<size_t>(k)) out->resize(static_cast<size_t>(k));
}

void fill_features(const std::vector<HPair>& top, int k, ProgressFeatures* f) {
  std::vector<float> s(top.size());
  for (size_t i = 0; i < top.size(); ++i) s[i] = top[i].first;
  if (s.empty()) return;
  const size_t kk = std::min<size_t>(static_cast<size_t>(k), s.size()) - 1;
  std::nth_element(s.begin(), s.begin() + kk, s.end());
  f->s_kth = s[kk];
  f->s_first = *std::min_element(s.begin(), s.begin() + kk + 1);
  f->s_bound = top.front().first;
}
}  // namespace

HnswExecutor::HnswExecutor(const std::string& index_path, size_t dim)
    : dim_(dim) {
  space_ = std::make_unique<hnswlib::L2Space>(dim);
  index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(space_.get(),
                                                             index_path);
  fdist_ = space_->get_dist_func();
  dparam_ = space_->get_dist_func_param();
  TRACE_CHECK(index_->cur_element_count > 0, "empty hnswlib index");
}

id_t HnswExecutor::internal_of(id_t lbl) const {
  auto it = index_->label_lookup_.find(static_cast<hnswlib::labeltype>(lbl));
  TRACE_CHECK(it != index_->label_lookup_.end(), "label not in index");
  return static_cast<id_t>(it->second);
}

void HnswExecutor::fetch_vectors(const id_t* ids, size_t n, float* out,
                                 SearchStats* /*st*/) const {
  for (size_t i = 0; i < n; ++i)
    std::memcpy(out + i * dim_, vec(ids[i]), dim_ * sizeof(float));
}

id_t HnswExecutor::descend(const float* q, VisitRecord* visited,
                           SearchStats* st) const {
  auto* H = index_.get();
  hnswlib::tableint cur = H->enterpoint_node_;
  float curdist = fdist_(q, H->getDataByInternalId(cur), dparam_);
  st->n_dist++;
  if (visited) visited->push(cur, curdist);
  for (int level = H->maxlevel_; level > 0; level--) {
    bool changed = true;
    while (changed) {
      changed = false;
      unsigned int* data = (unsigned int*)H->get_linklist(cur, level);
      int size = H->getListCount(data);
      st->n_hops++;
      hnswlib::tableint* datal = (hnswlib::tableint*)(data + 1);
      for (int i = 0; i < size; i++) {
        hnswlib::tableint c = datal[i];
        float d = fdist_(q, H->getDataByInternalId(c), dparam_);
        st->n_dist++;
        if (visited) visited->push(c, d);
        if (d < curdist) {
          curdist = d;
          cur = c;
          changed = true;
        }
      }
    }
  }
  return static_cast<id_t>(cur);
}

void HnswExecutor::search(const float* q, const std::vector<id_t>& seeds,
                          const SearchParams& p, const BoundProvider* prune,
                          const EarlyStop* es, SearchResult* out,
                          VisitRecord* visited, SearchStats* st) const {
  auto* H = index_.get();
  const size_t ef = std::max<size_t>(static_cast<size_t>(p.L),
                                     static_cast<size_t>(p.k));
  VisitRecord* vr = p.collect_visited ? visited : nullptr;

  std::vector<HPair> top;   // max-heap on s: the result list W
  std::vector<HPair> cand;  // max-heap on -s: candidates to expand
  top.reserve(ef + 1);
  float lowerBound = kInf;

  hnswlib::VisitedList* vl = H->visited_list_pool_->getFreeVisitedList();
  hnswlib::vl_type* va = vl->mass;
  const hnswlib::vl_type tag = vl->curV;

  // ---- entry: cold descent and/or seeds
  std::vector<id_t> start;
  if (seeds.empty() || p.add_entry_to_seeds) start.push_back(descend(q, vr, st));
  start.insert(start.end(), seeds.begin(), seeds.end());
  for (id_t sid : start) {
    const auto hid = static_cast<hnswlib::tableint>(sid);
    if (va[hid] == tag) continue;
    va[hid] = tag;
    // hnswlib recomputes the entry distance at layer 0; we do the same.
    float d = fdist_(q, H->getDataByInternalId(hid), dparam_);
    st->n_dist++;
    if (vr) vr->push(sid, d);
    heap_push(top, d, hid);
    heap_push(cand, -d, hid);
    while (top.size() > ef) heap_pop(top);
  }
  lowerBound = top.front().first;

  ProgressFeatures pf;
  pf.s_entry = lowerBound;
  uint64_t next_check = es ? es->check_every() : UINT64_MAX;
  uint64_t admits = 0, evals_at_check = st->n_dist;

  // ---- layer-0 beam search (hnswlib searchBaseLayerST, bare-bone path)
  while (!cand.empty()) {
    const HPair cur = cand.front();
    const float cdist = -cur.first;
    if (cdist > lowerBound) break;
    heap_pop(cand);
    st->n_hops++;

    int* data = (int*)H->get_linklist0(cur.second);
    const size_t size = H->getListCount((hnswlib::linklistsizeint*)data);
    for (size_t j = 1; j <= size; j++) {
      const int cid = *(data + j);
      if (va[cid] == tag) continue;
      va[cid] = tag;
      // TRACE prune hook (Theorem 2): only when |W| = L_s, only for nodes
      // with a transferred bound, and only if lbS >= current threshold.
      if (prune != nullptr && top.size() == ef) {
        float lb;
        if (prune->lbS(static_cast<id_t>(cid), &lb) && lb >= lowerBound) {
          st->n_pruned++;
          prune->on_pruned(static_cast<id_t>(cid));
          continue;
        }
      }
      const float d = fdist_(q, H->getDataByInternalId(cid), dparam_);
      st->n_dist++;
      if (vr) vr->push(static_cast<id_t>(cid), d);
      if (top.size() < ef || lowerBound > d) {
        heap_push(cand, -d, static_cast<hnswlib::tableint>(cid));
        heap_push(top, d, static_cast<hnswlib::tableint>(cid));
        while (top.size() > ef) heap_pop(top);
        if (!top.empty()) lowerBound = top.front().first;
        ++admits;
      }
    }

    if (es != nullptr && st->n_dist >= next_check) {
      pf.n_dist = st->n_dist;
      pf.n_hops = st->n_hops;
      pf.n_admit = admits;
      pf.n_eval = st->n_dist - evals_at_check;
      fill_features(top, p.k, &pf);
      admits = 0;
      evals_at_check = st->n_dist;
      next_check = st->n_dist + es->check_every();
      std::vector<Neighbor> tk;
      if (es->wants_topk()) current_topk(top, p.k, &tk);
      if (es->should_stop(pf, es->wants_topk() ? &tk : nullptr)) {
        st->early_stopped = true;
        break;
      }
    }
  }
  H->visited_list_pool_->releaseVisitedList(vl);

  // ---- result: keep k (as hnswlib::searchKnn), then sort by (s, id)
  while (top.size() > static_cast<size_t>(p.k)) heap_pop(top);
  out->topk.clear();
  for (const auto& e : top) out->topk.push_back({e.first, static_cast<id_t>(e.second)});
  std::sort(out->topk.begin(), out->topk.end());
  if (p.want_vectors) {
    out->vectors.resize(out->topk.size() * dim_);
    for (size_t i = 0; i < out->topk.size(); ++i)
      std::memcpy(out->vectors.data() + i * dim_, vec(out->topk[i].id),
                  dim_ * sizeof(float));
  } else {
    out->vectors.clear();
  }
}

}  // namespace trace
