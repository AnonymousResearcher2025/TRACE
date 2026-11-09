// In-memory Vamana graphs, following DiskANN's in-memory index.
//
//   VamanaGraph     float32 vectors plus adjacency lists. Built incrementally
//                   with FreshVamana insertion (DiskANN insert_point:
//                   search_for_point_and_prune + inter_insert), or loaded from
//                   an index saved by DiskANN (diskannpy.build_memory_index).
//                   QVCache's mini-indexes are VamanaGraphs.
//   VamanaExecutor  Executor over a static VamanaGraph. CatapultDB evaluates
//                   on in-memory Vamana, so its fidelity check runs here.
//
// Search follows Index::iterate_to_fixed_point: a candidate list of capacity
// L sorted by (distance, id) (DiskANN's NeighborPriorityQueue), expanding the
// closest unexpanded node until none remains. Pruning follows occlude_list:
// alpha is stepped by 1.2 from 1 up to alpha, the pool is capped at maxc, and
// distances are squared L2 (DiskANN's L2 metric). Distances use hnswlib's L2
// kernel, the same kernel as HnswExecutor.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "hnswlib/hnswlib.h"
#include "trace/executor.h"

namespace trace {

// DiskANN defaults (build_memory_index: R=64, L=100, alpha=1.2; maxc=750;
// GRAPH_SLACK_FACTOR=1.3).
struct VamanaParams {
  uint32_t R = 64;
  uint32_t L_build = 100;
  float alpha = 1.2f;
  uint32_t maxc = 750;
  float slack = 1.3f;
};

// DiskANN NeighborPriorityQueue: sorted by (s, id), capacity L, a cursor at
// the first unexpanded entry.
class CandidateList {
 public:
  explicit CandidateList(size_t cap = 0) { reset(cap); }
  void reset(size_t cap) {
    cap_ = cap;
    size_ = 0;
    cur_ = 0;
    data_.resize(cap + 1);
  }
  size_t size() const { return size_; }
  bool full() const { return size_ == cap_; }
  const Neighbor& worst() const { return data_[size_ - 1].nb; }
  const Neighbor& at(size_t i) const { return data_[i].nb; }
  bool has_unexpanded() const { return cur_ < size_; }
  void insert(const Neighbor& nb);
  Neighbor closest_unexpanded();

 private:
  struct Entry {
    Neighbor nb;
    bool expanded;
  };
  std::vector<Entry> data_;
  size_t cap_ = 0, size_ = 0, cur_ = 0;
};

class VamanaGraph {
 public:
  VamanaGraph(size_t dim, size_t capacity, const VamanaParams& p = {});
  // Loads a graph saved by DiskANN (no frozen points) and its float32 vectors
  // (DiskANN .bin / .fbin: int32 n, int32 d, then n x d floats).
  static std::unique_ptr<VamanaGraph> load_diskann(const std::string& graph_path,
                                                   const std::string& vectors_path,
                                                   const VamanaParams& p = {});
  void save_diskann(const std::string& graph_path) const;

  size_t dim() const { return dim_; }
  size_t size() const { return n_; }
  size_t capacity() const { return cap_; }
  id_t start() const { return start_; }
  const VamanaParams& params() const { return p_; }
  const float* vec(id_t i) const { return data_.data() + static_cast<size_t>(i) * dim_; }
  const std::vector<id_t>& neighbors(id_t i) const { return adj_[i]; }
  float dist(const float* a, const float* b) const { return fdist_(a, b, dparam_); }
  size_t bytes() const;  // vectors + adjacency of the n stored points

  void clear();  // drops all points (QVCache evicts a mini-index whole)
  // FreshVamana insertion; returns the new local id. Build-time distance
  // evaluations are added to st->n_aux if st is given.
  id_t insert(const float* v, SearchStats* st = nullptr);

  // iterate_to_fixed_point from init (default: the start node). The prune
  // test runs immediately before a node's evaluation and only when the list
  // is full; a node is skipped only if lbS > worst.s, so it would have been
  // rejected by the list (Theorem 2). expanded receives expanded nodes in
  // order (insertion pool); visited receives every evaluated node.
  void greedy(const float* q, size_t L, const std::vector<id_t>& init,
              CandidateList* cand, const BoundProvider* prune,
              VisitRecord* visited, std::vector<Neighbor>* expanded,
              SearchStats* st) const;

  // Top-k of a greedy search with list size L (L >= k), sorted.
  void search(const float* q, size_t k, size_t L, std::vector<Neighbor>* out,
              SearchStats* st) const;

 private:
  void prune_list(id_t loc, std::vector<Neighbor>* pool, std::vector<id_t>* out,
                  SearchStats* st) const;
  float* mvec(id_t i) { return data_.data() + static_cast<size_t>(i) * dim_; }

  size_t dim_, cap_;
  VamanaParams p_;
  size_t n_ = 0;
  id_t start_ = 0;
  std::vector<float> data_;
  std::vector<std::vector<id_t>> adj_;
  std::unique_ptr<hnswlib::L2Space> space_;
  hnswlib::DISTFUNC<float> fdist_;
  void* dparam_;
  std::unique_ptr<hnswlib::VisitedListPool> pool_;
};

class VamanaExecutor : public Executor {
 public:
  VamanaExecutor(const std::string& graph_path, const std::string& vectors_path);

  size_t dim() const override { return g_->dim(); }
  size_t size() const override { return g_->size(); }
  const char* tier() const override { return "memory"; }
  float dist(const float* a, const float* b) const override { return g_->dist(a, b); }
  void fetch_vectors(const id_t* ids, size_t n, float* out,
                     SearchStats* st) const override;
  // Early termination is not supported on this executor (es must be null).
  void search(const float* q, const std::vector<id_t>& seeds,
              const SearchParams& p, const BoundProvider* prune,
              const EarlyStop* es, SearchResult* out, VisitRecord* visited,
              SearchStats* st) const override;
  id_t entry_point() const override { return g_->start(); }
  id_t label(id_t internal) const override { return internal; }
  const VamanaGraph& graph() const { return *g_; }

 private:
  std::unique_ptr<VamanaGraph> g_;
};

}  // namespace trace
