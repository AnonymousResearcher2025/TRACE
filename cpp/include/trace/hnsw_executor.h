// In-memory executor over an unmodified hnswlib 0.8.0 index.
//
// The cold path reproduces hnswlib::HierarchicalNSW::searchKnn exactly (same
// heaps, comparator, admission rule, and stop rule), so cold results are
// identical to hnswlib's own.
// TRACE adds three hooks: seed injection, a prune test immediately before a
// node's evaluation, and export of the visited set (Sec. IV-D).
#pragma once

#include <memory>
#include <string>

#include "hnswlib/hnswlib.h"
#include "trace/executor.h"

namespace trace {

class HnswExecutor : public Executor {
 public:
  HnswExecutor(const std::string& index_path, size_t dim);

  size_t dim() const override { return dim_; }
  size_t size() const override { return index_->cur_element_count; }
  const char* tier() const override { return "memory"; }

  float dist(const float* a, const float* b) const override {
    return fdist_(a, b, dparam_);
  }

  void fetch_vectors(const id_t* ids, size_t n, float* out,
                     SearchStats* st) const override;

  void search(const float* q, const std::vector<id_t>& seeds,
              const SearchParams& p, const BoundProvider* prune,
              const EarlyStop* es, SearchResult* out, VisitRecord* visited,
              SearchStats* st) const override;

  id_t entry_point() const override {
    return static_cast<id_t>(index_->enterpoint_node_);
  }
  id_t label(id_t internal) const override {
    return static_cast<id_t>(index_->getExternalLabel(internal));
  }

  // Internal id for an external label (throws if absent).
  id_t internal_of(id_t label) const;
  const float* vec(id_t internal) const {
    return reinterpret_cast<const float*>(
        index_->getDataByInternalId(static_cast<hnswlib::tableint>(internal)));
  }

 private:
  // Greedy descent over the upper layers, as in hnswlib::searchKnn.
  id_t descend(const float* q, VisitRecord* visited, SearchStats* st) const;

  size_t dim_;
  std::unique_ptr<hnswlib::L2Space> space_;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;
  hnswlib::DISTFUNC<float> fdist_;
  void* dparam_;
};

}  // namespace trace
