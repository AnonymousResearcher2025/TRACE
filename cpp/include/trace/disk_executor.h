// Disk-tier executor over an unmodified DiskANN disk index
// (<prefix>_disk.index, <prefix>_pq_pivots.bin, <prefix>_pq_compressed.bin).
//
// Layout (DiskANN v0.6/0.7): sector 0 holds metadata; node i lives in sector
// 1 + i / nnodes_per_sector at offset (i % nnodes_per_sector) * max_node_len
// and stores [float32 vector][uint32 nnbrs][uint32 nbrs...]. PQ codes stay in
// RAM. The search follows DiskANN's cached_beam_search: candidates are ordered
// by PQ distance, node pages are read in beams of beam_width, and the result is
// reranked with full-precision distances. Reads use O_DIRECT + libaio, or
// buffered pread for the page-cache experiment.
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "hnswlib/hnswlib.h"
#include "trace/executor.h"

namespace trace {

struct DiskScratch;

class DiskExecutor : public Executor {
 public:
  DiskExecutor(const std::string& prefix, bool direct_io, size_t cache_nodes,
               int max_beam_width = 16);
  ~DiskExecutor() override;

  size_t dim() const override { return dim_; }
  size_t size() const override { return npts_; }
  const char* tier() const override { return "disk"; }

  float dist(const float* a, const float* b) const override {
    return fdist_(a, b, dparam_);
  }

  void fetch_vectors(const id_t* ids, size_t n, float* out,
                     SearchStats* st) const override;

  void search(const float* q, const std::vector<id_t>& seeds,
              const SearchParams& p, const BoundProvider* prune,
              const EarlyStop* es, SearchResult* out, VisitRecord* visited,
              SearchStats* st) const override;

  id_t entry_point() const override { return static_cast<id_t>(medoid_); }
  id_t label(id_t internal) const override { return internal; }

  size_t max_degree() const { return max_degree_; }
  size_t num_pq_chunks() const { return nchunks_; }
  size_t cached_nodes() const { return cache_pos_.size(); }

 private:
  friend struct DiskScratch;
  // Reads one node synchronously (uncounted); returns its neighbor list.
  std::vector<id_t> read_node(id_t id, float* coords) const;
  void load_metadata();
  void load_pq(const std::string& prefix);
  void build_cache(size_t n);
  void compute_pq_table(const float* q, float* table) const;
  inline float pq_dist(const float* table, id_t id) const;
  uint64_t node_sector(id_t id) const;
  size_t node_offset_in_buf(id_t id) const;
  size_t read_len() const { return sectors_per_read_ * kSector; }
  // Reads the pages of ids into scratch buffers; returns elapsed ns.
  uint64_t read_pages(DiskScratch* sc, const id_t* ids, size_t n) const;
  DiskScratch* acquire() const;
  void release(DiskScratch* s) const;
  // Returns the scratch to the pool when it goes out of scope, including when
  // a read or a check throws.
  struct ScratchReturn {
    const DiskExecutor* ex;
    void operator()(DiskScratch* s) const;
  };
  using ScratchLease = std::unique_ptr<DiskScratch, ScratchReturn>;
  ScratchLease lease_scratch() const;

  static constexpr size_t kSector = 4096;

  std::string index_path_;
  bool direct_io_;
  int fd_ = -1;
  int max_beam_;
  size_t dim_ = 0;
  uint64_t npts_ = 0, medoid_ = 0, max_node_len_ = 0, nnodes_per_sector_ = 0;
  size_t sectors_per_read_ = 1;
  size_t max_degree_ = 0;

  // PQ (DiskANN FixedChunkPQTable, L2, no OPQ)
  size_t nchunks_ = 0;
  std::vector<float> pivots_;    // 256 x dim
  std::vector<float> centroid_;  // dim
  std::vector<uint32_t> chunk_off_;
  std::vector<uint8_t> codes_;   // npts x nchunks

  // node cache (BFS around the medoid, as DiskANN cache_bfs_levels)
  std::unordered_map<id_t, uint32_t> cache_pos_;
  std::vector<float> cache_coords_;
  std::vector<uint32_t> cache_nbr_off_;
  std::vector<id_t> cache_nbrs_;

  std::unique_ptr<hnswlib::L2Space> space_;
  hnswlib::DISTFUNC<float> fdist_;
  void* dparam_;

  mutable std::mutex pool_mu_;
  mutable std::vector<DiskScratch*> pool_;
  mutable std::vector<std::unique_ptr<DiskScratch>> owned_;
};

}  // namespace trace
