#include "trace/disk_executor.h"

#include <fcntl.h>
#include <libaio.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fstream>

#include "trace/idmap.h"

namespace trace {

// ----------------------------------------------------------------- scratch
struct DiskScratch {
  io_context_t ctx = 0;
  char* buf = nullptr;  // max_beam * read_len, 4 KB aligned
  size_t nbuf = 0;
  IdMap visited{4096};
  std::vector<float> pq_table;
  std::vector<float> coords;  // arena: one vector per evaluated node
  std::vector<iocb> cbs;
  std::vector<iocb*> cbp;
  std::vector<io_event> evs;

  DiskScratch(const DiskExecutor& ex, int max_beam) {
    nbuf = static_cast<size_t>(max_beam);
    void* p = nullptr;
    TRACE_CHECK(posix_memalign(&p, 4096, nbuf * ex.read_len()) == 0,
                "posix_memalign failed");
    buf = static_cast<char*>(p);
    std::memset(buf, 0, nbuf * ex.read_len());
    if (ex.direct_io_) {
      int r = io_setup(static_cast<int>(nbuf), &ctx);
      TRACE_CHECK(r == 0, "io_setup failed (" + std::to_string(-r) + ")");
    }
    pq_table.resize(256 * ex.nchunks_);
    cbs.resize(nbuf);
    cbp.resize(nbuf);
    evs.resize(nbuf);
  }
  ~DiskScratch() {
    if (ctx) io_destroy(ctx);
    free(buf);
  }
};

namespace {
// DiskANN NeighborPriorityQueue: sorted array with an expansion cursor.
struct RetCand {
  id_t id;
  float d;
  bool expanded;
};
class RetSet {
 public:
  explicit RetSet(size_t cap) : cap_(cap) { data_.resize(cap + 1); }
  void insert(id_t id, float d) {
    if (size_ == cap_ && less(data_[size_ - 1], d, id)) return;
    size_t lo = 0, hi = size_;
    while (lo < hi) {
      size_t mid = (lo + hi) >> 1;
      if (less_new(d, id, data_[mid])) {
        hi = mid;
      } else if (data_[mid].id == id) {
        return;
      } else {
        lo = mid + 1;
      }
    }
    if (lo < cap_) {
      std::memmove(&data_[lo + 1], &data_[lo], (size_ - lo) * sizeof(RetCand));
    }
    data_[lo] = {id, d, false};
    if (size_ < cap_) size_++;
    if (lo < cur_) cur_ = lo;
  }
  bool has_unexpanded() const { return cur_ < size_; }
  RetCand closest_unexpanded() {
    data_[cur_].expanded = true;
    size_t pre = cur_;
    while (cur_ < size_ && data_[cur_].expanded) cur_++;
    return data_[pre];
  }

 private:
  // a < (d,id) in DiskANN's Neighbor order
  static bool less(const RetCand& a, float d, id_t id) {
    return a.d < d || (a.d == d && a.id < id);
  }
  static bool less_new(float d, id_t id, const RetCand& b) {
    return d < b.d || (d == b.d && id < b.id);
  }
  std::vector<RetCand> data_;
  size_t cap_, size_ = 0, cur_ = 0;
};

template <typename T>
T read_pod(std::ifstream& in) {
  T v;
  in.read(reinterpret_cast<char*>(&v), sizeof(T));
  TRACE_CHECK(in.good(), "unexpected end of file");
  return v;
}
}  // namespace

// ------------------------------------------------------------------- load
DiskExecutor::DiskExecutor(const std::string& prefix, bool direct_io,
                           size_t cache_nodes, int max_beam_width)
    : index_path_(prefix + "_disk.index"),
      direct_io_(direct_io),
      max_beam_(max_beam_width) {
  load_metadata();
  load_pq(prefix);
  space_ = std::make_unique<hnswlib::L2Space>(dim_);
  fdist_ = space_->get_dist_func();
  dparam_ = space_->get_dist_func_param();
  int flags = O_RDONLY;
  if (direct_io_) flags |= O_DIRECT;
  fd_ = ::open(index_path_.c_str(), flags);
  TRACE_CHECK(fd_ >= 0, "cannot open " + index_path_ + ": " + std::strerror(errno));
  build_cache(cache_nodes);
}

DiskExecutor::~DiskExecutor() {
  if (fd_ >= 0) ::close(fd_);
}

void DiskExecutor::load_metadata() {
  std::ifstream in(index_path_, std::ios::binary);
  TRACE_CHECK(in.good(), "cannot open " + index_path_);
  const int32_t nr = read_pod<int32_t>(in);
  const int32_t nc = read_pod<int32_t>(in);
  TRACE_CHECK(nc == 1 && nr >= 9, "unexpected disk index metadata header");
  std::vector<uint64_t> meta(static_cast<size_t>(nr));
  in.read(reinterpret_cast<char*>(meta.data()), nr * sizeof(uint64_t));
  TRACE_CHECK(in.good(), "truncated metadata");
  npts_ = meta[0];
  dim_ = static_cast<size_t>(meta[1]);
  medoid_ = meta[2];
  max_node_len_ = meta[3];
  nnodes_per_sector_ = meta[4];
  const uint64_t num_frozen = meta[5];
  TRACE_CHECK(num_frozen == 0, "frozen points are not supported");
  const uint64_t coord_bytes = dim_ * sizeof(float);
  TRACE_CHECK(max_node_len_ > coord_bytes + 4 &&
                  (max_node_len_ - coord_bytes) % 4 == 0,
              "max_node_len inconsistent with float32 vectors");
  max_degree_ = (max_node_len_ - coord_bytes) / 4 - 1;
  if (nnodes_per_sector_ > 0) {
    TRACE_CHECK(nnodes_per_sector_ * max_node_len_ <= kSector,
                "nnodes_per_sector inconsistent");
    sectors_per_read_ = 1;
  } else {
    sectors_per_read_ = (max_node_len_ + kSector - 1) / kSector;
  }
  TRACE_CHECK(medoid_ < npts_, "medoid out of range");
}

void DiskExecutor::load_pq(const std::string& prefix) {
  {
    std::ifstream in(prefix + "_pq_pivots.bin", std::ios::binary);
    TRACE_CHECK(in.good(), "cannot open " + prefix + "_pq_pivots.bin");
    const int32_t nr = read_pod<int32_t>(in);
    const int32_t nc = read_pod<int32_t>(in);
    TRACE_CHECK(nc == 1 && nr == 4,
                "unsupported PQ pivots layout (expected 4 offsets)");
    std::vector<uint64_t> off(4);
    in.read(reinterpret_cast<char*>(off.data()), 4 * sizeof(uint64_t));
    // pivots: 256 x dim
    in.seekg(static_cast<std::streamoff>(off[0]));
    int32_t r = read_pod<int32_t>(in), c = read_pod<int32_t>(in);
    TRACE_CHECK(r == 256 && static_cast<size_t>(c) == dim_, "bad pivots shape");
    pivots_.resize(256 * dim_);
    in.read(reinterpret_cast<char*>(pivots_.data()), pivots_.size() * sizeof(float));
    // centroid: dim x 1
    in.seekg(static_cast<std::streamoff>(off[1]));
    r = read_pod<int32_t>(in);
    c = read_pod<int32_t>(in);
    TRACE_CHECK(static_cast<size_t>(r) == dim_ && c == 1, "bad centroid shape");
    centroid_.resize(dim_);
    in.read(reinterpret_cast<char*>(centroid_.data()), dim_ * sizeof(float));
    // chunk offsets: (nchunks+1) x 1
    in.seekg(static_cast<std::streamoff>(off[2]));
    r = read_pod<int32_t>(in);
    c = read_pod<int32_t>(in);
    TRACE_CHECK(r >= 2 && c == 1, "bad chunk offsets shape");
    chunk_off_.resize(static_cast<size_t>(r));
    in.read(reinterpret_cast<char*>(chunk_off_.data()), r * sizeof(uint32_t));
    TRACE_CHECK(in.good(), "truncated PQ pivots file");
    nchunks_ = static_cast<size_t>(r) - 1;
    TRACE_CHECK(chunk_off_.front() == 0 && chunk_off_.back() == dim_,
                "chunk offsets do not cover the dimensions");
  }
  {
    std::ifstream in(prefix + "_pq_compressed.bin", std::ios::binary);
    TRACE_CHECK(in.good(), "cannot open " + prefix + "_pq_compressed.bin");
    const int32_t n = read_pod<int32_t>(in);
    const int32_t nc = read_pod<int32_t>(in);
    TRACE_CHECK(static_cast<uint64_t>(n) == npts_ &&
                    static_cast<size_t>(nc) == nchunks_,
                "PQ codes inconsistent with index");
    codes_.resize(npts_ * nchunks_);
    in.read(reinterpret_cast<char*>(codes_.data()), codes_.size());
    TRACE_CHECK(in.good(), "truncated PQ codes");
  }
}

uint64_t DiskExecutor::node_sector(id_t id) const {
  if (nnodes_per_sector_ > 0) return 1 + id / nnodes_per_sector_;
  return 1 + static_cast<uint64_t>(id) * sectors_per_read_;
}

size_t DiskExecutor::node_offset_in_buf(id_t id) const {
  if (nnodes_per_sector_ > 0) return (id % nnodes_per_sector_) * max_node_len_;
  return 0;
}

std::vector<id_t> DiskExecutor::read_node(id_t id, float* coords) const {
  TRACE_CHECK(id < npts_, "node id out of range");
  std::ifstream in(index_path_, std::ios::binary);
  std::vector<char> page(read_len());
  in.seekg(static_cast<std::streamoff>(node_sector(id) * kSector));
  in.read(page.data(), static_cast<std::streamsize>(page.size()));
  TRACE_CHECK(in.good(), "cannot read node page");
  const char* node = page.data() + node_offset_in_buf(id);
  if (coords) std::memcpy(coords, node, dim_ * sizeof(float));
  uint32_t nn;
  std::memcpy(&nn, node + dim_ * sizeof(float), 4);
  TRACE_CHECK(nn <= max_degree_, "corrupt neighbor count");
  std::vector<id_t> nbrs(nn);
  std::memcpy(nbrs.data(), node + dim_ * sizeof(float) + 4, nn * 4);
  return nbrs;
}

void DiskExecutor::build_cache(size_t n) {
  n = std::min<size_t>(n, npts_);
  if (n == 0) return;
  std::vector<id_t> level{static_cast<id_t>(medoid_)};
  IdMap seen(4 * n + 16);
  seen.insert(static_cast<id_t>(medoid_), 0);
  std::vector<float> coords(dim_);
  while (cache_pos_.size() < n && !level.empty()) {
    std::vector<id_t> next;
    for (id_t id : level) {
      if (cache_pos_.size() >= n) break;
      std::vector<id_t> nbrs = read_node(id, coords.data());
      const uint32_t pos = static_cast<uint32_t>(cache_pos_.size());
      cache_pos_.emplace(id, pos);
      cache_coords_.insert(cache_coords_.end(), coords.begin(), coords.end());
      cache_nbr_off_.push_back(static_cast<uint32_t>(cache_nbrs_.size()));
      cache_nbrs_.insert(cache_nbrs_.end(), nbrs.begin(), nbrs.end());
      for (id_t nb : nbrs)
        if (seen.insert(nb, 0).second) next.push_back(nb);
    }
    level.swap(next);
  }
  cache_nbr_off_.push_back(static_cast<uint32_t>(cache_nbrs_.size()));
}

// ---------------------------------------------------------------- PQ
void DiskExecutor::compute_pq_table(const float* q, float* table) const {
  std::vector<float> qr(dim_);
  for (size_t j = 0; j < dim_; ++j) qr[j] = q[j] - centroid_[j];
  std::fill(table, table + 256 * nchunks_, 0.f);
  for (size_t c = 0; c < nchunks_; ++c) {
    float* t = table + 256 * c;
    for (size_t j = chunk_off_[c]; j < chunk_off_[c + 1]; ++j) {
      for (size_t i = 0; i < 256; ++i) {
        const float diff = pivots_[i * dim_ + j] - qr[j];
        t[i] += diff * diff;
      }
    }
  }
}

inline float DiskExecutor::pq_dist(const float* table, id_t id) const {
  const uint8_t* code = codes_.data() + static_cast<size_t>(id) * nchunks_;
  float s = 0.f;
  for (size_t c = 0; c < nchunks_; ++c) s += table[256 * c + code[c]];
  return s;
}

// ------------------------------------------------------------------ I/O
DiskScratch* DiskExecutor::acquire() const {
  std::lock_guard<std::mutex> g(pool_mu_);
  if (!pool_.empty()) {
    DiskScratch* s = pool_.back();
    pool_.pop_back();
    return s;
  }
  owned_.push_back(std::make_unique<DiskScratch>(*this, max_beam_));
  return owned_.back().get();
}

void DiskExecutor::release(DiskScratch* s) const {
  std::lock_guard<std::mutex> g(pool_mu_);
  pool_.push_back(s);
}

DiskExecutor::ScratchLease DiskExecutor::lease_scratch() const {
  return ScratchLease(acquire(), ScratchReturn{this});
}

void DiskExecutor::ScratchReturn::operator()(DiskScratch* s) const {
  ex->release(s);
}

uint64_t DiskExecutor::read_pages(DiskScratch* sc, const id_t* ids,
                                  size_t n) const {
  TRACE_CHECK(n <= sc->nbuf, "beam larger than scratch buffers");
  if (n == 0) return 0;
  const uint64_t t0 = now_ns();
  const size_t len = read_len();
  if (direct_io_) {
    for (size_t i = 0; i < n; ++i) {
      io_prep_pread(&sc->cbs[i], fd_, sc->buf + i * len, len,
                    static_cast<long long>(node_sector(ids[i]) * kSector));
      sc->cbp[i] = &sc->cbs[i];
    }
    size_t submitted = 0;
    while (submitted < n) {
      int r = io_submit(sc->ctx, static_cast<long>(n - submitted),
                        sc->cbp.data() + submitted);
      if (r == -EAGAIN || r == -EINTR) continue;
      TRACE_CHECK(r > 0, "io_submit failed (" + std::to_string(-r) + ")");
      submitted += static_cast<size_t>(r);
    }
    size_t done = 0;
    while (done < n) {
      int r = io_getevents(sc->ctx, static_cast<long>(n - done),
                           static_cast<long>(n - done), sc->evs.data(), nullptr);
      if (r == -EINTR) continue;
      TRACE_CHECK(r > 0, "io_getevents failed (" + std::to_string(-r) + ")");
      for (int e = 0; e < r; ++e)
        TRACE_CHECK(sc->evs[e].res == len, "short or failed read");
      done += static_cast<size_t>(r);
    }
  } else {
    for (size_t i = 0; i < n; ++i) {
      ssize_t got = ::pread(fd_, sc->buf + i * len, len,
                            static_cast<off_t>(node_sector(ids[i]) * kSector));
      TRACE_CHECK(got == static_cast<ssize_t>(len), "pread failed");
    }
  }
  return now_ns() - t0;
}

void DiskExecutor::fetch_vectors(const id_t* ids, size_t n, float* out,
                                 SearchStats* st) const {
  const auto lease = lease_scratch();
  DiskScratch* sc = lease.get();
  std::vector<id_t> batch;
  std::vector<size_t> where;
  auto flush = [&]() {
    if (batch.empty()) return;
    const uint64_t ns = read_pages(sc, batch.data(), batch.size());
    if (st) {
      st->n_reads += batch.size();
      st->io_ns += ns;
    }
    for (size_t b = 0; b < batch.size(); ++b) {
      const char* node = sc->buf + b * read_len() + node_offset_in_buf(batch[b]);
      std::memcpy(out + where[b] * dim_, node, dim_ * sizeof(float));
    }
    batch.clear();
    where.clear();
  };
  for (size_t i = 0; i < n; ++i) {
    auto it = cache_pos_.find(ids[i]);
    if (it != cache_pos_.end()) {
      std::memcpy(out + i * dim_, cache_coords_.data() + it->second * dim_,
                  dim_ * sizeof(float));
      if (st) st->n_cache_hits++;
      continue;
    }
    batch.push_back(ids[i]);
    where.push_back(i);
    if (batch.size() == sc->nbuf) flush();
  }
  flush();
}

// --------------------------------------------------------------- search
void DiskExecutor::search(const float* q, const std::vector<id_t>& seeds,
                          const SearchParams& p, const BoundProvider* prune,
                          const EarlyStop* es, SearchResult* out,
                          VisitRecord* visited, SearchStats* st) const {
  TRACE_CHECK(p.beam_width >= 1 && p.beam_width <= max_beam_, "bad beam width");
  const auto lease = lease_scratch();
  DiskScratch* sc = lease.get();
  VisitRecord* vr = p.collect_visited ? visited : nullptr;
  const size_t L = static_cast<size_t>(std::max(p.L, p.k));
  const size_t W = static_cast<size_t>(p.beam_width);

  compute_pq_table(q, sc->pq_table.data());
  const float* table = sc->pq_table.data();
  sc->visited.clear();
  sc->coords.clear();

  RetSet retset(L);
  std::vector<Neighbor> full;      // evaluated nodes with full distance
  std::vector<uint32_t> full_pos;  // position of each node's vector in arena
  std::vector<float> theta_heap;   // max-heap of the L smallest full s
  float thetaS = kInf;

  auto push_theta = [&](float s) {
    theta_heap.push_back(s);
    std::push_heap(theta_heap.begin(), theta_heap.end());
    if (theta_heap.size() > L) {
      std::pop_heap(theta_heap.begin(), theta_heap.end());
      theta_heap.pop_back();
    }
    if (theta_heap.size() == L) thetaS = theta_heap.front();
  };

  std::vector<id_t> start;
  if (seeds.empty() || p.add_entry_to_seeds) start.push_back(static_cast<id_t>(medoid_));
  start.insert(start.end(), seeds.begin(), seeds.end());
  for (id_t sid : start) {
    if (!sc->visited.insert(sid, 0).second) continue;
    retset.insert(sid, pq_dist(table, sid));
    st->n_pq++;
  }

  // Expands one evaluated node: full distance, residue record, neighbors.
  auto expand = [&](id_t id, const float* coords, const id_t* nbrs, uint32_t nn) {
    const float s = fdist_(q, coords, dparam_);
    st->n_dist++;
    full.push_back({s, id});
    full_pos.push_back(static_cast<uint32_t>(sc->coords.size() / dim_));
    if (p.want_vectors) sc->coords.insert(sc->coords.end(), coords, coords + dim_);
    push_theta(s);
    if (vr) vr->push(id, s);
    for (uint32_t m = 0; m < nn; ++m) {
      const id_t nb = nbrs[m];
      if (!sc->visited.insert(nb, 0).second) continue;
      retset.insert(nb, pq_dist(table, nb));
      st->n_pq++;
    }
  };

  std::vector<id_t> frontier, cached;
  frontier.reserve(W);
  uint64_t next_check = es ? es->check_every() : UINT64_MAX;
  uint64_t evals_at_check = 0;
  ProgressFeatures pf;
  bool first_eval = true;

  while (retset.has_unexpanded()) {
    frontier.clear();
    cached.clear();
    size_t num_seen = 0;
    while (retset.has_unexpanded() && frontier.size() < W && num_seen < W) {
      const RetCand c = retset.closest_unexpanded();
      // TRACE prune (Proposition 1): skip a retained node whose bound shows
      // it cannot beat the L-th best full distance among fetched nodes.
      if (prune != nullptr && theta_heap.size() == L) {
        float lb;
        if (prune->lbS(c.id, &lb) && lb >= thetaS) {
          st->n_pruned++;
          prune->on_pruned(c.id);
          continue;
        }
      }
      num_seen++;
      if (cache_pos_.count(c.id)) {
        cached.push_back(c.id);
        st->n_cache_hits++;
      } else {
        frontier.push_back(c.id);
      }
    }
    if (!frontier.empty()) {
      st->io_ns += read_pages(sc, frontier.data(), frontier.size());
      st->n_reads += frontier.size();
    }
    for (id_t id : cached) {
      const uint32_t pos = cache_pos_.at(id);
      const uint32_t b = cache_nbr_off_[pos], e = cache_nbr_off_[pos + 1];
      expand(id, cache_coords_.data() + static_cast<size_t>(pos) * dim_,
             cache_nbrs_.data() + b, e - b);
    }
    for (size_t i = 0; i < frontier.size(); ++i) {
      const char* node = sc->buf + i * read_len() + node_offset_in_buf(frontier[i]);
      uint32_t nn;
      std::memcpy(&nn, node + dim_ * sizeof(float), 4);
      TRACE_CHECK(nn <= max_degree_, "corrupt node page");
      expand(frontier[i], reinterpret_cast<const float*>(node),
             reinterpret_cast<const id_t*>(node + dim_ * sizeof(float) + 4), nn);
    }
    if (!cached.empty() || !frontier.empty()) st->n_hops++;
    if (first_eval && !full.empty()) {
      pf.s_entry = full.front().s;
      first_eval = false;
    }

    if (es != nullptr && st->n_dist >= next_check && !full.empty()) {
      std::vector<float> s(full.size());
      for (size_t i = 0; i < full.size(); ++i) s[i] = full[i].s;
      const size_t kk = std::min<size_t>(static_cast<size_t>(p.k), s.size()) - 1;
      std::nth_element(s.begin(), s.begin() + kk, s.end());
      pf.s_kth = s[kk];
      pf.s_first = *std::min_element(s.begin(), s.begin() + kk + 1);
      pf.s_bound = thetaS;
      pf.n_dist = st->n_dist;
      pf.n_hops = st->n_hops;
      pf.n_eval = st->n_dist - evals_at_check;
      pf.n_admit = pf.n_eval;
      evals_at_check = st->n_dist;
      next_check = st->n_dist + es->check_every();
      std::vector<Neighbor> tk;
      if (es->wants_topk()) {
        tk = full;
        std::sort(tk.begin(), tk.end());
        if (tk.size() > static_cast<size_t>(p.k)) tk.resize(static_cast<size_t>(p.k));
      }
      if (es->should_stop(pf, es->wants_topk() ? &tk : nullptr)) {
        st->early_stopped = true;
        break;
      }
    }
  }

  // Rerank with full-precision distances (DiskANN: sort full_retset).
  std::vector<uint32_t> order(full.size());
  for (uint32_t i = 0; i < order.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(),
            [&](uint32_t a, uint32_t b) { return full[a] < full[b]; });
  const size_t kk = std::min<size_t>(static_cast<size_t>(p.k), full.size());
  out->topk.clear();
  out->vectors.clear();
  for (size_t i = 0; i < kk; ++i) out->topk.push_back(full[order[i]]);
  if (p.want_vectors) {
    out->vectors.resize(kk * dim_);
    for (size_t i = 0; i < kk; ++i)
      std::memcpy(out->vectors.data() + i * dim_,
                  sc->coords.data() + static_cast<size_t>(full_pos[order[i]]) * dim_,
                  dim_ * sizeof(float));
  }
}

}  // namespace trace
