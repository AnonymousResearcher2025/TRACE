#include <algorithm>
#include <cstring>
#include <numeric>

#include "trace/baselines.h"
#include "trace/engine.h"

namespace trace {

// ------------------------------------------------------------- transfer
void transfer_bounds(const std::vector<const Residue*>& ring,
                     const std::vector<float>& deltas, const Envelope& env,
                     Bounds* B) {
  B->clear();
  size_t total = 0;
  for (const Residue* r : ring) total += r->ids.size();
  B->map.reserve(total);
  B->ids.reserve(total);
  for (size_t i = 0; i < ring.size(); ++i) {
    const Residue& r = *ring[i];
    const float delta = deltas[i];
    const uint8_t ai = static_cast<uint8_t>(i);
    for (size_t j = 0; j < r.ids.size(); ++j) {
      const float D = f16_to_f32(r.d16[j]);
      const float lbi = lb_from_anchor(D, delta, env.eps_plus);
      const float ubi = ub_from_anchor(D, delta, env.eps_plus);
      const uint32_t next = static_cast<uint32_t>(B->ids.size());
      auto ins = B->map.insert(r.ids[j], next);
      if (ins.second) {
        B->ids.push_back(r.ids[j]);
        B->lb.push_back(lbi);
        B->ub.push_back(ubi);
        B->lb_arg.push_back(ai);
        B->ub_arg.push_back(ai);
      } else {
        const uint32_t s = ins.first;
        if (lbi > B->lb[s]) {
          B->lb[s] = lbi;
          B->lb_arg[s] = ai;
        }
        if (ubi < B->ub[s]) {
          B->ub[s] = ubi;
          B->ub_arg[s] = ai;
        }
      }
    }
  }
  B->lbS_.resize(B->ids.size());
  for (size_t s = 0; s < B->ids.size(); ++s) B->lbS_[s] = lbS_from_lb(B->lb[s], env);
}

// --------------------------------------------------------------- engine
TraceEngine::TraceEngine(const Executor& ex, const TraceConfig& cfg,
                         Catapult* catapult, const EarlyStop* es)
    : ex_(ex), cfg_(cfg), catapult_(catapult), es_(es) {
  TRACE_CHECK(cfg_.k >= 1 && cfg_.L >= cfg_.k, "need 1 <= k <= L");
  TRACE_CHECK(cfg_.m >= 1 && cfg_.m <= 255, "m must be in [1, 255]");
  TRACE_CHECK(cfg_.w >= 1, "w must be >= 1");
  TRACE_CHECK(cfg_.s_max >= 0 && cfg_.B_r >= 0, "bad s_max or B_r");
  TRACE_CHECK(cfg_.transfer || cfg_.seeds == TraceConfig::Seeds::PrevAnswer ||
                  !cfg_.enabled,
              "transfer=false requires seeds=PrevAnswer");
  env_ = Envelope::make(ex.dim(), cfg_.max_norm, /*fp16_storage=*/true);
  credit_.assign(static_cast<size_t>(cfg_.m), 0.0);
}

TraceEngine::Session& TraceEngine::session(int64_t sid) {
  if (cfg_.share_ring) sid = 0;
  return sessions_[sid];
}

void TraceEngine::end_session(int64_t sid) {
  if (!cfg_.share_ring) sessions_.erase(sid);
}

size_t TraceEngine::state_bytes() const {
  size_t b = 0;
  for (const auto& kv : sessions_)
    for (const auto& r : kv.second.ring) b += r->bytes();
  return b;
}

float TraceEngine::scale(const Session& S) const {
  std::vector<float> v(S.scale_hist.begin(), S.scale_hist.end());
  const size_t mid = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + mid, v.end());
  if (v.size() % 2 == 1) return v[mid];
  const float hi = v[mid];
  const float lo = *std::max_element(v.begin(), v.begin() + mid);
  return 0.5f * (lo + hi);
}

void TraceEngine::record_scale(Session& S, const std::vector<Neighbor>& topk) {
  if (topk.size() < static_cast<size_t>(cfg_.k)) return;
  S.scale_hist.push_back(e_from_s(topk[static_cast<size_t>(cfg_.k) - 1].s));
  while (S.scale_hist.size() > static_cast<size_t>(cfg_.w)) S.scale_hist.pop_front();
}

std::unique_ptr<Residue> TraceEngine::make_full_residue(
    const float* q, const VisitRecord& vr, const SearchResult& res) const {
  auto r = std::make_unique<Residue>();
  r->anchor.assign(q, q + ex_.dim());
  r->epoch = ex_.epoch();
  IdMap& seen = dedupe_;
  seen.clear();
  seen.reserve(vr.ids.size());
  r->ids.reserve(vr.ids.size());
  r->d16.reserve(vr.ids.size());
  for (size_t i = 0; i < vr.ids.size(); ++i) {
    if (!seen.insert(vr.ids[i], 0).second) continue;  // upper-layer repeats
    r->ids.push_back(vr.ids[i]);
    r->d16.push_back(f32_to_f16(e_from_s(vr.s[i])));
  }
  for (const Neighbor& nb : res.topk) {
    TRACE_CHECK(seen.contains(nb.id), "answer node missing from visited set");
    r->ans.push_back(nb.id);
  }
  r->ans_vecs = res.vectors;
  TRACE_CHECK(r->ans_vecs.size() == r->ans.size() * ex_.dim(),
              "answer vectors missing");
  return r;
}

// Older residue: anchor + pinned answers + the B_r closest other entries.
void TraceEngine::compact(Residue* r) const {
  if (r->compacted) return;
  r->compacted = true;
  r->ans_vecs.clear();
  r->ans_vecs.shrink_to_fit();
  if (r->thin) return;
  IdMap pinned(r->ans.size() + 16);
  for (id_t a : r->ans) pinned.insert(a, 0);
  std::vector<std::pair<float, id_t>> others;
  std::vector<id_t> keep_ids;
  std::vector<uint16_t> keep_d;
  for (size_t j = 0; j < r->ids.size(); ++j) {
    if (pinned.contains(r->ids[j])) {
      keep_ids.push_back(r->ids[j]);
      keep_d.push_back(r->d16[j]);
    } else {
      others.emplace_back(f16_to_f32(r->d16[j]), r->ids[j]);
    }
  }
  const size_t B = static_cast<size_t>(cfg_.B_r);
  if (others.size() > B) {
    std::nth_element(others.begin(), others.begin() + B, others.end());
    others.resize(B);
  }
  for (const auto& o : others) {
    keep_ids.push_back(o.second);
    keep_d.push_back(f32_to_f16(o.first));  // exact: o.first came from f16
  }
  r->ids.swap(keep_ids);
  r->d16.swap(keep_d);
  r->ids.shrink_to_fit();
  r->d16.shrink_to_fit();
}

// Ring update. The newest residue is never evicted, and neither is the most
// recent traversal (non-thin) residue: thin residues from resolution must not
// push the last real search out of R (Sec. III-D). For m >= 2 the ring keeps
// m residues; with m = 1 the traversal residue stays as one extra slot.
void TraceEngine::push_residue(Session& S, std::unique_ptr<Residue> r) {
  if (!S.ring.empty()) compact(S.ring.front().get());
  S.ring.push_front(std::move(r));
  size_t keep = 0;  // index of the most recent traversal residue
  while (keep < S.ring.size() && S.ring[keep]->thin) ++keep;
  while (S.ring.size() > static_cast<size_t>(cfg_.m)) {
    size_t victim = 0;  // 0 = no candidate
    for (size_t i = S.ring.size() - 1; i >= 1; --i) {
      if (i == keep) continue;
      if (victim == 0) {
        victim = i;  // oldest candidate
        if (cfg_.retention == TraceConfig::Retention::Recency) break;
        continue;
      }
      // Hybrid: smallest utility; ties go to the oldest (found first).
      if (S.ring[i]->utility < S.ring[victim]->utility) victim = i;
    }
    if (victim == 0) break;
    S.ring.erase(S.ring.begin() + static_cast<long>(victim));
    if (victim < keep) --keep;
  }
}

void TraceEngine::finish(Session& S, const float* /*q*/,
                         const std::vector<Neighbor>& topk,
                         QueryOut* out) const {
  out->labels.clear();
  out->s.clear();
  for (const Neighbor& nb : topk) {
    out->labels.push_back(ex_.label(nb.id));
    out->s.push_back(nb.s);
  }
  out->sigma = S.sigma;
  out->ell = S.ell;
}

// ---------------------------------------------------------------- query
void TraceEngine::query(int64_t sid, const float* q, QueryOut* out) {
  out->reset();
  const size_t d = ex_.dim();
  if (cfg_.enabled && cfg_.check_norms) {  // the envelope needs ||q|| <= max_norm
    double n2 = 0;
    for (size_t j = 0; j < d; ++j) n2 += static_cast<double>(q[j]) * q[j];
    TRACE_CHECK(std::sqrt(n2) <= cfg_.max_norm, "query norm exceeds max_norm");
  }
  Session& S = session(sid);
  if (S.epoch != ex_.epoch()) {  // R3: index changed, residues invalid
    S.ring.clear();
    S.scale_hist.clear();
    S.sigma = 0.f;
    S.ell = 0;
    S.prev_answer.clear();
    S.epoch = ex_.epoch();
  }

  const uint64_t t0 = now_ns();
  const uint64_t c0 = thread_cpu_ns();
  SearchStats& st = out->st;

  SearchParams sp;
  sp.k = cfg_.k;
  sp.L = cfg_.L;
  sp.beam_width = cfg_.beam_width;
  sp.collect_visited = cfg_.enabled && cfg_.transfer;
  sp.want_vectors = cfg_.enabled && cfg_.transfer;
  uint64_t excl_wall = 0, excl_cpu = 0;  // untimed verification work
  // verify_prune: unpruned minus pruned seeded search (distances, wall time)
  bool np_valid = false;
  int64_t np_dist = 0, np_ns = 0;

  uint32_t lsh = 0;
  if (catapult_) lsh = catapult_->hash(q, &st);

  SearchResult res;
  vr_.clear();
  bool transferred = false;
  std::vector<const Residue*> ring_ptrs;
  std::fill(credit_.begin(), credit_.end(), 0.0);
  bounds_.credit = &credit_;

  // Runs a cold search (baseline entry, optionally CatapultDB entry points).
  auto run_cold = [&]() {
    std::vector<id_t> seeds;
    SearchParams p = sp;
    if (catapult_) {
      seeds = catapult_->seeds(lsh);
      p.add_entry_to_seeds = true;
    }
    out->n_seeds = static_cast<uint32_t>(seeds.size());
    ex_.search(q, seeds, p, nullptr, es_, &res, &vr_, &st);
    out->path = Path::Cold;
  };

  enum class Next { Cold, Transfer } next = Next::Cold;
  std::vector<float> deltas;

  if (!cfg_.enabled) {
    run_cold();
  } else if (!cfg_.transfer) {
    // Warm-seed baseline: previous answer as seeds, nothing else.
    if (S.prev_answer.empty()) {
      run_cold();
    } else {
      ex_.search(q, S.prev_answer, sp, nullptr, es_, &res, &vr_, &st);
      out->path = Path::Seeded;
    }
  } else {
    if (!S.ring.empty() && (!cfg_.jump || scale_ready(S))) {
      deltas.resize(S.ring.size());
      float mind = kInf;
      for (size_t i = 0; i < S.ring.size(); ++i) {
        deltas[i] = e_from_s(ex_.dist(S.ring[i]->anchor.data(), q));
        st.n_dist++;
        mind = std::min(mind, deltas[i]);
      }
      out->min_delta = mind;
      if (!cfg_.jump || !(mind > cfg_.tau * scale(S))) next = Next::Transfer;
    }
    if (next == Next::Cold) {
      run_cold();
    } else {
      for (const auto& r : S.ring) ring_ptrs.push_back(r.get());
      transfer_bounds(ring_ptrs, deltas, env_, &bounds_);
      transferred = true;
      out->R = static_cast<uint32_t>(bounds_.size());
      const Residue& newest = *S.ring.front();
      bool resolved = false;

      if (cfg_.resolve && !newest.ans.empty()) {
        // Rescore the pinned previous answer from cached vectors.
        const size_t na = newest.ans.size();
        std::vector<Neighbor> cands;
        cands.reserve(na + static_cast<size_t>(cfg_.s_max) + 1);
        for (size_t j = 0; j < na; ++j) {
          cands.push_back({ex_.dist(q, newest.ans_vecs.data() + j * d), newest.ans[j]});
          st.n_dist++;
        }
        std::vector<float> sa(na);
        for (size_t j = 0; j < na; ++j) sa[j] = cands[j].s;
        const size_t kk = std::min<size_t>(static_cast<size_t>(cfg_.k), na);
        std::nth_element(sa.begin(), sa.begin() + (kk - 1), sa.end());
        const float theta_k = sa[kk - 1];
        IdMap in_a(na + 16);
        for (id_t a : newest.ans) in_a.insert(a, 0);
        // Straddlers: retained nodes whose bound does not exclude them.
        std::vector<id_t> strad;
        std::vector<double> excl_credit(credit_.size(), 0.0);
        for (size_t slot = 0; slot < bounds_.size(); ++slot) {
          const id_t x = bounds_.ids[slot];
          if (in_a.contains(x)) continue;
          if (bounds_.lbS_at(slot) <= theta_k) {
            if (strad.size() <= static_cast<size_t>(cfg_.s_max)) strad.push_back(x);
          } else {
            excl_credit[bounds_.lb_arg[slot]] += 1.0;
          }
        }
        out->S = static_cast<uint32_t>(strad.size());
        bool budget_ok = S.ell < static_cast<uint32_t>(std::max(cfg_.L_max, 0));
        if (std::isfinite(cfg_.gamma_s))
          budget_ok = budget_ok && scale_ready(S) &&
                      (S.sigma + deltas[0] <= cfg_.gamma_s * scale(S));
        if (strad.size() <= static_cast<size_t>(cfg_.s_max) && budget_ok) {
          // Rescore straddlers (disk: page reads unless cached).
          std::vector<float> sv(strad.size() * d);
          ex_.fetch_vectors(strad.data(), strad.size(), sv.data(), &st);
          for (size_t j = 0; j < strad.size(); ++j) {
            cands.push_back({ex_.dist(q, sv.data() + j * d), strad[j]});
            st.n_dist++;
          }
          std::vector<uint32_t> order(cands.size());
          std::iota(order.begin(), order.end(), 0u);
          std::sort(order.begin(), order.end(),
                    [&](uint32_t a, uint32_t b) { return cands[a] < cands[b]; });
          const size_t kres = std::min(order.size(), static_cast<size_t>(cfg_.k));
          res.topk.clear();
          res.vectors.resize(kres * d);
          IdMap in_res(kres + 16);
          for (size_t i = 0; i < kres; ++i) {
            const uint32_t o = order[i];
            res.topk.push_back(cands[o]);
            in_res.insert(cands[o].id, 0);
            const float* src = (o < na) ? newest.ans_vecs.data() + o * d
                                        : sv.data() + (o - na) * d;
            std::memcpy(res.vectors.data() + i * d, src, d * sizeof(float));
          }
          // Separation margin mu (execution record).
          float min_lb = kInf, max_s = 0.f;
          for (size_t slot = 0; slot < bounds_.size(); ++slot)
            if (!in_res.contains(bounds_.ids[slot]))
              min_lb = std::min(min_lb, bounds_.lbS_at(slot));
          for (const Neighbor& nb : res.topk) max_s = std::max(max_s, nb.s);
          out->mu = min_lb - max_s;
          // Thin residue anchored at q: ids, distances, vectors of A'.
          auto thin = std::make_unique<Residue>();
          thin->anchor.assign(q, q + d);
          thin->epoch = ex_.epoch();
          thin->thin = true;
          for (const Neighbor& nb : res.topk) {
            thin->ids.push_back(nb.id);
            thin->d16.push_back(f32_to_f16(e_from_s(nb.s)));
            thin->ans.push_back(nb.id);
          }
          thin->ans_vecs = res.vectors;
          S.sigma += deltas[0];
          S.ell += 1;
          out->path = Path::Resolve;
          for (size_t i = 0; i < credit_.size(); ++i) credit_[i] += excl_credit[i];
          // utility update before the ring changes
          double tot = std::accumulate(credit_.begin(), credit_.end(), 0.0);
          if (tot > 0)
            for (size_t i = 0; i < S.ring.size(); ++i)
              S.ring[i]->utility = (1.0 - 1.0 / cfg_.w) * S.ring[i]->utility + credit_[i] / tot;
          push_residue(S, std::move(thin));
          resolved = true;
        }
      }

      if (!resolved) {
        // Seeds: the min(L, |R|) retained nodes with the smallest upper bound.
        std::vector<id_t> seeds;
        if (cfg_.seeds == TraceConfig::Seeds::Bound) {
          std::vector<uint32_t> idx(bounds_.size());
          std::iota(idx.begin(), idx.end(), 0u);
          const size_t ns = std::min(idx.size(), static_cast<size_t>(cfg_.L));
          auto by_ub = [&](uint32_t a, uint32_t b) {
            return bounds_.ub[a] < bounds_.ub[b] ||
                   (bounds_.ub[a] == bounds_.ub[b] && bounds_.ids[a] < bounds_.ids[b]);
          };
          if (ns < idx.size()) std::nth_element(idx.begin(), idx.begin() + ns, idx.end(), by_ub);
          idx.resize(ns);
          std::sort(idx.begin(), idx.end(), by_ub);
          for (uint32_t i : idx) {
            seeds.push_back(bounds_.ids[i]);
            credit_[bounds_.ub_arg[i]] += 1.0;
          }
        } else {
          seeds = newest.ans;
        }
        SearchParams p = sp;
        if (catapult_ && seeds.size() < static_cast<size_t>(cfg_.L)) {
          IdMap have(seeds.size() + 64);
          for (id_t s0 : seeds) have.insert(s0, 0);
          for (id_t c : catapult_->seeds(lsh))
            if (have.insert(c, 0).second) seeds.push_back(c);
        }
        const BoundProvider* pr = cfg_.prune ? &bounds_ : nullptr;
        const bool verify = cfg_.verify_prune && pr != nullptr;
        out->n_seeds = static_cast<uint32_t>(seeds.size());
        const uint64_t dist_before = st.n_dist;
        const uint64_t ts = verify ? now_ns() : 0;
        ex_.search(q, seeds, p, pr, es_, &res, &vr_, &st);
        const uint64_t pruned_ns = verify ? now_ns() - ts : 0;
        out->path = Path::Seeded;

        if (verify) {
          // Theorem 2 check (untimed): same seeds, no pruning.
          const uint64_t tv = now_ns(), cv = thread_cpu_ns();
          SearchResult ref;
          SearchStats tmp;
          SearchParams p2 = p;
          p2.collect_visited = false;
          p2.want_vectors = false;
          ex_.search(q, seeds, p2, nullptr, es_, &ref, nullptr, &tmp);
          const uint64_t unpruned_ns = now_ns() - tv;
          np_valid = true;
          np_dist = static_cast<int64_t>(tmp.n_dist) -
                    static_cast<int64_t>(st.n_dist - dist_before);
          np_ns = static_cast<int64_t>(unpruned_ns) - static_cast<int64_t>(pruned_ns);
          bool eq = ref.topk.size() == res.topk.size();
          for (size_t i = 0; eq && i < ref.topk.size(); ++i)
            eq = ref.topk[i].id == res.topk[i].id && ref.topk[i].s == res.topk[i].s;
          out->prune_equiv = eq ? 1 : 0;
          excl_wall += now_ns() - tv;
          excl_cpu += thread_cpu_ns() - cv;
        }
      }
    }
  }

  // Publish a full residue after any real traversal (cold or seeded).
  if (out->path != Path::Resolve) {
    if (cfg_.enabled && cfg_.transfer) {
      if (transferred) {
        double tot = std::accumulate(credit_.begin(), credit_.end(), 0.0);
        if (tot > 0)
          for (size_t i = 0; i < S.ring.size(); ++i)
            S.ring[i]->utility = (1.0 - 1.0 / cfg_.w) * S.ring[i]->utility + credit_[i] / tot;
      }
      push_residue(S, make_full_residue(q, vr_, res));
    }
    if (cfg_.enabled) record_scale(S, res.topk);
    S.sigma = 0.f;
    S.ell = 0;
  }
  S.prev_answer.clear();
  for (const Neighbor& nb : res.topk) S.prev_answer.push_back(nb.id);
  if (catapult_ && !res.topk.empty()) catapult_->update(lsh, res.topk.front().id);
  finish(S, q, res.topk, out);

  out->wall_ns = now_ns() - t0 - excl_wall;
  out->cpu_ns = thread_cpu_ns() - c0 - excl_cpu;
  if (np_valid) {
    out->noprune_n_dist = static_cast<uint64_t>(static_cast<int64_t>(st.n_dist) + np_dist);
    out->noprune_wall_ns = static_cast<uint64_t>(
        std::max<int64_t>(0, static_cast<int64_t>(out->wall_ns) + np_ns));
  }

  // ------------------------------------------------ audits (untimed)
  if (cfg_.audit && transferred) {
    std::vector<float> v(d);
    SearchStats untimed;
    std::vector<Neighbor> all;
    all.reserve(bounds_.size());
    for (size_t slot = 0; slot < bounds_.size(); ++slot) {
      const id_t x = bounds_.ids[slot];
      ex_.fetch_vectors(&x, 1, v.data(), &untimed);
      const float sx = ex_.dist(q, v.data());
      out->audit_checked++;
      if (bounds_.lbS_at(slot) > sx) out->audit_viol++;
      all.push_back({sx, x});
      out->retained.push_back(ex_.label(x));
    }
    if (out->path == Path::Resolve) {
      std::sort(all.begin(), all.end());
      const size_t kk = std::min(all.size(), static_cast<size_t>(cfg_.k));
      bool ok = res.topk.size() == kk;
      // Set equality under s; equal distances may order differently only if
      // they tie exactly, in which case (s, id) order is used by both.
      for (size_t i = 0; ok && i < kk; ++i)
        ok = all[i].id == res.topk[i].id && all[i].s == res.topk[i].s;
      out->resolve_exact = ok ? 1 : 0;
    }
  }
}

}  // namespace trace
