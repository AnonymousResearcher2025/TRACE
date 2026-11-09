// Python bindings (module trace_harness._trace) and the replay driver.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "trace/baselines.h"
#include "trace/disk_executor.h"
#include "trace/engine.h"
#include "trace/fp.h"
#include "trace/hnsw_executor.h"
#include "trace/vamana.h"

namespace py = pybind11;
using namespace trace;

namespace {

using F32 = py::array_t<float, py::array::c_style | py::array::forcecast>;
using I64 = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;

struct Arrays {
  size_t n = 0, k = 0;
  std::vector<int64_t> labels;
  std::vector<float> s;
  std::vector<int8_t> path, resolve_exact, prune_equiv, early_stopped;
  std::vector<int64_t> n_dist, n_pq, n_aux, n_reads, n_cache_hits, n_pruned,
      n_hops, io_ns, wall_ns, cpu_ns, noprune_n_dist, noprune_wall_ns;
  std::vector<int32_t> R, S, ell, audit_checked, audit_viol, n_seeds;
  std::vector<float> min_delta, sigma, mu;
  std::vector<std::vector<id_t>> retained;

  void init(size_t n_, size_t k_, bool keep_retained) {
    n = n_;
    k = k_;
    labels.assign(n * k, -1);
    s.assign(n * k, std::numeric_limits<float>::quiet_NaN());
    for (auto* v : {&path, &resolve_exact, &prune_equiv, &early_stopped}) v->assign(n, 0);
    for (auto* v : {&n_dist, &n_pq, &n_aux, &n_reads, &n_cache_hits, &n_pruned,
                    &n_hops, &io_ns, &wall_ns, &cpu_ns, &noprune_n_dist, &noprune_wall_ns})
      v->assign(n, 0);
    for (auto* v : {&R, &S, &ell, &audit_checked, &audit_viol, &n_seeds}) v->assign(n, 0);
    for (auto* v : {&min_delta, &sigma, &mu}) v->assign(n, 0.f);
    if (keep_retained) retained.assign(n, {});
  }

  void put(size_t i, QueryOut& o) {
    for (size_t j = 0; j < o.labels.size() && j < k; ++j) {
      labels[i * k + j] = static_cast<int64_t>(o.labels[j]);
      s[i * k + j] = o.s[j];
    }
    path[i] = static_cast<int8_t>(o.path);
    resolve_exact[i] = o.resolve_exact;
    prune_equiv[i] = o.prune_equiv;
    early_stopped[i] = o.st.early_stopped ? 1 : 0;
    n_dist[i] = static_cast<int64_t>(o.st.n_dist);
    n_pq[i] = static_cast<int64_t>(o.st.n_pq);
    n_aux[i] = static_cast<int64_t>(o.st.n_aux);
    n_reads[i] = static_cast<int64_t>(o.st.n_reads);
    n_cache_hits[i] = static_cast<int64_t>(o.st.n_cache_hits);
    n_pruned[i] = static_cast<int64_t>(o.st.n_pruned);
    n_hops[i] = static_cast<int64_t>(o.st.n_hops);
    io_ns[i] = static_cast<int64_t>(o.st.io_ns);
    wall_ns[i] = static_cast<int64_t>(o.wall_ns);
    cpu_ns[i] = static_cast<int64_t>(o.cpu_ns);
    noprune_n_dist[i] = static_cast<int64_t>(o.noprune_n_dist);
    noprune_wall_ns[i] = static_cast<int64_t>(o.noprune_wall_ns);
    R[i] = static_cast<int32_t>(o.R);
    n_seeds[i] = static_cast<int32_t>(o.n_seeds);
    S[i] = static_cast<int32_t>(o.S);
    ell[i] = static_cast<int32_t>(o.ell);
    audit_checked[i] = static_cast<int32_t>(o.audit_checked);
    audit_viol[i] = static_cast<int32_t>(o.audit_viol);
    min_delta[i] = o.min_delta;
    sigma[i] = o.sigma;
    mu[i] = o.mu;
    if (!retained.empty()) retained[i] = std::move(o.retained);
  }

  py::dict to_dict() const {
    py::dict d;
    auto arr2 = [&](const auto& v, size_t cols) {
      using T = typename std::decay_t<decltype(v)>::value_type;
      py::array_t<T> a({n, cols});
      std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(T));
      return a;
    };
    auto arr1 = [&](const auto& v) {
      using T = typename std::decay_t<decltype(v)>::value_type;
      py::array_t<T> a(n);
      std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(T));
      return a;
    };
    d["labels"] = arr2(labels, k);
    d["s"] = arr2(s, k);
    d["path"] = arr1(path);
    d["resolve_exact"] = arr1(resolve_exact);
    d["prune_equiv"] = arr1(prune_equiv);
    d["early_stopped"] = arr1(early_stopped);
    d["n_dist"] = arr1(n_dist);
    d["n_pq"] = arr1(n_pq);
    d["n_aux"] = arr1(n_aux);
    d["n_reads"] = arr1(n_reads);
    d["n_cache_hits"] = arr1(n_cache_hits);
    d["n_pruned"] = arr1(n_pruned);
    d["n_hops"] = arr1(n_hops);
    d["io_ns"] = arr1(io_ns);
    d["wall_ns"] = arr1(wall_ns);
    d["cpu_ns"] = arr1(cpu_ns);
    d["noprune_n_dist"] = arr1(noprune_n_dist);
    d["noprune_wall_ns"] = arr1(noprune_wall_ns);
    d["R"] = arr1(R);
    d["n_seeds"] = arr1(n_seeds);
    d["S"] = arr1(S);
    d["ell"] = arr1(ell);
    d["audit_checked"] = arr1(audit_checked);
    d["audit_viol"] = arr1(audit_viol);
    d["min_delta"] = arr1(min_delta);
    d["sigma"] = arr1(sigma);
    d["mu"] = arr1(mu);
    if (!retained.empty()) {
      py::list L;
      for (const auto& r : retained) {
        py::array_t<uint32_t> a(r.size());
        if (!r.empty()) std::memcpy(a.mutable_data(), r.data(), r.size() * 4);
        L.append(a);
      }
      d["retained"] = L;
    }
    return d;
  }
};

// Runs queries idx (in order) on one engine. Sessions are ended when the
// next query of this sequence belongs to another session (contiguous
// sessions); pass end_sessions=false for interleaved streams.
void run_seq(Engine& eng, const float* Q, const int64_t* sess, size_t d,
             const std::vector<size_t>& idx, bool end_sessions, LogStop* log,
             Arrays* A) {
  QueryOut o;
  for (size_t t = 0; t < idx.size(); ++t) {
    const size_t i = idx[t];
    if (log) log->set_query(static_cast<int64_t>(i));
    eng.query(sess[i], Q + i * d, &o);
    A->put(i, o);
    if (end_sessions && (t + 1 == idx.size() || sess[idx[t + 1]] != sess[i]))
      eng.end_session(sess[i]);
  }
}

py::dict replay(Engine& eng, F32 Q, I64 sessions, int k, bool end_sessions,
                bool keep_retained, LogStop* log) {
  TRACE_CHECK(Q.ndim() == 2, "Q must be 2-D");
  const size_t n = static_cast<size_t>(Q.shape(0)), d = static_cast<size_t>(Q.shape(1));
  TRACE_CHECK(d == eng.executor().dim(), "query dimension mismatch");
  TRACE_CHECK(static_cast<size_t>(sessions.size()) == n, "sessions length mismatch");
  TRACE_CHECK(k >= 1, "k must be >= 1");
  Arrays A;
  A.init(n, static_cast<size_t>(k), keep_retained);
  std::vector<size_t> idx(n);
  for (size_t i = 0; i < n; ++i) idx[i] = i;
  uint64_t t0;
  {
    py::gil_scoped_release nogil;
    t0 = now_ns();
    run_seq(eng, Q.data(), sessions.data(), d, idx, end_sessions, log, &A);
    t0 = now_ns() - t0;
  }
  py::dict out = A.to_dict();
  out["total_wall_ns"] = t0;
  return out;
}

// One engine per thread; queries of a session stay on one thread (session
// id modulo threads), in stream order.
py::dict replay_parallel(std::vector<Engine*> engines, F32 Q, I64 sessions,
                         int k, bool end_sessions) {
  const size_t T = engines.size();
  TRACE_CHECK(T >= 1, "need at least one engine");
  const size_t n = static_cast<size_t>(Q.shape(0)), d = static_cast<size_t>(Q.shape(1));
  TRACE_CHECK(static_cast<size_t>(sessions.size()) == n, "sessions length mismatch");
  std::vector<std::vector<size_t>> parts(T);
  const int64_t* sp = sessions.data();
  for (size_t i = 0; i < n; ++i) {
    const int64_t sid = sp[i];
    const size_t t = static_cast<size_t>(((sid % static_cast<int64_t>(T)) + static_cast<int64_t>(T)) %
                                         static_cast<int64_t>(T));
    parts[t].push_back(i);
  }
  TRACE_CHECK(k >= 1, "k must be >= 1");
  Arrays A;
  A.init(n, static_cast<size_t>(k), false);
  uint64_t wall;
  {
    py::gil_scoped_release nogil;
    std::vector<std::thread> th;
    std::vector<std::string> errs(T);
    const uint64_t t0 = now_ns();
    for (size_t t = 0; t < T; ++t) {
      th.emplace_back([&, t]() {
        try {
          run_seq(*engines[t], Q.data(), sp, d, parts[t], end_sessions, nullptr, &A);
        } catch (const std::exception& e) {
          errs[t] = e.what();
        }
      });
    }
    for (auto& x : th) x.join();
    wall = now_ns() - t0;
    for (const auto& e : errs) TRACE_CHECK(e.empty(), e);
  }
  py::dict out = A.to_dict();
  out["total_wall_ns"] = wall;
  return out;
}

py::dict query_one(Engine& eng, int64_t session, F32 q) {
  QueryOut o;
  eng.query(session, q.data(), &o);
  Arrays A;
  A.init(1, std::max<size_t>(o.labels.size(), 1), true);
  A.put(0, o);
  return A.to_dict();
}

}  // namespace

PYBIND11_MODULE(_trace, m) {
  m.doc() = "TRACE runtime (C++ core)";

  // ------------------------------------------------------------ executors
  py::class_<Executor>(m, "Executor")
      .def_property_readonly("dim", &Executor::dim)
      .def_property_readonly("size", &Executor::size)
      .def_property_readonly("tier", &Executor::tier)
      .def_property_readonly("entry_point", &Executor::entry_point)
      .def_property_readonly("epoch", &Executor::epoch)
      .def("bump_epoch", &Executor::bump_epoch)
      .def("label", &Executor::label)
      .def("dist", [](const Executor& e, F32 a, F32 b) {
        return e.dist(a.data(), b.data());
      })
      .def("search", [](const Executor& e, F32 q, std::vector<id_t> seeds, int k,
                        int L, int beam_width) {
        SearchParams p;
        p.k = k;
        p.L = L;
        p.beam_width = beam_width;
        p.collect_visited = false;
        SearchResult r;
        SearchStats st;
        e.search(q.data(), seeds, p, nullptr, nullptr, &r, nullptr, &st);
        std::vector<id_t> lab, ids;
        std::vector<float> s;
        for (auto& nb : r.topk) {
          lab.push_back(e.label(nb.id));
          ids.push_back(nb.id);
          s.push_back(nb.s);
        }
        py::dict d;
        d["labels"] = lab;
        d["ids"] = ids;
        d["s"] = s;
        d["n_dist"] = st.n_dist;
        d["n_reads"] = st.n_reads;
        d["n_pq"] = st.n_pq;
        return d;
      }, py::arg("q"), py::arg("seeds") = std::vector<id_t>{}, py::arg("k") = 10,
         py::arg("L") = 100, py::arg("beam_width") = 4)
      .def("labels", [](const Executor& e,
                        py::array_t<int64_t, py::array::c_style | py::array::forcecast> ids) {
        py::array_t<int64_t> out(ids.request().shape);
        for (py::ssize_t i = 0; i < ids.size(); ++i) {
          const int64_t v = ids.data()[i];
          out.mutable_data()[i] = v < 0 ? -1 : static_cast<int64_t>(e.label(static_cast<id_t>(v)));
        }
        return out;
      })
      .def("fetch_vectors", [](const Executor& e, std::vector<id_t> ids) {
        py::array_t<float> out({ids.size(), e.dim()});
        SearchStats st;
        e.fetch_vectors(ids.data(), ids.size(), out.mutable_data(), &st);
        return out;
      });

  py::class_<HnswExecutor, Executor>(m, "HnswExecutor")
      .def(py::init<const std::string&, size_t>(), py::arg("index_path"), py::arg("dim"))
      .def("internal_of", &HnswExecutor::internal_of);

  py::class_<VamanaExecutor, Executor>(m, "VamanaExecutor")
      .def(py::init<const std::string&, const std::string&>(), py::arg("graph_path"),
           py::arg("vectors_path"))
      .def_property_readonly("max_degree", [](const VamanaExecutor& e) {
        size_t m = 0;
        for (size_t i = 0; i < e.graph().size(); ++i)
          m = std::max(m, e.graph().neighbors(static_cast<id_t>(i)).size());
        return m;
      })
      .def("neighbors", [](const VamanaExecutor& e, id_t i) {
        TRACE_CHECK(i < e.graph().size(), "id out of range");
        return e.graph().neighbors(i);
      });

  // Builds a Vamana graph by FreshVamana insertion (row order) and saves it in
  // DiskANN's in-memory graph format.
  m.def("build_vamana", [](F32 X, const std::string& graph_path, uint32_t R,
                           uint32_t L_build, float alpha) {
    TRACE_CHECK(X.ndim() == 2, "X must be 2-D");
    VamanaParams p;
    p.R = R;
    p.L_build = L_build;
    p.alpha = alpha;
    p.maxc = std::max<uint32_t>(p.maxc, R);
    const size_t n = static_cast<size_t>(X.shape(0)), d = static_cast<size_t>(X.shape(1));
    VamanaGraph g(d, n, p);
    {
      py::gil_scoped_release nogil;
      for (size_t i = 0; i < n; ++i) g.insert(X.data() + i * d);
    }
    g.save_diskann(graph_path);
  }, py::arg("X"), py::arg("graph_path"), py::arg("R") = 64, py::arg("L_build") = 100,
     py::arg("alpha") = 1.2f);

  py::class_<DiskExecutor, Executor>(m, "DiskExecutor")
      .def(py::init<const std::string&, bool, size_t, int>(), py::arg("prefix"),
           py::arg("direct_io") = true, py::arg("cache_nodes") = 16384,
           py::arg("max_beam_width") = 16)
      .def_property_readonly("max_degree", &DiskExecutor::max_degree)
      .def_property_readonly("num_pq_chunks", &DiskExecutor::num_pq_chunks)
      .def_property_readonly("cached_nodes", &DiskExecutor::cached_nodes);

  // ------------------------------------------------------------ configs
  py::class_<TraceConfig> tc(m, "TraceConfig");
  py::enum_<TraceConfig::Retention>(tc, "Retention")
      .value("Recency", TraceConfig::Retention::Recency)
      .value("Hybrid", TraceConfig::Retention::Hybrid);
  py::enum_<TraceConfig::Seeds>(tc, "Seeds")
      .value("Bound", TraceConfig::Seeds::Bound)
      .value("PrevAnswer", TraceConfig::Seeds::PrevAnswer);
  tc.def(py::init<>())
      .def_readwrite("enabled", &TraceConfig::enabled)
      .def_readwrite("k", &TraceConfig::k)
      .def_readwrite("L", &TraceConfig::L)
      .def_readwrite("beam_width", &TraceConfig::beam_width)
      .def_readwrite("m", &TraceConfig::m)
      .def_readwrite("tau", &TraceConfig::tau)
      .def_readwrite("w", &TraceConfig::w)
      .def_readwrite("s_max", &TraceConfig::s_max)
      .def_readwrite("gamma_s", &TraceConfig::gamma_s)
      .def_readwrite("L_max", &TraceConfig::L_max)
      .def_readwrite("B_r", &TraceConfig::B_r)
      .def_readwrite("retention", &TraceConfig::retention)
      .def_readwrite("seeds", &TraceConfig::seeds)
      .def_readwrite("transfer", &TraceConfig::transfer)
      .def_readwrite("prune", &TraceConfig::prune)
      .def_readwrite("resolve", &TraceConfig::resolve)
      .def_readwrite("jump", &TraceConfig::jump)
      .def_readwrite("share_ring", &TraceConfig::share_ring)
      .def_readwrite("max_norm", &TraceConfig::max_norm)
      .def_readwrite("check_norms", &TraceConfig::check_norms)
      .def_readwrite("audit", &TraceConfig::audit)
      .def_readwrite("verify_prune", &TraceConfig::verify_prune);

  py::class_<DCacheConfig>(m, "DCacheConfig")
      .def(py::init<>())
      .def_readwrite("k", &DCacheConfig::k)
      .def_readwrite("L", &DCacheConfig::L)
      .def_readwrite("beam_width", &DCacheConfig::beam_width)
      .def_readwrite("pivots", &DCacheConfig::pivots)
      .def_readwrite("budget_bytes", &DCacheConfig::budget_bytes)
      .def_readwrite("max_norm", &DCacheConfig::max_norm);

  py::class_<SnakeConfig>(m, "SnakeConfig")
      .def(py::init<>())
      .def_readwrite("k", &SnakeConfig::k)
      .def_readwrite("L", &SnakeConfig::L)
      .def_readwrite("beam_width", &SnakeConfig::beam_width)
      .def_readwrite("slots", &SnakeConfig::slots)
      .def_readwrite("max_norm", &SnakeConfig::max_norm);

  py::enum_<QVStrategy>(m, "QVStrategy")
      .value("Adaptive", QVStrategy::Adaptive)
      .value("Eager", QVStrategy::Eager)
      .value("Exhaustive", QVStrategy::Exhaustive);

  py::class_<QVCacheConfig>(m, "QVCacheConfig")
      .def(py::init<>())
      .def_readwrite("k", &QVCacheConfig::k)
      .def_readwrite("L", &QVCacheConfig::L)
      .def_readwrite("beam_width", &QVCacheConfig::beam_width)
      .def_readwrite("n_mini", &QVCacheConfig::n_mini)
      .def_readwrite("mini_capacity", &QVCacheConfig::mini_capacity)
      .def_readwrite("mini_R", &QVCacheConfig::mini_R)
      .def_readwrite("mini_L_build", &QVCacheConfig::mini_L_build)
      .def_readwrite("mini_L_search", &QVCacheConfig::mini_L_search)
      .def_readwrite("mini_alpha", &QVCacheConfig::mini_alpha)
      .def_readwrite("deviation", &QVCacheConfig::deviation)
      .def_readwrite("alpha", &QVCacheConfig::alpha)
      .def_readwrite("strategy", &QVCacheConfig::strategy)
      .def_readwrite("eager_hit_ratio", &QVCacheConfig::eager_hit_ratio)
      .def_readwrite("exclude_insert_time", &QVCacheConfig::exclude_insert_time)
      .def_readwrite("region_model", &QVCacheConfig::region_model);

  // ------------------------------------------------------- stop policies
  py::class_<EarlyStop>(m, "EarlyStop");
  py::class_<DarthStop, EarlyStop>(m, "DarthStop")
      .def(py::init<const std::string&, double, uint64_t>(), py::arg("model_path"),
           py::arg("target_recall"), py::arg("interval"));
  py::class_<LogStop, EarlyStop>(m, "LogStop")
      .def(py::init<uint64_t>(), py::arg("interval"))
      .def_readwrite("k", &LogStop::k)
      .def("clear", &LogStop::clear)
      .def("features", [](const LogStop& l) {
        const size_t n = l.feats.size() / kDarthFeatures;
        py::array_t<float> a({n, static_cast<size_t>(kDarthFeatures)});
        std::memcpy(a.mutable_data(), l.feats.data(), l.feats.size() * 4);
        return a;
      })
      .def("topk", [](const LogStop& l) {
        const size_t n = l.query.size();
        py::array_t<int64_t> a({n, static_cast<size_t>(l.k)});
        std::memcpy(a.mutable_data(), l.topk.data(), l.topk.size() * 8);
        return a;
      })
      .def("query_index", [](const LogStop& l) {
        py::array_t<int64_t> a(l.query.size());
        std::memcpy(a.mutable_data(), l.query.data(), l.query.size() * 8);
        return a;
      });
  m.attr("DARTH_NUM_FEATURES") = kDarthFeatures;

  py::class_<Catapult>(m, "Catapult")
      .def(py::init<size_t, int, int, uint64_t>(), py::arg("dim"),
           py::arg("n_bits") = 8, py::arg("bucket_cap") = 40, py::arg("seed") = 42)
      .def("reset", &Catapult::reset)
      .def("bytes", &Catapult::bytes);

  // -------------------------------------------------------------- engines
  py::class_<Engine>(m, "Engine")
      .def("reset", &Engine::reset)
      .def("end_session", &Engine::end_session)
      .def("state_bytes", &Engine::state_bytes)
      .def("query", &query_one, py::arg("session"), py::arg("q"));

  py::class_<TraceEngine, Engine>(m, "TraceEngine")
      .def(py::init<const Executor&, const TraceConfig&, Catapult*, const EarlyStop*>(),
           py::arg("executor"), py::arg("config"), py::arg("catapult") = nullptr,
           py::arg("early_stop") = nullptr, py::keep_alive<1, 2>(),
           py::keep_alive<1, 4>(), py::keep_alive<1, 5>());
  py::class_<DCacheEngine, Engine>(m, "DCacheEngine")
      .def(py::init<const Executor&, const DCacheConfig&>(), py::arg("executor"),
           py::arg("config"), py::keep_alive<1, 2>());
  py::class_<SnakeEngine, Engine>(m, "SnakeEngine")
      .def(py::init<const Executor&, const SnakeConfig&>(), py::arg("executor"),
           py::arg("config"), py::keep_alive<1, 2>());
  py::class_<QVCacheEngine, Engine>(m, "QVCacheEngine")
      .def(py::init<const Executor&, const QVCacheConfig&>(), py::arg("executor"),
           py::arg("config"), py::keep_alive<1, 2>())
      .def("region_of", [](const QVCacheEngine& e, F32 q) { return e.region_of(q.data()); })
      .def("theta_of", [](const QVCacheEngine& e, F32 q) { return e.theta_of(q.data()); })
      .def_property_readonly("cached_vectors", &QVCacheEngine::cached_vectors)
      .def("mini_size", &QVCacheEngine::mini_size)
      .def_property_readonly("order", &QVCacheEngine::order)
      .def_property_readonly("hits", &QVCacheEngine::hits)
      .def_property_readonly("lookups", &QVCacheEngine::lookups);

  // --------------------------------------------------------------- replay
  m.def("replay", &replay, py::arg("engine"), py::arg("Q"), py::arg("sessions"),
        py::arg("k") = 10, py::arg("end_sessions") = true,
        py::arg("keep_retained") = false, py::arg("log") = nullptr);
  m.def("replay_parallel", &replay_parallel, py::arg("engines"), py::arg("Q"),
        py::arg("sessions"), py::arg("k") = 10, py::arg("end_sessions") = true);
}
