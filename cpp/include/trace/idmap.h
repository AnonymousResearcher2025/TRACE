// Open-addressing map from node id to a dense slot index, cleared in O(1)
// with generation stamps. Used for the retained set R during transfer and as
// a visited set on the disk tier.
#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "trace/common.h"

namespace trace {

class IdMap {
 public:
  explicit IdMap(size_t initial_capacity = 1024) { reserve(initial_capacity); }

  // Removes all entries; keeps capacity. Amortized O(1).
  void clear() {
    size_ = 0;
    if (++gen_ == 0) {  // stamp wrap-around: hard reset
      std::fill(stamp_.begin(), stamp_.end(), 0u);
      gen_ = 1;
    }
  }

  // Ensures room for n entries at load factor <= 0.5 (rehashes if needed).
  void reserve(size_t n) {
    size_t cap = 16;
    while (cap < 2 * n) cap <<= 1;
    if (cap <= keys_.size()) return;
    rehash(cap);
  }

  size_t size() const { return size_; }

  // Returns the slot value for id, or kInvalidId if absent.
  uint32_t find(id_t id) const {
    size_t h = hash(id) & mask_;
    while (true) {
      if (stamp_[h] != gen_) return kInvalidId;
      if (keys_[h] == id) return vals_[h];
      h = (h + 1) & mask_;
    }
  }

  // Inserts (id -> val) if absent. Returns {existing_or_new_val, inserted}.
  std::pair<uint32_t, bool> insert(id_t id, uint32_t val) {
    if (2 * (size_ + 1) > keys_.size()) rehash(keys_.size() * 2);
    size_t h = hash(id) & mask_;
    while (true) {
      if (stamp_[h] != gen_) {
        stamp_[h] = gen_;
        keys_[h] = id;
        vals_[h] = val;
        ++size_;
        return {val, true};
      }
      if (keys_[h] == id) return {vals_[h], false};
      h = (h + 1) & mask_;
    }
  }

  bool contains(id_t id) const { return find(id) != kInvalidId; }

 private:
  static inline size_t hash(id_t id) {
    uint64_t x = static_cast<uint64_t>(id) * 0x9E3779B97F4A7C15ull;
    return static_cast<size_t>(x >> 32);
  }

  void rehash(size_t new_cap) {
    std::vector<id_t> ok;
    std::vector<uint32_t> ov;
    ok.reserve(size_);
    ov.reserve(size_);
    for (size_t i = 0; i < keys_.size(); ++i) {
      if (stamp_[i] == gen_) {
        ok.push_back(keys_[i]);
        ov.push_back(vals_[i]);
      }
    }
    keys_.assign(new_cap, 0);
    vals_.assign(new_cap, 0);
    stamp_.assign(new_cap, 0);
    mask_ = new_cap - 1;
    gen_ = 1;
    size_ = 0;
    for (size_t i = 0; i < ok.size(); ++i) insert(ok[i], ov[i]);
  }

  std::vector<id_t> keys_;
  std::vector<uint32_t> vals_;
  std::vector<uint32_t> stamp_;
  size_t mask_ = 0;
  size_t size_ = 0;
  uint32_t gen_ = 1;
};

}  // namespace trace
