// Finite-precision helpers: float16 storage, outward rounding, and the
// envelope that makes transferred bounds sound for the executor's own float32
// comparisons (Sec. III-B: Eqs. (3) and (4), Lemma 1).
//
// This header must be compiled WITHOUT -ffast-math and with
// -ffp-contract=off (see CMakeLists.txt). Outward rounding relies on IEEE-754
// binary32 round-to-nearest-even arithmetic.
#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "trace/common.h"

namespace trace {

static_assert(std::numeric_limits<float>::is_iec559, "IEEE-754 float required");

// ---------------------------------------------------------------- float16
// float32 -> float16 with round-to-nearest-even (F. Giesen's
// float_to_half_fast3_rtne), matching IEEE-754 conversion.
inline uint16_t f32_to_f16(float f) {
  const uint32_t f32infty = 255u << 23;
  const uint32_t f16max = (127u + 16u) << 23;
  const uint32_t denorm_magic_u = ((127u - 15u) + (23u - 10u) + 1u) << 23;
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = x & 0x80000000u;
  x ^= sign;
  uint16_t o;
  if (x >= f16max) {
    o = (x > f32infty) ? 0x7e00 : 0x7c00;  // NaN or Inf
  } else if (x < (113u << 23)) {
    // Subnormal or zero: align mantissa with a magic addition (RNE add).
    float fx, magic;
    std::memcpy(&fx, &x, 4);
    std::memcpy(&magic, &denorm_magic_u, 4);
    volatile float sum = fx + magic;  // volatile: keep the rounding add
    float sv = sum;
    uint32_t xs;
    std::memcpy(&xs, &sv, 4);
    o = static_cast<uint16_t>(xs - denorm_magic_u);
  } else {
    const uint32_t mant_odd = (x >> 13) & 1u;
    x += 0xC8000FFFu;  // ((15 - 127) << 23) + 0xfff, modulo 2^32
    x += mant_odd;
    o = static_cast<uint16_t>(x >> 13);
  }
  return static_cast<uint16_t>(o | (sign >> 16));
}

// float16 -> float32 (exact).
inline float f16_to_f32(uint16_t h) {
  const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x3FFu;
  uint32_t out;
  if (exp == 0) {
    if (mant == 0) {
      out = sign;
    } else {
      exp = 127 - 15 + 1;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x3FFu;
      out = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    out = sign | 0x7F800000u | (mant << 13);
  } else {
    out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &out, 4);
  return f;
}

// ------------------------------------------------------ outward rounding
// a (+) b : fl32(a+b) moved one representable value toward +inf.
// a (-) b : fl32(a-b) moved one representable value toward -inf.
// a (x) b : fl32(a*b) moved one representable value toward -inf.
// Bit-level nextafter toward +inf / -inf: the same result as std::nextafter,
// but inlined, since transfer applies it to every retained entry.
inline float next_up(float x) {
  uint32_t b;
  std::memcpy(&b, &x, 4);
  if ((b & 0x7F800000u) == 0x7F800000u) return std::nextafter(x, kInf);  // inf/nan
  if ((b & 0x7FFFFFFFu) == 0) {
    b = 1u;  // +/-0 -> smallest positive subnormal
  } else if ((b & 0x80000000u) == 0) {
    b += 1u;
  } else {
    b -= 1u;
  }
  std::memcpy(&x, &b, 4);
  return x;
}
inline float next_down(float x) {
  uint32_t b;
  std::memcpy(&b, &x, 4);
  if ((b & 0x7F800000u) == 0x7F800000u) return std::nextafter(x, -kInf);
  if ((b & 0x7FFFFFFFu) == 0) {
    b = 0x80000001u;  // +/-0 -> smallest negative subnormal
  } else if ((b & 0x80000000u) == 0) {
    b -= 1u;
  } else {
    b += 1u;
  }
  std::memcpy(&x, &b, 4);
  return x;
}
inline float up_add(float a, float b) { return next_up(a + b); }
inline float dn_sub(float a, float b) { return next_down(a - b); }
inline float dn_mul(float a, float b) { return next_down(a * b); }

// ----------------------------------------------------------- envelope
// eta  >= |e(u,v) - ||u-v|||, where e = fl32(sqrt(s)) and s is the float32
//         squared-L2 kernel over d terms (any summation order, FMA or not).
// eps+ = (storage (+) eta) (+) eta, the outward-rounded widening of Eq. (4).
struct Envelope {
  float eta = 0.f;
  float storage = 0.f;
  float eps_plus = 0.f;
  float one_minus_2u = 1.f;
  float max_norm = 1.f;

  // dim: vector dimension; max_norm: upper bound on the L2 norm of every
  // corpus vector and every query/anchor; fp16_storage: retained distances
  // are stored as float16 (TRACE) or float32 (D-Cache, Snake Table).
  static Envelope make(size_t dim, double max_norm_bound, bool fp16_storage) {
    TRACE_CHECK(max_norm_bound > 0 && max_norm_bound < 1.9,
                "max_norm must be < 1.9 so stored distances stay below 4");
    const double u = std::ldexp(1.0, -24);
    const double j = static_cast<double>(dim) + 2.0;
    TRACE_CHECK(j * u < 0.01, "dimension too large for the envelope");
    const double gamma = (j * u) / (1.0 - j * u);
    const double dmax = 2.0 * max_norm_bound;
    // s = d^2 (1 + theta) with |theta| <= gamma_{d+2} (nonnegative terms, any
    // summation order); e = sqrt(s)(1 + zeta), |zeta| <= u. Since
    // |sqrt(1+theta) - 1| <= (|theta|/2)(1 + |theta|) for |theta| <= 1,
    //   |e - d| <= d (1 + gamma) (gamma/2 + u) <= dmax (1 + gamma)(gamma/2 + u),
    // which is gamma_{d+2} + 2u to first order (the bound on eta_d in Sec. III-B).
    const double eta = dmax * (1.0 + gamma) * (0.5 * gamma + u);
    Envelope e;
    e.eta = round_up(eta);
    // float16 RNE error is at most half a spacing; spacing is 2^-10 in [1,2)
    // and 2^-9 in [2,4), so the error is at most 2^-10 for values below 4.
    e.storage = fp16_storage ? std::ldexp(1.0f, -10) : 0.0f;
    e.eps_plus = up_add(up_add(e.storage, e.eta), e.eta);
    e.one_minus_2u = 1.0f - std::ldexp(1.0f, -23);  // exact in float32
    e.max_norm = static_cast<float>(max_norm_bound);
    return e;
  }

 private:
  static float round_up(double v) {
    float f = static_cast<float>(v);
    if (static_cast<double>(f) < v) f = std::nextafter(f, kInf);
    return std::nextafter(f, kInf);  // one extra step of margin
  }
};

// Lower bound on ||q - x|| from one anchor: max(0, (D (-) delta) (-) eps+).
inline float lb_from_anchor(float D, float delta, float eps_plus) {
  float t = dn_sub(D, delta);
  t = dn_sub(t, eps_plus);
  return t > 0.f ? t : 0.f;
}

// Upper bound on ||q - x|| from one anchor: (D (+) delta) (+) eps+.
inline float ub_from_anchor(float D, float delta, float eps_plus) {
  return up_add(up_add(D, delta), eps_plus);
}

// Squared-domain bound of Eq. (4); Lemma 1 gives lbS <= s(q, x).
//   lbF = max(0, lb (-) eta);  lbS = (1 - 2u) (x) (lbF (x) lbF).
inline float lbS_from_lb(float lb, const Envelope& env) {
  const float lbF = dn_sub(lb, env.eta);
  if (!(lbF > 0.f)) return 0.f;
  const float sq = dn_mul(lbF, lbF);
  const float r = dn_mul(env.one_minus_2u, sq);
  return r > 0.f ? r : 0.f;
}

// Executor distance e = fl32(sqrt(s)); IEEE sqrt is correctly rounded.
inline float e_from_s(float s) { return std::sqrt(s); }

}  // namespace trace
