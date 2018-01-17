#include <chrono>
#include <iostream>

#include "test.h"

#include <btas/btas.h>
#include <btas/tarray.h>
#include "btas/tarray.h"
#include "btas/tensor.h"
#include "btas/tensorview.h"

/// TimerPool aggregates \c N C++11 "timers"; used to high-resolution profile
/// stages of integral computation
/// @tparam N the number of timers
/// @note member functions are not reentrant, use one Timers object per thread
template <size_t N = 1>
class TimerPool {
 public:
  typedef std::chrono::duration<double> dur_t;
  typedef std::chrono::high_resolution_clock clock_t;
  typedef std::chrono::time_point<clock_t> time_point_t;

  TimerPool() {
    clear();
    set_now_overhead(0);
  }

  /// returns the current time point
  static time_point_t now() { return clock_t::now(); }

  /// use this to report the overhead of now() call; if set, the reported
  /// timings will be adjusted for this overhead
  /// @note this is clearly compiler and system dependent, please measure
  /// carefully (turn off turboboost, etc.)
  ///       using src/bin/profile/chrono.cc
  void set_now_overhead(size_t ns) { overhead_ = std::chrono::nanoseconds(ns); }

  /// starts timer \c t
  void start(size_t t = 0) { tstart_[t] = now(); }
  /// stops timer \c t
  /// @return the duration, corrected for overhead, elapsed since the last call
  /// to \c start(t)
  dur_t stop(size_t t = 0) {
    const auto tstop = now();
    const dur_t result = (tstop - tstart_[t]) - overhead_;
    timers_[t] += result;
    return result;
  }
  /// reads value (in seconds) of timer \c t , converted to \c double
  double read(size_t t = 0) const { return timers_[t].count(); }
  /// resets timers to zero
  void clear() {
    for (auto t = 0; t != ntimers; ++t) {
      timers_[t] = dur_t::zero();
      tstart_[t] = time_point_t();
    }
  }

 private:
  constexpr static auto ntimers = N;
  dur_t timers_[ntimers];
  time_point_t tstart_[ntimers];
  dur_t overhead_;  // the duration of now() call ... use this to automatically
                    // adjust reported timings is you need fine-grained timing
};

btas::Tensor<double> T3(3, 2, 4);

inline double f() { return T3(2, 1, 3); }

inline double g(const btas::DEFAULT::index_type& stride) {
  return T3.data()[stride[0] * 2 + stride[1] * 1 + stride[2] * 3];
}

inline double h() {
  static auto cview = make_cview(T3);
  return cview(2, 1, 3);
}

#define BTAS_PROFILE(call)                                                   \
  _Pragma("ivdep") for (int64_t nrepeats = 1; nrepeats < 10000000000;        \
                        nrepeats *= 2) {                                     \
    TimerPool<> timer;                                                       \
    timer.start();                                                           \
    _Pragma("novector") for (auto i = 0; i != nrepeats; ++i) { (call); }     \
    timer.stop();                                                            \
    auto elapsed_seconds = timer.read();                                     \
    if (elapsed_seconds > 1) {                                               \
      std::cout << "Tensor::operator(): " << std::scientific                 \
                << elapsed_seconds / nrepeats << " seconds/op" << std::endl; \
      break;                                                                 \
    }                                                                        \
  }

TEST_CASE("performance") {
  T3.fill(1.);

  SECTION("Tensor::operator()") {
    double sum1 = 0.0;
    BTAS_PROFILE(sum1 += f());
    std::cout << sum1 << std::endl;
  }

  SECTION("Tensor::operator() manual unroll") {
    btas::DEFAULT::index_type stride(3);
    stride[0] = 8;
    stride[1] = 4;
    stride[2] = 1;

    double sum2 = 0.0;
    BTAS_PROFILE(sum2 += g(stride));
    std::cout << sum2 << std::endl;
  }

  SECTION("make_view + TensorView::operator()") {
    double sum = 0.0;
    BTAS_PROFILE(sum += h());
    std::cout << sum << std::endl;
  }

}
