/// Checks that BTAS_ASSERT obeys BTAS_ASSERT_POLICY. Compiled once per policy
/// (see unittest/CMakeLists.txt), with NDEBUG defined, so that each policy is
/// also checked to be independent of NDEBUG.
/// Returns 0 iff the policy is obeyed.

#include <btas/error.h>

#include <csignal>
#include <cstdio>
#include <cstdlib>

namespace {
  int nevals = 0;

  /// \return false, and counts how many times it was evaluated
  bool false_with_side_effect() {
    ++nevals;
    return false;
  }

#if BTAS_ASSERT_POLICY == BTAS_ASSERT_ABORT
  /// only async-signal-safe calls are permitted here, hence no stdio;
  /// reaching this handler is the success condition
  extern "C" void abort_handler(int) { std::_Exit(EXIT_SUCCESS); }
#endif
}  // namespace

int main() {
#if BTAS_ASSERT_POLICY == BTAS_ASSERT_THROW

  try {
    BTAS_ASSERT(false_with_side_effect());
  } catch (const btas::exception& e) {
    std::fprintf(stdout, "BTAS_ASSERT threw btas::exception: %s\n", e.what());
    return nevals == 1 ? EXIT_SUCCESS : EXIT_FAILURE;
  }
  std::fprintf(stderr, "BTAS_ASSERT did not throw\n");
  return EXIT_FAILURE;

#elif BTAS_ASSERT_POLICY == BTAS_ASSERT_ABORT

  std::signal(SIGABRT, &abort_handler);
  BTAS_ASSERT(false_with_side_effect());
  std::fprintf(stderr, "BTAS_ASSERT did not abort\n");
  return EXIT_FAILURE;

#elif BTAS_ASSERT_POLICY == BTAS_ASSERT_IGNORE

  BTAS_ASSERT(false_with_side_effect());
  if (nevals != 0) {
    std::fprintf(stderr, "BTAS_ASSERT evaluated its argument\n");
    return EXIT_FAILURE;
  }
  std::fprintf(stdout, "BTAS_ASSERT ignored, as expected\n");
  return EXIT_SUCCESS;

#else
#  error "BTAS_ASSERT_POLICY not set to one of the 3 valid values"
#endif
}
