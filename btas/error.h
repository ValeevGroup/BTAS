#ifndef __BTAS_ERROR_H
#define __BTAS_ERROR_H

#include <cstdio>
#include <cstdlib>
#include <exception>

namespace btas {

  /// exception class, used to mark exceptions specific to BTAS
  class exception : public std::exception {
    public:
      exception(const char* m) : message_(m) { }

      virtual const char* what() const noexcept { return message_; }

    private:
      const char* message_;
  }; // class exception

  /// Place a break point in this function to stop before btas::exception is thrown.
  inline void exception_break() { }

} // namespace btas

#define BTAS_STRINGIZE( s ) #s

#define BTAS_EXCEPTION_MESSAGE( file , line , mess ) \
  "BTAS: exception at " file "(" BTAS_STRINGIZE( line ) "): " mess ". Break in btas::exception_break to learn more."

#define BTAS_EXCEPTION( m ) \
    { \
      btas::exception_break(); \
      throw btas::exception ( BTAS_EXCEPTION_MESSAGE( __FILE__ , __LINE__ , m ) ); \
    }

// configure BTAS_ASSERT

/// value of BTAS_ASSERT_POLICY that makes BTAS_ASSERT throw btas::exception
#define BTAS_ASSERT_THROW 2
/// value of BTAS_ASSERT_POLICY that makes BTAS_ASSERT abort
#define BTAS_ASSERT_ABORT 3
/// value of BTAS_ASSERT_POLICY that makes BTAS_ASSERT a no-op
#define BTAS_ASSERT_IGNORE 4

#ifndef BTAS_ASSERT_POLICY
#  ifdef BTAS_ASSERT_THROWS
// BTAS_ASSERT_THROWS is deprecated in favor of BTAS_ASSERT_POLICY, but is still honored
#    define BTAS_ASSERT_POLICY BTAS_ASSERT_THROW
#  else
#    define BTAS_ASSERT_POLICY BTAS_ASSERT_ABORT
#  endif
#endif

#if BTAS_ASSERT_POLICY != BTAS_ASSERT_THROW && \
    BTAS_ASSERT_POLICY != BTAS_ASSERT_ABORT && \
    BTAS_ASSERT_POLICY != BTAS_ASSERT_IGNORE
#  error "invalid BTAS_ASSERT_POLICY; valid values are BTAS_ASSERT_THROW, BTAS_ASSERT_ABORT, and BTAS_ASSERT_IGNORE"
#endif

namespace btas {

  /// Reports a failed BTAS_ASSERT as prescribed by BTAS_ASSERT_POLICY: throws
  /// btas::exception (BTAS_ASSERT_THROW) or reports \p m to `stderr` and aborts
  /// (BTAS_ASSERT_ABORT). Neither is affected by `NDEBUG`.
  /// \param m the message; must have static storage duration
  inline void assert_failed(const char* m) {
#if BTAS_ASSERT_POLICY == BTAS_ASSERT_THROW
    btas::exception_break();
    throw btas::exception(m);
#elif BTAS_ASSERT_POLICY == BTAS_ASSERT_ABORT
    btas::exception_break();
    std::fprintf(stderr, "%s\n", m);
    std::fflush(stderr);
    std::abort();
#else  // BTAS_ASSERT_POLICY == BTAS_ASSERT_IGNORE
    (void)m;
#endif
  }

} // namespace btas

#if BTAS_ASSERT_POLICY == BTAS_ASSERT_IGNORE

// N.B. the argument is NOT evaluated, hence must be free of side effects
#  define BTAS_ASSERT( a )  do { } while(0)

#else // BTAS_ASSERT_POLICY == BTAS_ASSERT_IGNORE

#  define BTAS_ASSERT( a )  \
     do { \
       if(! ( a ) ) \
         btas::assert_failed( BTAS_EXCEPTION_MESSAGE( __FILE__ , __LINE__ , "assertion failed: " BTAS_STRINGIZE( a ) ) ); \
     } while(0)

#endif // BTAS_ASSERT_POLICY == BTAS_ASSERT_IGNORE

#endif // __BTAS_ERROR_H
