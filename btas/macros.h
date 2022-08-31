//
// Created by Eduard Valeyev on 8/31/22.
//

#ifndef __BTAS_MACROS_H
#define __BTAS_MACROS_H

/* detect C++ compiler id:
- ids taken from CMake
- macros are discussed at https://sourceforge.net/p/predef/wiki/Compilers/
*/
#define BTAS_CXX_COMPILER_ID_GNU 0
#define BTAS_CXX_COMPILER_ID_Clang 1
#define BTAS_CXX_COMPILER_ID_AppleClang 2
#define BTAS_CXX_COMPILER_ID_XLClang 3
#define BTAS_CXX_COMPILER_ID_Intel 4
#if defined(__INTEL_COMPILER_BUILD_DATE)  /* macros like __ICC and even __INTEL_COMPILER can be affected by command options like -no-icc */
# define BTAS_CXX_COMPILER_ID BTAS_CXX_COMPILER_ID_Intel
# define BTAS_CXX_COMPILER_IS_ICC 1
#endif
#if defined(__clang__) && !defined(BTAS_CXX_COMPILER_IS_ICC)
# define BTAS_CXX_COMPILER_IS_CLANG 1
# if defined(__apple_build_version__)
#  define BTAS_CXX_COMPILER_ID BTAS_CXX_COMPILER_ID_AppleClang
# elif defined(__ibmxl__)
#  define BTAS_CXX_COMPILER_ID BTAS_CXX_COMPILER_ID_XLClang
# else
#  define BTAS_CXX_COMPILER_ID BTAS_CXX_COMPILER_ID_Clang
# endif
#endif
#if defined(__GNUG__) && !defined(BTAS_CXX_COMPILER_IS_ICC) && !defined(BTAS_CXX_COMPILER_IS_CLANG)
# define BTAS_CXX_COMPILER_ID BTAS_CXX_COMPILER_ID_GNU
# define BTAS_CXX_COMPILER_IS_GCC 1
#endif

/* ----------- pragma helpers ---------------*/
#define BTAS_PRAGMA(x) _Pragma(#x)
/* same as BTAS_PRAGMA(x), but expands x */
#define BTAS_XPRAGMA(x) BTAS_PRAGMA(x)
/* "concats" a and b with a space in between */
#define BTAS_CONCAT(a,b) a b
#if defined(BTAS_CXX_COMPILER_IS_CLANG)
#define BTAS_PRAGMA_CLANG(x) BTAS_XPRAGMA( BTAS_CONCAT(clang,x) )
#else
#define BTAS_PRAGMA_CLANG(x)
#endif
#if defined(BTAS_CXX_COMPILER_IS_GCC)
#define BTAS_PRAGMA_GCC(x) BTAS_XPRAGMA( BTAS_CONCAT(GCC,x) )
#else
#define BTAS_PRAGMA_GCC(x)
#endif

#endif  // __BTAS_MACROS_H
