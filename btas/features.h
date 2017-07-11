/**
 * @file features.h
 *
 * include this to import macros describing features of BTAS
 * the only available macros are:
 * - BTAS_IS_USABLE : #define'd to 1 if BTAS is usable
 * - BTAS_HAS_BOOST_CONTAINER : #define'd to 1 if BTAS detected Boost.Container
 *
 * ALSO: a library configured with cmake (hence, non-header-only) will define BTAS_HAS_BOOST_SERIALIZATION (via a compiler flag) to 1 if
 * Boost.Serialization library were found.
 */

#ifndef BTAS_FEATURES_H_
#define BTAS_FEATURES_H_

#ifdef __has_include
#if !defined(BTAS_HAS_BOOST_ITERATOR) && \
    __has_include(<boost/iterator/transform_iterator.hpp>)
#define BTAS_HAS_BOOST_ITERATOR 1
#endif  // define BTAS_HAS_BOOST_ITERATOR if Boost.Iterator headers are
        // available

#if !defined(BTAS_HAS_BOOST_CONTAINER) && \
    __has_include(<boost/container/small_vector.hpp>)
#define BTAS_HAS_BOOST_CONTAINER 1
#endif  // define BTAS_HAS_BOOST_CONTAINER if Boost.Container headers are
        // available

#endif  // defined( __has_include)

#ifdef BTAS_HAS_BOOST_ITERATOR
#define BTAS_IS_USABLE 1
#else
#ifdef BTAS_SIGNAL_MISSING_PREREQUISITES
#error \
    "Cannot find Boost.Iterators headers => BTAS is not usable as a headers-only library; download latest Boost from boost.org and provide -I/path/to/boost to the compiler"
#endif
#endif

#endif /* BTAS_FEATURES_H_ */
