/*
 * deprecated.h
 *
 *  Created on: Jul 9, 2017
 *      Author: evaleev
 */

#ifndef BTAS_UTIL_DEPRECATED_H_
#define BTAS_UTIL_DEPRECATED_H_

// mark functions as deprecated using this macro
// will result in a warning
#if __cplusplus >= 201402L
#define DEPRECATED  [[deprecated]]
#elif defined(__GNUC__)
#define DEPRECATED __attribute__((deprecated))
#else
#pragma message("WARNING: You need to implement DEPRECATED for this compiler")
#define DEPRECATED
#endif

// same as DEPRECATED, but annotated with a message
// will result in a warning
#if __cplusplus >= 201402L
#define DEPRECATEDMSG(msg)  [[deprecated(msg)]]
#elif defined(__GNUC__)
#define DEPRECATEDMSG(msg) __attribute__((deprecated(msg)))
#else
#pragma message("WARNING: You need to implement DEPRECATEDMSG for this compiler")
#define DEPRECATEDMSG(msg)
#endif

#endif /* BTAS_UTIL_DEPRECATED_H_ */
