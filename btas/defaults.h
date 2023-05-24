/*
 * defaults.h
 *
 *  Created on: Dec 19, 2013
 *      Author: evaleev
 */

#ifndef BTAS_DEFAULTS_H_
#define BTAS_DEFAULTS_H_

#include <btas/features.h>

#include <vector>

#ifdef BTAS_HAS_BOOST_CONTAINER
#include <boost/container/container_fwd.hpp>
#else
#include <btas/varray/varray.h>
#endif

//
//  Default index type
//

namespace btas {
namespace DEFAULT {

/// default index type
#ifdef BTAS_HAS_BOOST_CONTAINER
#ifndef BTAS_TARGET_MAX_INDEX_RANK
#  ifdef BTAS_DEFAULT_TARGET_MAX_INDEX_RANK
#    define BTAS_TARGET_MAX_INDEX_RANK BTAS_DEFAULT_TARGET_MAX_INDEX_RANK
#  else
#    define BTAS_TARGET_MAX_INDEX_RANK 6
#endif
#endif
template <typename Integer>
using index =
    boost::container::small_vector<Integer, BTAS_TARGET_MAX_INDEX_RANK>;
#else
template <typename Integer>
using index = btas::varray<Integer>;
#endif
using index_type = index<long>;

/// default storage class
template <typename _T>
using storage = std::vector<_T>;
}
}  // namespace btas

#endif /* BTAS_DEFAULTS_H_ */
