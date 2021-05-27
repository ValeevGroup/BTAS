#ifndef __BTAS_SERIALIZATION_H
#define __BTAS_SERIALIZATION_H 1

#include <btas/features.h>

////// Boost serialization

#ifdef BTAS_HAS_BOOST_SERIALIZATION
#include <array>
#include <boost/version.hpp>
#include <boost/serialization/is_bitwise_serializable.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>

namespace boost { namespace serialization {
  // this is needed to serialize  efficiently corner cases, like std::vector<std::array<std::complex<T>>>.
  // since bitwise serialization is not portable anyway, this is OK in the context of btas
  template <typename T, size_t N>
  struct is_bitwise_serializable<std::array<T,N> > : is_bitwise_serializable<T> { };
}}
#endif  // BTAS_HAS_BOOST_SERIALIZATION

////// MADNESS serialization
#if __has_include(<madness/world/type_traits.h>)
# include <madness/world/type_traits.h>
#else  // __has_include(<madness/world/type_traits.h>)
namespace madness::archive {
    template <typename Archive, typename T, typename Enabler = void>
    struct ArchiveSerializeImpl;
    template <class Archive, class T, typename Enabler = void>
    struct ArchiveLoadImpl;
    template <class Archive, class T, typename Enabler = void>
    struct ArchiveStoreImpl;
}  // namespace madness::archive
#endif  // __has_include(<madness/world/type_traits.h>)

#endif
