#ifndef __BTAS_DOT_IMPL_H
#define __BTAS_DOT_IMPL_H 1

#include <algorithm>
#include <iterator>
#include <type_traits>

#include <btas/tensor_traits.h>
#include <btas/types.h>

#include <btas/generic/numeric_type.h>
#include <btas/generic/tensor_iterator_wrapper.h>
#include <btas/generic/blas_lapack_delegator.h>

namespace btas {

namespace detail {

template <typename... _Ts>
struct dot_return_type;

template <typename _T, typename... Tail>
struct dot_return_type<_T, Tail...> {
private:
  using tail_type = typename dot_return_type<Tail...>::type;
public:
  using type = decltype( std::declval<_T>() * std::declval<tail_type>() );
};

template <typename _T>
struct dot_return_type<_T> {
  using type = _T; 
};

template <typename... _Ts>
using dot_return_type_t = typename dot_return_type<_Ts...>::type;


/*
template <typename... _Tensors>
struct tensor_all_scalar_values;

template <typename Head, typename... Tail>
struct tensor_all_scalar_values< Head, Tail... > {
private:
  static constexpr bool head_value = tensor_all_scalar_values<Head   >::value;
  static constexpr bool tail_value = tensor_all_scalar_values<Tail...>::value;
public:
  static constexpr bool value = head_value and tail_value; 
};

template <typename _Tensor>
struct tensor_all_scalar_values< _Tensor > {
  static constexpr bool value = is_scalar_arithmetic_v< typename _Tensor::value_type >;
};

template <typename = void, typename... _Tensors>
struct tensor_dot_return_type;

template <typename... _Tensors>
struct tensor_dot_return_type< std::enable_if_t<tensor_all_scalar_values<_Tensors...>::value>, _Tensors... > {
  using type = dot_return_type_t<typename _Tensors::value_type...>;
};

template <typename... _Tensors>
struct tensor_dot_return_type< std::enable_if_t<not tensor_all_scalar_values<_Tensors...>::value>, _Tensors... > {
  using type = dot_return_type_t<typename _Tensors::value_type::value_type...>;
};

template <typename... _Tensors>
using tensor_dot_return_type_t = typename tensor_dot_return_type<_Tensors...>::type;
*/
}

template <bool _IsFinal>
struct dotc_impl;
template <bool _IsFinal>
struct dotu_impl;



// Finalized DOTC impl
template <>
struct dotc_impl<true> {

  template<class _IteratorX, class _IteratorY>
  static auto call_impl (
      const unsigned long& Nsize,
            _IteratorX itrX, const iterator_difference_t<_IteratorX>& incX,
            _IteratorY itrY, const iterator_difference_t<_IteratorY>& incY,
      generic_impl_tag)
  {

      auto val = impl::conj(*itrX) * (*itrY);
      itrX += incX;
      itrY += incY;
      for (unsigned long i = 1; i < Nsize; ++i, itrX += incX, itrY += incY)
      {
         val += impl::conj(*itrX) * (*itrY);
      }
      return val;

  }

#ifdef BTAS_HAS_BLAS_LAPACK
  template<class _IteratorX, class _IteratorY>
  static auto call_impl (
      const unsigned long& Nsize,
            _IteratorX itrX, const iterator_difference_t<_IteratorX>& incX,
            _IteratorY itrY, const iterator_difference_t<_IteratorY>& incY,
      blas_lapack_impl_tag)
  {

      using x_traits = std::iterator_traits<_IteratorX>;
      using y_traits = std::iterator_traits<_IteratorY>;

      using x_value_type = typename x_traits::value_type;
      using y_value_type = typename y_traits::value_type;

      using x_ptr_type = const x_value_type*;
      using y_ptr_type = const y_value_type*;

      //std::cout << "IN BLASPP DOTC IMPL" << std::endl;

      // XXX: DOTC == DOT in BLASPP
      return blas::dot( Nsize, static_cast<x_ptr_type>(&(*itrX)), incX,
                               static_cast<y_ptr_type>(&(*itrY)), incY );

  }
#endif

  template<class _IteratorX, class _IteratorY>
  static auto call (
      const unsigned long& Nsize,
            _IteratorX itrX, const iterator_difference_t<_IteratorX>& incX,
            _IteratorY itrY, const iterator_difference_t<_IteratorY>& incY )
  {
      return call_impl( Nsize, itrX, incX, itrY, incY, 
                        blas_lapack_impl_t<_IteratorX,_IteratorY>() );
  }

};


// Finalized DOTU impl
template <>
struct dotu_impl<true> {

  template<class _IteratorX, class _IteratorY>
  static auto call_impl (
      const unsigned long& Nsize,
            _IteratorX itrX, const iterator_difference_t<_IteratorX>& incX,
            _IteratorY itrY, const iterator_difference_t<_IteratorY>& incY,
      generic_impl_tag)
  {

      auto val = (*itrX) * (*itrY);
      itrX += incX;
      itrY += incY;
      for (unsigned long i = 1; i < Nsize; ++i, itrX += incX, itrY += incY)
      {
         val += (*itrX) * (*itrY);
      }
      return val;

  }

#ifdef BTAS_HAS_BLAS_LAPACK
  template<class _IteratorX, class _IteratorY>
  static auto call_impl (
      const unsigned long& Nsize,
            _IteratorX itrX, const iterator_difference_t<_IteratorX>& incX,
            _IteratorY itrY, const iterator_difference_t<_IteratorY>& incY,
      blas_lapack_impl_tag)
  {

      using x_traits = std::iterator_traits<_IteratorX>;
      using y_traits = std::iterator_traits<_IteratorY>;

      using x_value_type = typename x_traits::value_type;
      using y_value_type = typename y_traits::value_type;

      using x_ptr_type = const x_value_type*;
      using y_ptr_type = const y_value_type*;

      //std::cout << "IN BLASPP DOTU IMPL" << std::endl;

      return blas::dotu( Nsize, static_cast<x_ptr_type>(&(*itrX)), incX,
                                static_cast<y_ptr_type>(&(*itrY)), incY );

  }
#endif

  template<class _IteratorX, class _IteratorY>
  static auto call (
      const unsigned long& Nsize,
            _IteratorX itrX, const iterator_difference_t<_IteratorX>& incX,
            _IteratorY itrY, const iterator_difference_t<_IteratorY>& incY )
  {
      return call_impl( Nsize, itrX, incX, itrY, incY, 
                        blas_lapack_impl_t<_IteratorX,_IteratorY>() );
  }

};




/// Unfinalized DOTC impl
template <>
struct dotc_impl<false>
{

  template<class _IteratorX, class _IteratorY>
  static auto call (
      const unsigned long& Nsize,
            _IteratorX itrX, const iterator_difference_t<_IteratorX>& incX,
            _IteratorY itrY, const iterator_difference_t<_IteratorY>& incY )
  {
      auto val = dotc( *itrX, *itrY );

      itrX += incX;
      itrY += incY;
      for (unsigned long i = 1; i < Nsize; ++i, itrX += incX, itrY += incY)
      {
         val += dotc(*itrX, *itrY);
      }
      return val;
  }


};

/// Unfinalized DOTU impl
template <>
struct dotu_impl<false>
{

  template<class _IteratorX, class _IteratorY>
  static auto call (
      const unsigned long& Nsize,
            _IteratorX itrX, const iterator_difference_t<_IteratorX>& incX,
            _IteratorY itrY, const iterator_difference_t<_IteratorY>& incY )
  {
      auto val = dotu( *itrX, *itrY );

      itrX += incX;
      itrY += incY;
      for (unsigned long i = 1; i < Nsize; ++i, itrX += incX, itrY += incY)
      {
         val += dotu(*itrX, *itrY);
      }
      return val;
  }


};

//  ================================================================================================

/// Generic implementation of BLAS DOT in terms of C++ iterator
template<class _IteratorX, class _IteratorY>
auto dotc (
   const unsigned long& Nsize,
            _IteratorX itrX, const iterator_difference_t<_IteratorX>& incX,
            _IteratorY itrY, const iterator_difference_t<_IteratorY>& incY )
{
   typedef std::iterator_traits<_IteratorX> __traits_X;
   typedef std::iterator_traits<_IteratorY> __traits_Y;

   static_assert(std::is_same<typename __traits_X::value_type, typename __traits_Y::value_type>::value, "value type of Y must be the same as that of X");
   static_assert(is_random_access_iterator_v<_IteratorX>, "iterator X must be a random access iterator");
   static_assert(is_random_access_iterator_v<_IteratorY>, "iterator Y must be a random access iterator");

   constexpr bool value_is_scalar = is_scalar_arithmetic_v< typename __traits_X::value_type>;
   return dotc_impl<value_is_scalar>::call(Nsize, itrX, incX, itrY, incY);
}

/// Generic implementation of BLAS DOT in terms of C++ iterator
template<class _IteratorX, class _IteratorY>
auto dotu (
   const unsigned long& Nsize,
            _IteratorX itrX, const iterator_difference_t<_IteratorX>& incX,
            _IteratorY itrY, const iterator_difference_t<_IteratorY>& incY )
{
   typedef std::iterator_traits<_IteratorX> __traits_X;
   typedef std::iterator_traits<_IteratorY> __traits_Y;

   static_assert(std::is_same<typename __traits_X::value_type, typename __traits_Y::value_type>::value, "value type of Y must be the same as that of X");
   static_assert(is_random_access_iterator_v<_IteratorX>, "iterator X must be a random access iterator");
   static_assert(is_random_access_iterator_v<_IteratorY>, "iterator Y must be a random access iterator");

   constexpr bool value_is_scalar = is_scalar_arithmetic_v< typename __traits_X::value_type>;
   return dotu_impl<value_is_scalar>::call(Nsize, itrX, incX, itrY, incY);
}

/// Generic implementation of BLAS DOT in terms of C++ iterator
template<class _IteratorX, class _IteratorY>
auto dot (
   const unsigned long& Nsize,
         _IteratorX itrX, const typename std::iterator_traits<_IteratorX>::difference_type& incX,
         _IteratorY itrY, const typename std::iterator_traits<_IteratorY>::difference_type& incY)
{
   return dotc(Nsize, itrX, incX, itrY, incY);
}

//  ================================================================================================

/// Convenient wrapper to call BLAS DOT-C from tensor objects
template<
   class _TensorX,
   class _TensorY,
   class = typename std::enable_if<
      is_tensor<_TensorX>::value &
      is_tensor<_TensorY>::value
   >::type
>
detail::dot_return_type_t< typename _TensorX::numeric_type, typename _TensorY::numeric_type >
dotc (const _TensorX& X, const _TensorY& Y)
{
   typedef typename _TensorX::value_type value_type;
   static_assert(std::is_same<value_type, typename _TensorY::value_type>::value, "value type of Y must be the same as that of X");

   if (X.empty() || Y.empty())
   {
      return 0;
   }

   auto itrX = tbegin(X);
   auto itrY = tbegin(Y);

   return dotc(X.size(), itrX, 1, itrY, 1);
}

/// Convenient wrapper to call BLAS DOT-U from tensor objects
template<
   class _TensorX,
   class _TensorY,
   class = typename std::enable_if<
      is_tensor<_TensorX>::value &
      is_tensor<_TensorY>::value
   >::type
>
detail::dot_return_type_t< typename _TensorX::numeric_type, typename _TensorY::numeric_type >
dotu (const _TensorX& X, const _TensorY& Y)
{
   typedef typename _TensorX::value_type value_type;
   static_assert(std::is_same<value_type, typename _TensorY::value_type>::value, "value type of Y must be the same as that of X");

   if (X.empty() || Y.empty())
   {
      return 0;
   }

   auto itrX = tbegin(X);
   auto itrY = tbegin(Y);

   return dotu(X.size(), itrX, 1, itrY, 1);
}

/// Convenient wrapper to call BLAS DOT from tensor objects
template<
   class _TensorX,
   class _TensorY,
   class = typename std::enable_if<
      is_tensor<_TensorX>::value &
      is_tensor<_TensorY>::value
   >::type
>
auto dot (const _TensorX& X, const _TensorY& Y)
{
   return dotc(X, Y);
}

} // namespace btas

#endif // __BTAS_DOT_IMPL_H
