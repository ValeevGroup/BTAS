#ifndef __BTAS_BTAS_H
#define __BTAS_BTAS_H

#include <aliases.h>
#include <numeric_type.h>
#include <type_traits>

namespace btas {

/// copy x to y
template <class _Tensor>
void 
copy(const _Tensor& x, _Tensor& y)
{
   BTAS_ASSERT(x.size() == y.size());
   NUMERIC_TYPE<typename _Tensor::value_type>::copy(x.size(), x.data(), 1, y.data(), 1)
}

/// multiply a tensor x by a scalar
template <class _Tensor>
void 
scal(const typename _Tensor::value_type& alpha, _Tensor& x)
{
   NUMERIC_TYPE<typename _Tensor::value_type>::scal(x.size(), alpha, x.data(), 1)
}

/// compute y = alpha*x + y
template <class _Tensor>
void 
axpy(const typename _Tensor::value_type& alpha, const _Tensor& x, _Tensor& y)
{
   BTAS_ASSERT(x.size() == y.size(), "ERROR: size mismatched");
   NUMERIC_TYPE<typename _Tensor::value_type>::axpy(x.size(), alpha, x.data(), 1, y.data(), 1);
}

/// compute x.y
template <class _Tensor>
_Tensor::value_type 
dot(const _Tensor& x, const _Tensor& y)
{
   BTAS_ASSERT(x.size() == y.size(), "ERROR: size mismatched");
   NUMERIC_TYPE<typename _Tensor::value_type>::dot(x.size(), x.data(), 1, y.data(), 1);
}

/// compute sqrt(x.x)
/// _Tensor::value_type::value_type extract value_type of complex number tensor
template <class _Tensor>
std::conditional<std::is_arithmetic<typename _Tensor::value_type>, typename _Tensor::value_type, typename _Tensor::value_type::value_type>::type
nrm2(const _Tensor& x)
{
   return NUMERIC_TYPE<typename _Tensor::value_type>::nrm2(x.size(), x.data(), 1);
}

template <class _TensorA, class _TensorX, class _TensorY>
void 
gemv(const CBLAS_TRANSPOSE transa, const typename _TensorA::value_type& alpha, const _TensorA& a, const _TensorX& x, const typename _TensorY::value_type& beta, _TensorY& y)
{
   BTAS_STATIC_ASSERT(std::is_same<typename _TensorA::value_type, typename _TensorX::value_type>::value_type, "ERROR: type mismatched");
   BTAS_STATIC_ASSERT(std::is_same<typename _TensorA::value_type, typename _TensorY::value_type>::value_type, "ERROR: type mismatched");
   INTEGER_TYPE m = y.size();
   INTEGER_TYPE n = x.size();
   if (transa != NoTrans) std::swap(m, n);
   NUMERIC_TYPE<typename _TensorA::value_type>::gemv(transa, m, n, alpha, a.data(), n, x.data(), 1, beta, y.data(), 1);
}

template <class _TensorX, class _TensorY, class _TensorA>
void 
ger(const typename _TensorA::value_type& alpha, const _TensorX& x, const _TensorY& y, _TensorA& a)
{
   BTAS_STATIC_ASSERT(std::is_same<typename _TensorA::value_type, typename _TensorX::value_type>::value_type, "ERROR: type mismatched");
   BTAS_STATIC_ASSERT(std::is_same<typename _TensorA::value_type, typename _TensorY::value_type>::value_type, "ERROR: type mismatched");
   INTEGER_TYPE m = x.size();
   INTEGER_TYPE n = y.size();
   NUMERIC_TYPE<typename _TensorA::value_type>::ger(m, n, alpha, x.data(), 1, y.data(), 1, a.data(), n);
}

/// matrix-matrix multiplication in BLAS level 3
template <class _TensorA, class _TensorB, class _TensorC>
void 
gemm (
      const CBLAS_TRANSPOSE transa,
      const CBLAS_TRANSPOSE transb,
      const typename _TensorA::value_type& alpha,
      const _TensorA& a,
      const _TensorB& b,
      const typename _TensorC::value_type& beta,
            _TensorC& c
) {
   typedef typename _TensorA::value_type value_type;
   BTAS_STATIC_ASSERT(std::is_same<value_type, typename _TensorB::value_type>::value_type, "ERROR: type mismatched");
   BTAS_STATIC_ASSERT(std::is_same<value_type, typename _TensorC::value_type>::value_type, "ERROR: type mismatched");

   const INTEGER_TYPE a_rank = a.rank();
   const INTEGER_TYPE b_rank = b.rank();
   const INTEGER_TYPE c_rank = c.rank();
   const INTEGER_TYPE k_rank = (a_rank + b_rank - c_rank) / 2;

   const typename _TensorA::shape_type& a_shape = a.shape();
   const typename _TensorB::shape_type& b_shape = b.shape();
         typename _TensorC::shape_type  c_shape = c.shape();

   if (c.empty()) {

      const typename _TensorA::range_type& a_range = a.range();
      const typename _TensorB::range_type& b_range = b.range();
            typename _TensorC::range_type  c_range = c.range();

      if (transa == NoTrans) {
         for (size_t i = 0; i < a_rank-k_rank; ++i) c_range[i] = a_range[i];
      }
      else {
         for (size_t i = 0; i < a_rank-k_rank; ++i) c_range[i] = a_range[i+k_rank];
      }

      if (transb == NoTrans) {
         for (size_t i = 0; i < b_rank-k_rank; ++i) c_range[i+a_rank-k_rank] = b_range[i+k_rank];
      }
      else {
         for (size_t i = 0; i < b_rank-k_rank; ++i) c_range[i+a_rank-k_rank] = b_range[i];
      }

      c.resize(c_range, NUMERIC_TYPE<value_type>::zero());
   }
   else {

      if (transa == NoTrans) {
         for (size_t i = 0; i < a_rank-k_rank; ++i)
            c_shape[i] = a_shape[i];
      }
      else {
         for (size_t i = 0; i < a_rank-k_rank; ++i)
            c_shape[i] = a_shape[i+k_rank];
      }

      if (transb == NoTrans) {
         for (size_t i = 0; i < b_rank-k_rank; ++i)
            c_shape[i+a_rank-k_rank] = b_shape[i+k_rank];
      }
      else {
         for (size_t i = 0; i < b_rank-k_rank; ++i)
            c_shape[i+a_rank-k_rank] = b_shape[i];
      }

      BTAS_ASSERT(std::equal(c_shape.begin(), c_shape.end(), c.shape().begin()));
      scal(beta, c);
   }

   INTEGER_TYPE k;
   if (transa == NoTrans)
      k = std::accumulate(a_shape.begin()+a_rank-k_rank, a_shape.end(),   1, std::multiplies<INTEGER_TYPE>());
   else
      k = std::accumulate(a_shape.begin(), a_shape.begin()+a_rank-k_rank, 1, std::multiplies<INTEGER_TYPE>());

   BTAS_ASSERT(k > 0);
   INTEGER_TYPE m = a.size() / k;
   INTEGER_TYPE n = b.size() / k;

   INTEGER_TYPE lda = (transa == NoTrans) ? k : m;
   INTEGER_TYPE ldb = (transb == NoTrans) ? n : k;

   gemm(transa, transb, m, n, k, alpha, a.data(), lda, b.data(), ldb, beta, c.data(), n);
}

}; // namespace btas

#endif // __BTAS_BTAS_H
