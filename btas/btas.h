#ifndef __BTAS_BTAS_H
#define __BTAS_BTAS_H

#include <aliases.h>
#include <numeric_type.h>
#include <type_traits>

namespace btas
{

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
gemv(const TRANSPOSE transa, const typename _TensorA::value_type& alpha, const _TensorA& a, const _TensorX& x, const typename _TensorY::value_type& beta, _TensorY& y)
{
   BTAS_STATIC_ASSERT(std::is_same<typename _TensorA::value_type, typename _TensorX::value_type>::value_type, "ERROR: type mismatched");
   BTAS_STATIC_ASSERT(std::is_same<typename _TensorA::value_type, typename _TensorY::value_type>::value_type, "ERROR: type mismatched");
   INTEGER_TYPE m = y.size();
   INTEGER_TYPE n = x.size();
   if (transa != NoTrans) std::swap(m, n);
   gemv(transa, m, n, alpha, a.data(), n, x.data(), 1, beta, y.data(), 1);
}

template <class _TensorX, class _TensorY, class _TensorA>
void 
ger(const typename _TensorA::value_type& alpha, const _TensorX& x, const _TensorY& y, _TensorA& a)
{
   BTAS_STATIC_ASSERT(std::is_same<typename _TensorA::value_type, typename _TensorX::value_type>::value_type, "ERROR: type mismatched");
   BTAS_STATIC_ASSERT(std::is_same<typename _TensorA::value_type, typename _TensorY::value_type>::value_type, "ERROR: type mismatched");
   INTEGER_TYPE m = x.size();
   INTEGER_TYPE n = y.size();
   ger(m, n, alpha, x.data(), 1, y.data(), 1, a.data(), n);
}

template <class _TensorA, class _TensorB, class _TensorC>
void 
gemv(const TRANSPOSE transa, const TRANSPOSE transb, const typename _TensorA::value_type& alpha, const _TensorA& a, const _TensorB& b, const typename _TensorC::value_type& beta, _TensorC& c)
{
   BTAS_STATIC_ASSERT(std::is_same<typename _TensorA::value_type, typename _TensorB::value_type>::value_type, "ERROR: type mismatched");
   BTAS_STATIC_ASSERT(std::is_same<typename _TensorA::value_type, typename _TensorC::value_type>::value_type, "ERROR: type mismatched");
   INTEGER_TYPE a_size = a.size();
   INTEGER_TYPE b_size = b.size();
   INTEGER_TYPE c_size = c.size();
   INTEGER_TYPE k = static_cast<INTEGER_TYPE>(sqrt(static_cast<double>(a_size*b_size/c_size)));
   BTAS_ASSERT(k > 0, "ERROR: 0 division");
   INTEGER_TYPE m = a_size/k;
   INTEGER_TYPE n = b_size/k;
   INTEGER_TYPE lda = (transa == NoTrans) ? k : m;
   INTEGER_TYPE ldb = (transb == NoTrans) ? n : k;
   gemm(transa, transb, m, n, k, alpha, a.data(), lda, b.data(), ldb, beta, c.data(), n);
}

}; // namespace btas

#endif // __BTAS_BTAS_H
