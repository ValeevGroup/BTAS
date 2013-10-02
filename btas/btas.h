#ifndef __BTAS_BTAS_H
#define __BTAS_BTAS_H

namespace btas
{

//! multiply a tensor x by a scalar
template <typename T>
void 
scal(T alpha, Tensor<T>& x);

//! compute y=alpha*x+y
template <typename T>
void 
axpy(T alpha, const Tensor<T>& x, Tensor<T>& y);

//! compute x.y
template <typename T>
T 
dot(const Tensor<T>& x, const Tensor<T>& y);

//! compute x.x
template <typename T>
T 
nrm2(const Tensor<T>& x);

template <typename T>
void 
gemv(T alpha, const Tensor<T>& a, const Tensor<T>& b, T beta, Tensor<T>& c);

template <typename T>
void 
ger(T alpha, const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& c);

template <typename T>
void 
gemm(T alpha, const Tensor<T>& a, const Tensor<T>& b, T beta, Tensor<T>& c);

template <typename T>
void 
dimd(Tensor<T>& a, const Tensor<T>& b);

template <typename T>
void 
didm(const Tensor<T>& a, Tensor<T>& b);

template <typename T>
void 
normalize(Tensor<T>& x);

template <typename T>
void 
orthogonalize(const Tensor<T>& x, Tensor<T>& y);

} //namespace btas

#endif
