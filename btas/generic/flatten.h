#ifndef BTAS_FLATTEN
#define BTAS_FLATTEN
namespace btas {

/// methods to produce to matricize a tensor along the n-th fiber
template <typename Tensor>
Tensor flatten(Tensor &A, int mode) {
  using value_type = typename Tensor::value_type;
  typedef typename std::vector<value_type>::const_iterator iterator;
  if (mode >= A.rank())
    BTAS_EXCEPTION("Cannot flatten along mode outside of A.rank()");
  Tensor X(A.extent(mode), A.range().area() / A.extent(mode));
  int indexi = 0, indexj = 0;
  auto ndim = A.rank();
  std::vector<int> J(ndim, 1);
  for (auto i = 0; i < ndim; ++i)
    if (i != mode)
      for (auto m = 0; m < i; ++m)
        if (m != mode)
          J[i] *= A.extent(m);

  iterator tensor_itr = A.begin();
  fill(A, 0, X, mode, indexi, indexj, J, tensor_itr);
  return X;
}

/// following the formula for flattening layed out by Kolda and Bader
/// http://epubs.siam.org/doi/pdf/10.1137/07070111X
/// Recursive method utilized by flatten, if you want to flatten a tensor
/// call flatten, not fill. 
template <typename Tensor, typename iterator>
void fill(Tensor &A, int depth, Tensor &X, int mode, int indexi, int indexj,
          std::vector<int> &J, iterator &tensor_itr) {
  auto ndim = A.rank();
  if (depth < ndim) {
    for (auto i = 0; i < A.extent(depth); ++i) {
      if (depth != mode) {
        indexj += i * J[depth]; // column matrix index
      } else {
        indexi = i; // row matrix index
      }
      fill(A, depth + 1, X, mode, indexi, indexj, J, tensor_itr);
      if (depth != mode)
        indexj -= i * J[depth];
    }
  } else {
    X(indexi, indexj) = *tensor_itr;
    tensor_itr++;
  }
}
} // namespace btas
#endif
