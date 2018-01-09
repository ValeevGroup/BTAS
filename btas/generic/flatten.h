#ifndef BTAS_FLATTEN_H
#define BTAS_FLATTEN_H

namespace btas {

/// methods to produce to matricize a tensor along the n-th fiber
/// \param[in] A Nth order tensor one wishes to flatten.
/// \param[in] mode The mode of A to be flattened, i.e. \n
/// A(I_1, I_2, I_3, ..., I_mode, ..., I_N) -> A(I_mode, J)\n
/// where J = I_1 * I_2 * ...I_mode-1 * I_mode+1 * ... * I_N.
/// \return Flattened matrix with dimension (I_mode, J)

template <typename Tensor> Tensor flatten(Tensor &A, int mode) {
  using value_type = typename Tensor::value_type;
  typedef typename std::vector<value_type>::const_iterator iterator;
  
  if (mode >= A.rank())
    BTAS_EXCEPTION("Cannot flatten along mode outside of A.rank()");
  
  // make X the correct size
  Tensor X(A.extent(mode), A.range().area() / A.extent(mode));

  int indexi = 0, indexj = 0;
  auto ndim = A.rank();
  // J is the new step size found by removing the mode of interest
  std::vector<int> J(ndim, 1);
  for (auto i = 0; i < ndim; ++i)
    if (i != mode)
      for (auto m = 0; m < i; ++m)
        if (m != mode)
          J[i] *= A.extent(m);

  iterator tensor_itr = A.begin();

  // Fill X with the correct values
  fill(A, 0, X, mode, indexi, indexj, J, tensor_itr);

  // return the flattened matrix
  return X;
}

/// following the formula for flattening layed out by Kolda and Bader
/// http://epubs.siam.org/doi/pdf/10.1137/07070111X
/// Recursive method utilized by flatten, if you want to flatten a tensor
/// call flatten, not fill.

/// \param[in] A The reference tensor which is being flattened
/// \param[in] depth The recursion depth should not exceed the A.rank()
/// \param[in, out] X In: The flattened matrix to be filled with correct
/// elements of A. Out: The flattened A matrix along the mode-th mode \param[in]
/// mode The mode which A is being flattened. \param[in] indexi The row index of
/// matrix X \param[in] indexj The column index of matrix X \param[in] J The
/// step size for the row dimension of X \param[in] tensor_itr An iterator of A,
/// the value of the iterator is placed in the correct location in X using
/// recursive function calls.

template <typename Tensor, typename iterator>
void fill(Tensor &A, int depth, Tensor &X, int mode, int indexi, int indexj,
          std::vector<int> &J, iterator &tensor_itr) {
  auto ndim = A.rank();
  if (depth < ndim) {
    
    // Creates a for loop based on the number of modes A has
    for (auto i = 0; i < A.extent(depth); ++i) {
      
      // use the for loop to find the column dimension index
      if (depth != mode) {
        indexj += i * J[depth]; // column matrix index
      }
      
      // if this depth is the mode being flattened use the for loop to find the
      // row dimension
      else {
        indexi = i; // row matrix index
      }

      fill(A, depth + 1, X, mode, indexi, indexj, J, tensor_itr);

      // remove the indexing from earlier in this loop.
      if (depth != mode)
        indexj -= i * J[depth];
    }
  }
  
  // When depth steps out of the number of dimensions, set X to be the correct
  // value from the iterator then increment the iterator.
  else {
    X(indexi, indexj) = *tensor_itr;
    tensor_itr++;
  }
}

} // namespace btas

#endif // BTAS_FLATTEN_H
