#ifndef BTAS_FLATTEN_H
#define BTAS_FLATTEN_H

namespace btas {

/// methods to produce to matricize an order-N tensor along the n-th fiber
/// \param[in] A The order-N tensor one wishes to flatten.
/// \param[in] mode The mode of \c A to be flattened, i.e.
/// \f[ A(I_1, I_2, I_3, ..., I_{mode}, ..., I_N) -> A(I_{mode}, J)\f]
/// where \f$J = I_1 * I_2 * ...I_{mode-1} * I_{mode+1} * ... * I_N.\f$
/// \return Matrix with dimension \f$(I_{mode}, J)\f$

template <typename Tensor> Tensor flatten(const Tensor &A, int mode) {
  
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

  auto tensor_itr = A.begin();

  // Fill X with the correct values
  fill(A, 0, X, mode, indexi, indexj, J, tensor_itr);

  // return the flattened matrix
  return X;
}

/// following the formula for flattening layed out by Kolda and Bader
/// <a href=http://epubs.siam.org/doi/pdf/10.1137/07070111X> See reference. </a>
/// Recursive method utilized by flatten.\n **Important** if you want to flatten a tensor
/// call flatten, not fill.

/// \param[in] A  The reference tensor to be flattened
/// \param[in] depth The recursion depth. Should not exceed the A.rank()
/// \param[in, out] X In: An empty matrix to be filled with correct
/// elements of \c A flattened on the \c mode fiber. Should be size \f$ (I_{mode}, J)\f$
/// Out: The flattened A matrix along the \c mode fiber \param[in]
/// mode The mode which A is to be flattened. \param[in] indexi The row index of
/// matrix X \param[in] indexj The column index of matrix X \param[in] J The
/// step size for the row dimension of X \param[in] tensor_itr An iterator of \c A.
/// The value of the iterator is placed in the correct position of X using
/// recursive calls of fill()m.

template <typename Tensor, typename iterator>
void fill(const Tensor &A, int depth, Tensor &X, int mode, int indexi, int indexj,
          const std::vector<int> &J, iterator &tensor_itr) {
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
