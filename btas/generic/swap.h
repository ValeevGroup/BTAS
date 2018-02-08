#ifndef BTAS_SWAP_H
#define BTAS_SWAP_H

#ifdef _HAS_INTEL_MKL

#include <btas/btas.h>
#include <mkl_trans.h>

#include <vector>

//***IMPORTANT***//
// do not use swap to first then use swap to back
// swap to first preserves order while swap to back does not
// If you use swap to first, to undo the transpositions
// make is_in_front = true and same goes for swap to back
// do not mix swap to first and swap to back

namespace btas {

/// Swaps the nth mode of an Nth order tensor to the front preserving the
/// order of the other modes. \n
/// swap_to_first(A, I3, false, false) =
/// A(I1, I2, I3, I4, I5) --> A(I3, I1, I2, I4, I5)

/// \param[in, out] A In: An order-N tensor.  Out: the order-N tensor with mode \c mode permuted.
/// \param[in] mode The mode of \c A one wishes to permute to the front.
/// \param[in] is_in_front \c Mode of \c A has already been permuted to the
/// front. Default = false. \param[in] for_ALS_update Different indexing is
/// required for the ALS, if making a general swap this should be false.

template <typename Tensor>
void swap_to_first(Tensor &A, int mode, bool is_in_front = false,
                   bool for_ALS_update = true) {
  // If the mode of interest is the the first mode you are done.
  if (mode == 0)
    return;

  // Build the resize vector for reference tensor to update dimensions
  std::vector<int> aug_dims;
  auto size = A.range().area();
  for (int i = 0; i < A.rank(); i++) {
    aug_dims.push_back(A.extent(i));
  }

  // Special indexing for ALS update
  if (for_ALS_update) {
    auto temp = aug_dims[0];
    aug_dims[0] = aug_dims[mode];
    aug_dims[mode] = temp;
  }

  // Order preserving swap of indices.
  else {
    auto temp = (is_in_front) ? aug_dims[0] : aug_dims[mode];
    auto erase = (is_in_front) ? aug_dims.begin() : aug_dims.begin() + mode;
    auto begin = (is_in_front) ? aug_dims.begin() + mode : aug_dims.begin();
    aug_dims.erase(erase);
    aug_dims.insert(begin, temp);
  }

  size_t rows = 1;
  size_t cols = 1;
  auto step = 1;

  // The last mode is an easier swap, make all dimensions before last, row
  // dimension Last dimension is column dimension, then permute.
  if (mode == A.rank() - 1) {
    rows = (is_in_front) ? A.extent(0) : size / A.extent(mode);
    cols = (is_in_front) ? size / A.extent(0) : A.extent(mode);

    double *data_ptr = A.data();
    mkl_dimatcopy('R', 'T', rows, cols, 1.0, data_ptr, cols, rows);
  }

  // All other dimension not so easy all indices up to mode of interest row
  // dimension all othrs column dimension, then swap. After swapping, there are
  // row dimension many smaller tensors of size column dimension do row
  // dimension many swaps with inner row dimension = between the outer row
  // dimension and the last dimension and inner col dimension = last dimension,
  // now the mode of interest. Swapping the rows and columns back at the end
  // will preserve order of the dimensions.
  else {
    for (int i = 0; i <= mode; i++)
      rows *= A.extent(i);
    cols = size / rows;
    double *data_ptr = A.data();

    mkl_dimatcopy('R', 'T', rows, cols, 1.0, data_ptr, cols, rows);

    step = rows;
    size_t in_rows = (is_in_front) ? A.extent(0) : rows / A.extent(mode);
    size_t in_cols = (is_in_front) ? rows / A.extent(0) : A.extent(mode);

    for (int i = 0; i < cols; i++) {
      data_ptr = A.data() + i * step;
      mkl_dimatcopy('R', 'T', in_rows, in_cols, 1.0, data_ptr, in_cols,
                    in_rows);
    }
    data_ptr = A.data();
    mkl_dimatcopy('R', 'T', cols, rows, 1.0, data_ptr, rows, cols);
  }
  A.resize(aug_dims);
}

/// Swaps the nth order of an Nth order tensor to the end.
/// Does not preserve order.\n
/// swap_to_back(T, I2, false) =
/// T(I1, I2, I3) --> T(I3, I1, I2)

/// \param[in, out] A In: An order-N tensor.  Out: the order-N tensor with mode \c mode permuted.
/// \param[in] mode The mode of \c A one wishes to permute to the back.
/// \param[in] is_in_back \c Mode of \c A has already been permuted to the
/// back. Default = false.

template <typename Tensor>
void swap_to_back(Tensor &A, int mode, bool is_in_back = false) {
  if (mode > A.rank())
    BTAS_EXCEPTION_MESSAGE(__FILE__, __LINE__,
                           "mode > A.rank(), mode out of range");
  if (mode == A.rank() - 1)
    return;

  size_t rows = 1;
  size_t cols = 1;

  // counts the modes up to and including the mode of interest, these are stored
  // in rows counts all the modes beyond the mode of interest, these are stored
  // as columns
  auto ndim = A.rank();
  auto midpoint = (is_in_back) ? ndim - 1 - mode : mode + 1;
  std::vector<size_t> aug_dims;
  for (int i = midpoint; i < ndim; i++) {
    aug_dims.push_back(A.extent(i));
    cols *= A.extent(i);
  }
  for (int i = 0; i < midpoint; i++) {
    aug_dims.push_back(A.extent(i));
    rows *= A.extent(i);
  }

  // Permutes the rows and columns
  double *data_ptr = A.data();
  mkl_dimatcopy('R', 'T', rows, cols, 1.0, data_ptr, cols, rows);

  // resized to the new correct order.
  A.resize(aug_dims);
  return;
}

} // namespace btas

#endif //_HAS_INTEL_MKL

#endif // BTAS_SWAP_H
