#ifndef BTAS_SWAP_H
#define BTAS_SWAP_H

#ifdef BTAS_HAS_INTEL_MKL

#include <vector>
#define MKL_Complex16 std::complex<double>
#include <mkl_trans.h>
#include <btas/error.h>
#include <btas/range_traits.h>

//***IMPORTANT***//
// do not use swap to first then use swap to back
// swap to first preserves order while swap to back does not
// If you use swap to first, to undo the transpositions
// make is_in_front = true and same goes for swap to back
// do not mix swap to first and swap to back

namespace btas {
/// Generalized implementation for mkl_?imatcopy
template<typename T_ , typename indx>
void imatcopy(char ordering, char trans, indx rows, indx cols, T_ *alpha, T_ *a, indx src_lda, indx dst_lda){
  if constexpr (is_complex_type_v<T_>){
    mkl_zimatcopy(ordering , trans , rows, cols, alpha , a, src_lda, dst_lda);
  }
  else{
    mkl_dimatcopy(ordering , trans , rows, cols, alpha , a, src_lda, dst_lda);
  }
}


/// Swaps the nth mode of an Nth order tensor to the front preserving the
/// order of the other modes. \n
/// swap_to_first(A, I3, false, false) =
/// A(I1, I2, I3, I4, I5) --> A(I3, I1, I2, I4, I5)

/// \param[in, out] A In: An order-N tensor.  Out: the order-N tensor with mode \c mode permuted.
/// \param[in] mode The mode of \c A one wishes to permute to the front.
/// \param[in] is_in_front \c Mode of \c A has already been permuted to the
/// front. Default = false. \param[in] for_ALS_update Different indexing is
/// required for the ALS, if making a general swap this should be false.

template<typename Tensor>
void swap_to_first(Tensor &A, size_t mode, bool is_in_front = false,
                   bool for_ALS_update = true) {
  using ind_t = typename Tensor::range_type::index_type::value_type;
  using ord_t = typename range_traits<typename Tensor::range_type>::ordinal_type;
  using dtype = typename Tensor::numeric_type;
  auto ndim = A.rank();
  // If the mode of interest is the the first mode you are done.
  if (mode > ndim) {
    BTAS_EXCEPTION("Mode index is greater than tensor rank");
  }
  if (mode == 0)
    return;

  // Build the resize vector for reference tensor to update dimensions
  std::vector<ind_t> aug_dims;
  dtype one_ {1.0}; dtype *one = &one_;
  ord_t size = A.range().area();
  for (size_t i = 0; i < ndim; i++) {
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

  ord_t rows = 1;
  ord_t cols = 1;
  ind_t step = 1;

  // The last mode is an easier swap, make all dimensions before last, row
  // dimension Last dimension is column dimension, then permute.
  if (mode == ndim - 1) {
    rows = (is_in_front) ? A.extent(0) : size / A.extent(mode);
    cols = (is_in_front) ? size / A.extent(0) : A.extent(mode);

    dtype *data_ptr = A.data();
    imatcopy('R', 'T', rows, cols, one, data_ptr, cols, rows);
  }

    // All other dimension not so easy all indices up to mode of interest row
    // dimension all othrs column dimension, then swap. After swapping, there are
    // row dimension many smaller tensors of size column dimension do row
  // dimension many swaps with inner row dimension = between the outer row
  // dimension and the last dimension and inner col dimension = last dimension,
  // now the mode of interest. Swapping the rows and columns back at the end
  // will preserve order of the dimensions.
  else {
    for (size_t i = 0; i <= mode; i++)
      rows *= A.extent(i);
    cols = size / rows;
    dtype *data_ptr = A.data();

    imatcopy('R', 'T', rows, cols, one, data_ptr, cols, rows);

    step = rows;
    ind_t in_rows = (is_in_front) ? A.extent(0) : rows / A.extent(mode);
    ind_t in_cols = (is_in_front) ? rows / A.extent(0) : A.extent(mode);

    for (ind_t i = 0; i < cols; i++) {
      data_ptr = A.data() + i * step;
      imatcopy('R', 'T', in_rows, in_cols, one, data_ptr, in_cols,
                    in_rows);
    }
    data_ptr = A.data();
    imatcopy('R', 'T', cols, rows, one, data_ptr, rows, cols);
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

  template<typename Tensor>
  void swap_to_back(Tensor &A, size_t mode, bool is_in_back = false) {
    using ind_t = typename Tensor::range_type::index_type::value_type;
    using ord_t = typename range_traits<typename Tensor::range_type>::ordinal_type;
    using dtype = typename Tensor::numeric_type;
    auto ndim = A.rank();

    if (mode > ndim)
      BTAS_EXCEPTION_MESSAGE(__FILE__, __LINE__,
                             "mode > A.rank(), mode out of range");
    if (mode == ndim - 1)
      return;

    ord_t rows = 1;
    ord_t cols = 1;

    // counts the modes up to and including the mode of interest, these are stored
    // in rows counts all the modes beyond the mode of interest, these are stored
    // as columns
    auto midpoint = (is_in_back) ? ndim - 1 - mode : mode + 1;
    std::vector<ind_t> aug_dims;
    for (size_t i = midpoint; i < ndim; i++) {
      aug_dims.push_back(A.extent(i));
      cols *= A.extent(i);
    }
    for (size_t i = 0; i < midpoint; i++) {
      aug_dims.push_back(A.extent(i));
      rows *= A.extent(i);
    }

    // Permutes the rows and columns
    double *data_ptr = A.data();
  imatcopy('R', 'T', rows, cols, 1.0, data_ptr, cols, rows);

  // resized to the new correct order.
  A.resize(aug_dims);
  return;
}

} // namespace btas

#endif //BTAS_HAS_INTEL_MKL

#endif // BTAS_SWAP_H
