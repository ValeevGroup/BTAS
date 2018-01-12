#ifndef BTAS_CORE_CONTRACT_H
#define BTAS_CORE_CONTRACT_H

#include "swap.h"
#include <btas/btas.h>

namespace btas {

/// Function used by Tucker and Randomized compression.
/// Takes an order-N tensor swaps the mode of interest, \c mode,
/// to the front and contracts it with a rank reducing
/// factor matrix, \c Q, discovered by Tucker or Randomized decomposition methods.

/// \param[in, out] A The order-N tensor to be contracted with Q
/// \param[in] Q Factor matrix to be contracted with mode \c mode of \c A
/// \param[in] mode Mode of A to be contracted with Q
/// \param[in] transpose Is Q transposed in the matrix/tensor contraction?
/// Default value = true.

template <typename Tensor>
void core_contract(Tensor &A, const Tensor &Q, int mode, bool transpose = true) {

  auto ndim = A.rank();

  // Reorder A so contraction of nth mode will be in the front
  swap_to_first(A, mode, false, false);

  std::vector<int> temp_dims, A_indices, Q_indicies;

  // Allocate the appropriate memory for the resulting tensor
  temp_dims.push_back((transpose) ? Q.extent(1) : Q.extent(0));
  for (int i = 1; i < ndim; i++)
    temp_dims.push_back(A.extent(i));
  Tensor temp(Range{temp_dims});
  temp_dims.clear();

  // Build index vectors to contract over the first index of A and
  // The correct index of Q depending if transpose == true.
  Q_indicies.push_back((transpose) ? 0 : ndim);
  Q_indicies.push_back((transpose) ? ndim : 0);
  temp_dims.push_back(ndim);
  A_indices.push_back(0);
  for (int i = 1; i < ndim; i++) {
    A_indices.push_back(i);
    temp_dims.push_back(i);
  }

  // contract Q^(T?) (x)_n A = temp;
  contract(1.0, Q, Q_indicies, A, A_indices, 0.0, temp, temp_dims);

  // A is now the (smaller) contracted tensor temp
  A = temp;

  // Reorder A as it was before contraction
  swap_to_first(A, mode, true, false);
}

} // namespace btas

#endif // BTAS_CORE_CONTRACT_H
