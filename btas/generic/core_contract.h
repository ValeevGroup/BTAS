#ifndef BTAS_CORE_CONTRACT_H
#define BTAS_CORE_CONTRACT_H
#include "swap.h"
#include <btas/btas.h>
namespace btas {
  /// Function used by Tucker and Randomized compression.
  /// Takes an Nth order tensor swaps the mode of interest
  /// to the front and contracts it with a rank reducing 
  /// factor matrix. A is returned as a contracted tensor
template <typename Tensor>
void core_contract(Tensor &A, Tensor &Q, int mode, bool transpose = true) {
  auto ndim = A.rank();
  swap_to_first(A, mode, false, false);
  std::vector<int> temp_dims, A_indices, Q_indicies;

  // make size of contraction for contract algorithm
  temp_dims.push_back((transpose) ? Q.extent(1) : Q.extent(0));
  for (int i = 1; i < ndim; i++)
    temp_dims.push_back(A.extent(i));
  Tensor temp(Range{temp_dims});
  temp_dims.clear();

  // Make contraction indices
  Q_indicies.push_back((transpose) ? 0 : ndim);
  Q_indicies.push_back((transpose) ? ndim : 0);
  temp_dims.push_back(ndim);
  A_indices.push_back(0);
  for (int i = 1; i < ndim; i++) {
    A_indices.push_back(i);
    temp_dims.push_back(i);
  }
  contract(1.0, Q, Q_indicies, A, A_indices, 0.0, temp, temp_dims);
  A = temp;
  swap_to_first(A, mode, true, false);
}
} // namespace btas
#endif // BTAS_CORE_CONTRACT_H