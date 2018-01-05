#ifndef BTAS_CORE_CONTRACT
#define BTAS_CORE_CONTRACT
#include "swap.h"
#include <btas/btas.h>
namespace btas {
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
#endif // BTAS_CORE_CONTRACT