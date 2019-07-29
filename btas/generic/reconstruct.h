//
// Created by Karl Pierce on 4/12/19.
//

#ifndef BTAS_GENERIC_RECONSTRUCT_H
#define BTAS_GENERIC_RECONSTRUCT_H

#include <btas/generic/scal_impl.h>

namespace btas {
  template <typename Tensor>
  Tensor reconstruct(std::vector<Tensor> & A, std::vector<int> dims_order) {
    if(A.size() - 1 != dims_order.size()){
      BTAS_EXCEPTION("A.size() - 1 != dims_order.size(), please verify that you have correctly assigned the "
                     "order of dimension reconstruction");
    }
    std::vector <size_t> dimensions;
    auto ndim = A.size() - 1;
    for (int i = 0; i < ndim; i++) {
      dimensions.push_back(A[dims_order[i]].extent(0));
    }
    auto rank = A[0].extent(1);
    for (int i = 0; i < rank; i++) {
      scal(A[dims_order[0]].extent(0), A[ndim](i), std::begin(A[dims_order[0]]) + i, rank);
    }

    // Make the Khatri-Rao product of all the factor matrices execpt the last dimension
    Tensor KRP = A[dims_order[0]];
    Tensor hold = A[dims_order[0]];
    for (int i = 1; i < A.size() - 2; i++) {
      khatri_rao_product(KRP, A[dims_order[i]], hold);
      KRP = hold;
    }

    // contract the rank dimension of the Khatri-Rao product with the rank dimension of
    // the last factor matrix. hold is now the reconstructed tensor
    hold = Tensor(KRP.extent(0), A[dims_order[ndim - 1]].extent(0));
    gemm(CblasNoTrans, CblasTrans, 1.0, KRP, A[dims_order[ndim - 1]], 0.0, hold);

    // resize the reconstructed tensor to the correct dimensions
    hold.resize(dimensions);

    // remove the scaling applied to the first factor matrix
    for (int i = 0; i < rank; i++) {
      scal(A[dims_order[0]].extent(0), 1 / A[ndim](i), std::begin(A[dims_order[0]]) + i, rank);
    }
    return hold;
  }
}
#endif //BTAS_GENERIC_RECONSTRUCT_H
