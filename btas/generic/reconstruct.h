//
// Created by Karl Pierce on 4/12/19.
//

#ifndef BTAS_GENERIC_RECONSTRUCT_H
#define BTAS_GENERIC_RECONSTRUCT_H

#include <btas/generic/scal_impl.h>

namespace btas {
  template<typename Tensor>
  Tensor reconstruct(std::vector<Tensor> &A, std::vector<size_t> dims_order, Tensor lambda = Tensor()) {
    using ind_t = typename Tensor::range_type::index_type::value_type;
    using ord_t = typename range_traits<typename Tensor::range_type>::ordinal_type;

    if (A.size() - 1 != dims_order.size() &&  lambda.empty()) {
      BTAS_EXCEPTION("A.size() - 1 != dims_order.size(), please verify that you have correctly assigned the "
                     "order of dimension reconstruction");
    }
    std::vector<ord_t> dimensions;
    // ndim is the size of dim_orders, this allows you
    // to resize a subtensor in the factors set A
    size_t ndim = dims_order.size();
    for (size_t i = 0; i < ndim; i++) {
      dimensions.push_back(A[dims_order[i]].extent(0));
    }
    ind_t rank = A[0].extent(1);
    lambda = (lambda.empty() ? A[ndim] : lambda);
    auto lam_ptr = lambda.data();
    for (ind_t i = 0; i < rank; i++) {
      scal(A[dims_order[0]].extent(0), *(lam_ptr + i), std::begin(A[dims_order[0]]) + i, rank);
    }

    // Make the Khatri-Rao product of all the factor matrices execpt the last dimension
    Tensor KRP = A[dims_order[0]];
    Tensor hold = A[dims_order[0]];
    for (size_t i = 1; i < ndim - 1; i++) {
      khatri_rao_product(KRP, A[dims_order[i]], hold);
      KRP = hold;
    }

    // contract the rank dimension of the Khatri-Rao product with the rank dimension of
    // the last factor matrix. hold is now the reconstructed tensor
    hold = Tensor(KRP.extent(0), A[dims_order[ndim - 1]].extent(0));
    gemm(blas::Op::NoTrans, blas::Op::Trans, 1.0, KRP, A[dims_order[ndim - 1]], 0.0, hold);

    // resize the reconstructed tensor to the correct dimensions
    hold.resize(dimensions);

    // remove the scaling applied to the first factor matrix
    // if the value of lambda is very small, don't invert it.
    for (ind_t i = 0; i < rank; i++) {
      auto val = *(lam_ptr + i);
      scal(A[dims_order[0]].extent(0), (abs(val) > 1e-12 ? 1.0/val : 1.0), std::begin(A[dims_order[0]]) + i, rank);
    }
    return hold;
  }
}
#endif //BTAS_GENERIC_RECONSTRUCT_H
