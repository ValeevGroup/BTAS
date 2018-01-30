#ifndef BTAS_GENERIC_RECONSTRUCT_H
#define BTAS_GENERIC_RECONSTRUCT_H

#include <algorithm>
#include <cstdlib>
#include <iostream>

#include <btas/btas.h>
#include "khatri_rao_product.h"

namespace btas {
  /// Function that takes the factor matrices from the CP decomposition and reconstructs the
  /// exact tensor
  /// \param[in] factors vector of factor matrices determined through cp-als.
  /// \returns The full reconstructed tensor constructed from the factor matrices
  template <typename Tensor>
  Tensor reconstruct(std::vector<Tensor> factors) {
    auto ndim = factors.size() - 1;

    // Find the dimensions of the reconstructed tensor
    std::vector<size_t> dimensions;
    for (int i = 0; i < ndim; i++) {
      dimensions.push_back(factors[i].extent(0));
    }

    // Scale the first factor matrix, this choice is arbitrary
    auto rank = factors[0].extent(1);
    for (int i = 0; i < rank; i++) {
      scal(factors[0].extent(0), factors[ndim](i), std::begin(factors[0]) + i, rank);
    }

    // Make the KRP of all the factor matrices execpt the last dimension
    Tensor KRP = factors[0];
    Tensor hold = factors[0];
    for (int i = 1; i < factors.size() - 2; i++) {
      khatri_rao_product(KRP, factors[i], hold);
      KRP = hold;
    }

    // contract the rank dimension of the Khatri-Rao product with the rank dimension of the last factor matrix
    // This is the reconstructed tensor
    hold = Tensor(KRP.extent(0), factors[ndim - 1].extent(0));
    gemm(CblasNoTrans, CblasTrans, 1.0, KRP, factors[ndim - 1], 0.0, hold);

    // resize the reconstructed tensor to the correct size
    hold.resize(dimensions);
    return hold;
  }
}  // namespace btas

#endif  // BTAS_GENERIC_RECONSTRUCT_H