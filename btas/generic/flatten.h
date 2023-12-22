#ifndef BTAS_FLATTEN_H
#define BTAS_FLATTEN_H

namespace btas {

/// methods to produce to matricize an order-N tensor along the n-th fiber
/// \param[in] A The order-N tensor one wishes to flatten.
/// \param[in] mode The mode of \c A to be flattened, i.e.
/// \f[ A(I_1, I_2, I_3, ..., I_{mode}, ..., I_N) -> A(I_{mode}, J)\f]
/// where \f$J = I_1 * I_2 * ...I_{mode-1} * I_{mode+1} * ... * I_N.\f$
/// \return Matrix with dimension \f$(I_{mode}, J)\f$
  template<typename Tensor>
  Tensor flatten(Tensor A, size_t mode) {
    using ord_t = typename range_traits<typename Tensor::range_type>::ordinal_type;
    using ind_t = typename Tensor::range_type::index_type::value_type;
    // We are going to first make the order N tensor into a order 3 tensor with
    // (modes before `mode`, `mode`, modes after `mode`

    auto dim_mode = A.extent(mode);
    Tensor flat(dim_mode, A.range().area() / dim_mode);
    size_t ndim = A.rank();
    ord_t dim1 = 1, dim3 = 1;
    for (ind_t i = 0; i < ndim; ++i) {
      if (i < mode)
        dim1 *= A.extent(i);
      else if (i > mode)
        dim3 *= A.extent(i);
    }
    
    A.resize(Range{Range1{dim1}, Range1{dim_mode}, Range1{dim3}});

    for (ord_t i = 0; i < dim1; ++i) {
      for (ind_t j = 0; j < dim_mode; ++j) {
        for (ord_t k = 0; k < dim3; ++k) {
          flat(j, i * dim3 + k) = A(i,j,k);
        }
      }
    }
    return flat;
}

} // namespace btas

#endif // BTAS_FLATTEN_H
