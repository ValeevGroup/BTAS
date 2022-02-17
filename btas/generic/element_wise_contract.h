//
// Created by Karl Pierce on 2/17/22.
//

#ifndef BTAS_GENERIC_ELEMENT_WISE_CONTRACT_H
#define BTAS_GENERIC_ELEMENT_WISE_CONTRACT_H

namespace btas{

  // This compute \alpha A(i, j, r) * B(j, r) + \beta C(i,r) = C(i,r)
  template <typename Tensor>
  void middle_contract(double alpha, const Tensor& A, const Tensor& B, double beta, Tensor& C){
    using ind_t = typename Tensor::range_type::index_type::value_type;
    using ord_t = typename range_traits<typename Tensor::range_type>::ordinal_type;

    BTAS_ASSERT(A.rank() == 3)
    BTAS_ASSERT(B.rank() == 2)
    BTAS_ASSERT(A.extent(1) == B.extent(0))
    BTAS_ASSERT(A.extent(2) == B.extent(1))

    if(!C.empty()){
      BTAS_ASSERT(C.rank() == 2);
      BTAS_ASSERT(C.extent(0) == A.extent(0))
      BTAS_ASSERT(C.extent(1) == A.extent(2))
    } else{
      C = Tensor(A.extent(0), A.extent(2));
      C.fill(0.0);
    }
    ind_t idx1 = A.extent(0), idx2 = A.extent(1), rank = A.extent(2);

    ord_t i_times_rank = 0, i_times_rank_idx2 = 0;
    for (ind_t i = 0; i < idx1; i++, i_times_rank += rank) {
      auto *C_ptr = C.data() + i_times_rank;
      ord_t j_times_rank = 0;
      for (ind_t j = 0; j < idx2; j++, j_times_rank += rank) {
        const auto *A_ptr = A.data() + i_times_rank_idx2 + j_times_rank;
        const auto *B_ptr = B.data() + j_times_rank;
        for (ind_t r = 0; r < rank; r++) {
          *(C_ptr + r) += alpha * (*(A_ptr + r) * *(B_ptr + r))
                          + beta * *(C_ptr + r);
        }
      }
      i_times_rank_idx2 += j_times_rank;
    }
  }

  // this does the elementwise contraction \alpha A(i,j,k,r) * B(j, r) + \beta C(i,k,r) = C(i,k,r)
  template <typename Tensor>
  void middle_contract_with_pseudorank(double alpha, const Tensor & A,
                                       const Tensor& B, double beta, Tensor& C){
    using ind_t = typename Tensor::range_type::index_type::value_type;
    using ord_t = typename range_traits<typename Tensor::range_type>::ordinal_type;

    BTAS_ASSERT(A.rank() == 3)
    BTAS_ASSERT(B.rank() == 2)
    BTAS_ASSERT(A.extent(1) == B.extent(0))
    ind_t rank = B.extent(1),
          idx3 = A.extent(2) / rank;
    BTAS_ASSERT(A.extent(2) / idx3 == B.extent(1));

    if(!C.empty()){
      BTAS_ASSERT(C.rank() == 2);
      BTAS_ASSERT(C.extent(0) == A.extent(0))
      BTAS_ASSERT(C.extent(1) == A.extent(2))
    } else{
      C = Tensor(A.extent(0), A.extent(2));
      C.fill(0.0);
    }

    ind_t idx1 = A.extent(0), idx2 = A.extent(1), pseudo_rank = A.extent(2);
    ord_t i_times_rank = 0, i_times_rank_idx2 = 0;
    for (ind_t i = 0; i < idx1; ++i, i_times_rank += pseudo_rank) {
      auto *C_ptr = C.data() + i_times_rank;
      ord_t j_times_prank = 0, j_times_rank = 0;
      for (ind_t j = 0; j < idx2; ++j, j_times_prank += pseudo_rank, j_times_rank += rank) {
        const auto *A_ptr = A.data() + i_times_rank_idx2 + j_times_prank;
        const auto *B_ptr = B.data() + j_times_rank;
        ord_t k_times_rank = 0;
        for (ind_t k = 0; k < idx3; ++k, k_times_rank += rank) {
          for (ind_t r = 0; r < rank; ++r) {
            *(C_ptr + k_times_rank + r) += alpha * ( *(A_ptr + k_times_rank + r) * *(B_ptr + r))
                                           + beta * *(C_ptr + k_times_rank + r);
          }
        }
      }
      i_times_rank_idx2 += j_times_prank;
    }

  }

  // this computes \alpha A(i,j,r) * B(i,r) + \beta C(j,r) = C(j,r)
  template<typename Tensor>
  void front_contract(double alpha, const Tensor & A, const Tensor & B, double beta, Tensor & C){
    using ind_t = typename Tensor::range_type::index_type::value_type;
    using ord_t = typename range_traits<typename Tensor::range_type>::ordinal_type;

    BTAS_ASSERT(A.rank() == 3)
    BTAS_ASSERT(B.rank() == 2)
    BTAS_ASSERT(A.extent(0) == B.extent(0))
    BTAS_ASSERT(A.extent(2) == B.extent(1))

    if(!C.empty()){
      BTAS_ASSERT(C.rank() == 2);
      BTAS_ASSERT(C.extent(0) == A.extent(1))
      BTAS_ASSERT(C.extent(1) == A.extent(2))
    } else{
      C = Tensor(A.extent(0), A.extent(1));
      C.fill(0.0);
    }

    ind_t idx1 = A.extent(0), idx2 =  A.extent(1), rank = A.extent(2);
    ord_t i_times_rank = 0, i_times_rank_idx2 = 0;
    for (ind_t i = 0; i < idx1; i++, i_times_rank += rank) {
      const auto *B_ptr = B.data() + i_times_rank;
      ord_t j_times_rank = 0;
      for (ind_t j = 0; j < idx2; j++, j_times_rank += rank) {
        const auto *A_ptr = A.data() + i_times_rank_idx2 + j_times_rank;
        auto *C_ptr = C.data() + j_times_rank;
        for (ind_t r = 0; r < rank; r++) {
          *(C_ptr + r) += *(B_ptr + r) * *(A_ptr + r);
        }
      }
      i_times_rank_idx2 += j_times_rank;
    }
  }
}
#endif  // BTAS_GENERIC_ELEMENT_WISE_CONTRACT_H
