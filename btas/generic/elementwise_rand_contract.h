//
// Created by Karl Pierce on 3/17/22.
//

#ifndef BTAS_GENERIC_ELEMENTWISE_RAND_CONTRACT_H
#define BTAS_GENERIC_ELEMENTWISE_RAND_CONTRACT_H

#include <random>
#include <algorithm>

namespace btas{
  // This compute \alpha A(i, j, r) * B(j, r) + \beta C(i,r) = C(i,r)
  template<
      typename _T,
      class _TensorA, class _TensorB, class _TensorC,
      class = typename std::enable_if<
          is_boxtensor<_TensorA>::value &
          is_boxtensor<_TensorB>::value &
          is_boxtensor<_TensorC>::value &
          std::is_same<typename _TensorA::value_type, typename _TensorB::value_type>::value &
          std::is_same<typename _TensorA::value_type, typename _TensorC::value_type>::value
          >::type
      >
  void middle_rand_contract(_T alpha, const _TensorA& A, const _TensorB& B, _T beta, _TensorC& C){
    static_assert(boxtensor_storage_order<_TensorA>::value == boxtensor_storage_order<_TensorC>::value &&
                      boxtensor_storage_order<_TensorB>::value == boxtensor_storage_order<_TensorC>::value,
                  "btas::middle_contract does not support mixed storage order");
    static_assert(boxtensor_storage_order<_TensorC>::value != boxtensor_storage_order<_TensorC>::other,
                  "btas::middle_contract does not support non-major storage order");

    typedef typename _TensorA::value_type value_type;
    using ind_t = typename _TensorA::range_type::index_type::value_type;
    using ord_t = typename range_traits<typename _TensorA::range_type>::ordinal_type;

    BTAS_ASSERT(A.rank() == 3)
    BTAS_ASSERT(B.rank() == 2)
    BTAS_ASSERT(A.extent(1) == B.extent(0))
    BTAS_ASSERT(A.extent(2) == B.extent(1))

    if(!C.empty()){
      BTAS_ASSERT(C.rank() == 2);
      BTAS_ASSERT(C.extent(0) == A.extent(0))
      BTAS_ASSERT(C.extent(1) == A.extent(2))
    } else{
      C = _TensorC(A.extent(0), A.extent(2));
      NumericType<value_type>::fill(std::begin(C), std::end(C), NumericType<value_type>::zero());
    }

    std::mt19937 generator(random_seed_accessor());
    std::uniform_real_distribution<> distribution(0.0, B.extent(0));
    std::vector<size_t> j_modes;
    j_modes.reserve(A.extent(1));
    for(auto i = 0; i < A.extent(1); ++i) j_modes.emplace_back(i);
    ind_t idx1 = A.extent(0), idx2 = A.extent(1), rank = A.extent(2);

    ord_t i_times_rank = 0, i_times_rank_idx2 = 0;
    for (ind_t i = 0; i < idx1; ++i,i_times_rank += rank) {
      std::shuffle(j_modes.begin(), j_modes.end(), generator);
      auto *C_ptr = C.data() + i_times_rank;
      ord_t j_times_rank = 0;
      auto norm_C = 0.0;
      double M = 0., S = 0., old_S = 0., old_M = 0.;
      //std::cout << i << std::endl;
      for (auto j = 0; j < idx2; ++j) {
        auto rand_j = j_modes[j];
        j_times_rank =  rand_j* rank;
        i_times_rank_idx2 = i * idx2 * rank + rand_j * rank;
        const auto *A_ptr = A.data() + i_times_rank_idx2 + j_times_rank;
        const auto *B_ptr = B.data() + j_times_rank;
        auto x = 0.0;
        for (ind_t r = 0; r < rank; r++) {
//          *(C_ptr + r) += alpha * (*(A_ptr + r) * *(B_ptr + r))
//                          + beta * *(C_ptr + r);
          C(i,r) += A(i,rand_j,r) * B(rand_j,r);
          x += C(i,r);
        }
        old_M = M;
        old_S = S;
        M += (x - M)/(j + 1);
        S += (x - M) * (x - old_M);
//        if(j>1) {
//          std::cout << sqrt(S / j / (j - 1)) << std::endl;
//        }
        if(j > 1  &&  sqrt(S / j / (j-1))< 5e-4) break;
        //std::cout << "Norm_C : " << sqrt(norm_C) << std::endl;
      }
      //i_times_rank_idx2 += j_times_rank;
    }
  }
} // namespace btas
#endif  // BTAS_GENERIC_ELEMENTWISE_RAND_CONTRACT_H
