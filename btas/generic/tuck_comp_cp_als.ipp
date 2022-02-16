//
// Created by Karl Pierce on 2/10/22.
//

#ifndef MPQC_TUCK_COMP_CP_ALS_IPP
#define MPQC_TUCK_COMP_CP_ALS_IPP

#include <btas/generic/cp_als.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace btas{
  template <typename Tensor, class ConvClass = NormCheck<Tensor> >
  class TUCKER_CP_ALS : public CP_ALS<Tensor, ConvClass> {
    using CP_ALS<Tensor, ConvClass>::tensor_ref;
    using typename CP_ALS<Tensor, ConvClass>::ind_t;
    using typename CP_ALS<Tensor, ConvClass>::ord_t;
    using CP_ALS<Tensor, ConvClass>::size;

   public:
    TUCKER_CP_ALS(Tensor & tensor, double epsilon_tucker) :CP_ALS<Tensor, ConvClass>(tensor), tcut_tucker(epsilon_tucker){
    }

   protected:
    std::vector<Tensor> tucker_factors;
    double tcut_tucker;

    void ALS(ind_t rank, ConvClass &converge_test, bool dir, int max_als, bool calculate_epsilon, double &epsilon,
                  bool &fast_pI) override{
      tucker_compression(tensor_ref, tcut_tucker, tucker_factors);
    }

  };
}//namespace btas

#endif  // MPQC_TUCK_COMP_CP_ALS_IPP
