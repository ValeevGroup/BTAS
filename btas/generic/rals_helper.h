//
// Created by Karl Pierce on 11/3/18.
//

#ifndef BTAS_RALS_HELPER_H
#define BTAS_RALS_HELPER_H

namespace btas{
/**
    \brief A helper function for the RALS solver
    Stores a previous iteration of factor
    matrices to compute the nth iteration step size
    see https://doi.org/10.1063/1.4977994 for more details
 **/
  template <typename Tensor>
  class RALSHelper{
  public:
    RALSHelper() = default;

    ~RALSHelper() = default;

    /// constructor of the helper function
    /// \param[in] prev an initial set of
    /// normalized factor matrices
    RALSHelper(std::vector<Tensor> prev): prev_(prev){
    }

    /// Operator to compute the nth iteration step size
    /// \param[in] mode which mode of the actual tensor
    /// is being updated
    /// \param[in] An the updated factor matrix
    double operator() (int mode, const Tensor& An){
      auto change = An - prev_[mode];
      double s = std::sqrt(dot(change, change));
      s /= std::sqrt(dot(An, An));

      prev_[mode] = An;
      return s;
    }

  private:
    std::vector<Tensor> prev_;  // stores a set normalized of factor matrices
  };

}

#endif //BTAS_RALS_HELPER_H
