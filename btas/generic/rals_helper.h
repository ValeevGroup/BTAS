//
// Created by Karl Pierce on 11/3/18.
//

#ifndef BTAS_RALS_HELPER_H
#define BTAS_RALS_HELPER_H

namespace btas{

  template <typename Tensor>
  class RALSHelper{
  public:
    RALSHelper() = default;

    ~RALSHelper() = default;

    RALSHelper(std::vector<Tensor> prev): prev_(prev){
    }

    double operator() (int mode, const Tensor& An){
      auto change = An - prev_[mode];
      double s = std::sqrt(dot(change, change));
      s /= std::sqrt(dot(An, An));

      prev_[mode] = An;
      return s;
    }

  private:
    std::vector<Tensor> prev_;
  };

}

#endif //BTAS_RALS_HELPER_H
