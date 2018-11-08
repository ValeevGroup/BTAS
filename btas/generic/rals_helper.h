//
// Created by Karl Pierce on 11/3/18.
//

#ifndef BTAS_RALS_HELPER_H
#define BTAS_RALS_HELPER_H
namespace btas{
  template <typename tensor>
  class RALS_HELPER{
  public:
    RALS_HELPER() = default;

    ~RALS_HELPER() = default;

    RALS_HELPER(std::vector<tensor> prev): prev_(prev){
    }

    double operator() (int mode, const tensor& An){
      auto change = An - prev_[mode];
      double s = std::sqrt(dot(change, change));
      s /= std::sqrt(dot(An, An));

      prev_[mode] = An;
      return s;
    }

  private:
    std::vector<tensor> prev_;
  };
}
#endif //BTAS_RALS_HELPER_H
