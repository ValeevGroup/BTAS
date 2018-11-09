#ifndef BTAS_GENERIC_CONV_BASE_CLASS
#define BTAS_GENERIC_CONV_BASE_CLASS
#include <btas/generic/dot_impl.h>
#include <vector>
#include "btas/varray/varray.h"

namespace btas {
  template <typename tensor>
  class NORM_CHECK{

  public:
    /// constructor for the base convergence test object
    /// \param[in] tol tolerance for ALS convergence
    /// \param[in] rank Rank of the CP problem
    /// \param[in] elements A varray of the number of elements
    NORM_CHECK(double tol = 1e-3): converged_(false), tol_(tol){
    }

    ~NORM_CHECK() = default;

    /// Function to check convergence of the ALS problem
    /// convergence when \sum_n^{ndim} \|A^{i}_n - A^{i+1}_n\| \leq \epsilon
    /// \param[in] btas_factors Current set of factor matrices
    bool operator () (const std::vector<tensor> & btas_factors){
      auto ndim = btas_factors.size() - 1;
      if (prev.empty() || prev[0].size() != btas_factors[0].size()){
        prev.clear();
        for(int i = 0; i < ndim; ++i){
          prev.push_back(tensor(btas_factors[i].range()));
          prev[i].fill(0.0);
        }
      }

      auto diff = 0.0;
      rank_ = btas_factors[0].extent(1);
      for(int r = 0; r < ndim; ++r){
        auto elements = btas_factors[r].size();
        auto change = prev[r] - btas_factors[r];
        diff += std::sqrt(btas::dot(change, change)/elements);
        prev[r] = btas_factors[r];
      }

      std::cout << diff << std::endl;

      if(diff < this->tol_){
        this->converged_ = true;
      }
      return converged_;
    }

  private:
    bool converged_;
    double tol_;
    std::vector<tensor> prev;     // Set of previous factor matrices
    int ndim;                     // Number of factor matrices
    int rank_;               // Rank of the CP problem
  };

} //namespace btas
#endif