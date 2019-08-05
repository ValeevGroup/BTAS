#ifndef BTAS_GENERIC_CONV_BASE_CLASS
#define BTAS_GENERIC_CONV_BASE_CLASS
#include <btas/generic/dot_impl.h>
#include <vector>
#include "btas/varray/varray.h"

namespace btas {
  /**
    \brief Default class to deciding when the ALS problem is converged
    Instead of using the change in the loss function
    \f$ \Delta \| \mathcal{T} - \mathcal{\hat{T}} \| \leq \epsilon \f$
    where \f$ \mathcal{\hat{T}} = \sum_{r=1}^R a_1 \circ a_2 \circ \dots \circ a_N \f$
    check the difference in the sum of average elements in factor matrices
    \f$ \sum_n^{ndim} \frac{\|A^{i}_n - A^{i+1}_n\|}{dim(A^{i}_n} \leq \epsilon \f$
  **/
  template <typename Tensor>
  class NormCheck{

  public:
    /// constructor for the base convergence test object
    /// \param[in] tol tolerance for ALS convergence
    explicit NormCheck(double tol = 1e-3): tol_(tol){
    }

    ~NormCheck() = default;

    /// Function to check convergence of the ALS problem
    /// convergence when \f$ \sum_n^{ndim} \frac{\|A^{i}_n - A^{i+1}_n\|}{dim(A^{i}_n} \leq \epsilon \f$
    /// \param[in] btas_factors Current set of factor matrices
    bool operator () (const std::vector<Tensor> & btas_factors){
      auto ndim = btas_factors.size() - 1;
      if (prev.empty() || prev[0].size() != btas_factors[0].size()){
        prev.clear();
        for(int i = 0; i < ndim; ++i){
          prev.push_back(Tensor(btas_factors[i].range()));
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

      if(diff < this->tol_){
        return true;
      }
      return false;
    }

  private:
    double tol_;
    std::vector<Tensor> prev;     // Set of previous factor matrices
    int ndim;                     // Number of factor matrices
    int rank_;               // Rank of the CP problem
  };

} //namespace btas
#endif