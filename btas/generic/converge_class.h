#ifndef BTAS_GENERIC_CONV_BASE_CLASS
#define BTAS_GENERIC_CONV_BASE_CLASS

#include <vector>

#include <btas/generic/dot_impl.h>
#include <btas/varray/varray.h>

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
  class NormCheck {
    using ind_t = long;
    using ord_t = typename range_traits<typename Tensor::range_type>::ordinal_type;

  public:
    /// constructor for the base convergence test object
    /// \param[in] tol tolerance for ALS convergence
    explicit NormCheck(double tol = 1e-3) : tol_(tol) {
    }

    ~NormCheck() = default;

    /// Function to check convergence of the ALS problem
    /// convergence when \f$ \sum_n^{ndim} \frac{\|A^{i}_n - A^{i+1}_n\|}{dim(A^{i}_n} \leq \epsilon \f$
    /// \param[in] btas_factors Current set of factor matrices
    bool operator () (const std::vector<Tensor> & btas_factors){
      size_t ndim = btas_factors.size() - 1;
      if (prev.empty() || prev[0].size() != btas_factors[0].size()) {
        prev.clear();
        for (size_t i = 0; i < ndim; ++i) {
          prev.push_back(Tensor(btas_factors[i].range()));
          prev[i].fill(0.0);
        }
      }

      auto diff = 0.0;
      rank_ = btas_factors[0].extent(1);
      for (size_t r = 0; r < ndim; ++r) {
        ord_t elements = btas_factors[r].size();
        auto change = prev[r] - btas_factors[r];
        diff += std::sqrt(btas::dot(change, change) / elements);
        prev[r] = btas_factors[r];
      }

      if (diff < this->tol_) {
        return true;
      }
      return false;
    }

  private:
    double tol_;
    std::vector<Tensor> prev;     // Set of previous factor matrices
    size_t ndim;                     // Number of factor matrices
    ind_t rank_;               // Rank of the CP problem
  };

  /**
   \brief Class used to decide when ALS problem is converged
   The "fit" is defined as \f$ 1 - \frac{\|X-full(M)\|}{\|X\|} \leq \epsilon\f$
   where X is the exact tensor and M is the reconstructed CP tensor.
   This fit is loosely the proportion of the data described by the
   CP model, i.e., a fit of 1 is perfect.
   **/
  template<typename Tensor>
  class FitCheck{
  public:
    using ind_t = long;
    using ord_t = typename range_traits<typename Tensor::range_type>::ordinal_type;

    /// constructor for the base convergence test object
    /// \param[in] tol tolerance for ALS convergence
    explicit FitCheck(double tol = 1e-4): tol_(tol){
    }

    ~FitCheck() = default;

    /// Function to check convergence of the ALS problem
    /// convergence when \f$ \|T - \hat{T}^{i+1}_n\|}{dim(A^{i}_n} \leq \epsilon \f$
    /// \param[in] btas_factors Current set of factor matrices
    bool operator()(const std::vector<Tensor> &btas_factors) {
      if (normT_ < 0) BTAS_EXCEPTION("One must set the norm of the reference tensor");
      auto n = btas_factors.size() - 2;
      ord_t size = btas_factors[n].size();
      ind_t rank = btas_factors[n].extent(1);
      Tensor temp(btas_factors[n + 1].range());
      temp.fill(0.0);
      auto *ptr_A = btas_factors[n].data();
      auto *ptr_MtKRP = MtKRP_.data();
      for (ord_t i = 0; i < size; ++i) {
        *(ptr_MtKRP + i) *= *(ptr_A + i);
      }

      auto * ptr_temp = temp.data();
      for (ord_t i = 0; i < size; ++i) {
        *(ptr_temp + i % rank) += *(ptr_MtKRP + i);
      }

      size = temp.size();
      ptr_A = btas_factors[n+1].data();
      double iprod = 0.0;
      for (ord_t i = 0; i < size; ++i) {
        iprod += *(ptr_temp + i) * *(ptr_A + i);
      }

      double normFactors = norm(btas_factors);
      double normResidual = sqrt(abs(normT_ * normT_ + normFactors * normFactors - 2 * iprod));
      double fit = 1 - (normResidual / normT_);

      double fitChange = abs(fitOld_ - fit);
      fitOld_ = fit;
      if(verbose_) {
        std::cout << fit << "\t" << fitChange << std::endl;
      }
      if(fitChange < tol_) {
        converged_num++;
        if(converged_num == 2){
          iter_ = 0;
          converged_num = 0;
          final_fit_ = fitOld_;
          fitOld_ = -1.0;
          return true;
        }
      }

      ++iter_;
      return false;
    }

    void set_norm(double normT){
      normT_ = normT;
    }

    void set_MtKRP(Tensor & MtKRP){
      MtKRP_ = MtKRP;
    }

    double get_fit(){
      return final_fit_;
    }

    void verbose(bool verb) {
      verbose_ = verb;
    }

  private:
    double tol_;
    double fitOld_ = -1.0;
    double normT_ = -1.0;
    double final_fit_ = 0.0;
    size_t iter_ = 0;
    size_t converged_num = 0;
    Tensor MtKRP_;
    bool verbose_ = false;

    double norm(const std::vector<Tensor> &btas_factors) {
      ind_t rank = btas_factors[0].extent(1), one = 1.0;
      auto n = btas_factors.size() - 1;
      Tensor coeffMat(rank, rank);
      auto &temp = btas_factors[n];
      ger(1.0, temp, temp, coeffMat);

      //temp.resize(Range{Range1{rank}, Range1{one}});
      //gemm(CblasNoTrans, CblasTrans, 1.0, temp, temp, 0.0, coeffMat);

      auto rank2 = rank * (ord_t) rank;
      for (size_t i = 0; i < n; ++i) {
        Tensor temp(rank, rank);
        gemm(CblasTrans, CblasNoTrans, 1.0, btas_factors[i], btas_factors[i], 0.0, temp);
        auto *ptr_coeff = coeffMat.data();
        auto *ptr_temp = temp.data();
        for (ord_t j = 0; j < rank2; ++j) {
          *(ptr_coeff + j) *= *(ptr_temp + j);
        }
      }

      auto nrm = 0.0;
      for(auto & i: coeffMat){
        nrm += i;
      }
      return sqrt(abs(nrm));
    }
  };

  /**
   \brief Class used to decide when ALS problem is converged
   The "fit" is defined as \f$ 1 - \frac{\|X_1-full(M_1)\|}{\|X_1\|} -
   \frac{\|X_2-full(M_2)\|}{\|X_2\|}\leq \epsilon\f$
   where \f$ X_1 \f$ and \f$ X_2 \f$ are tensors coupled by a single mode
   \f$ M_1 \f$ and \f$ M_2 \f$ are the coupled reconstructed CP tensors.
   This fit is loosely the proportion of the data described by the
   CP model, i.e., a fit of 1 is perfect.
   **/
  template <typename Tensor>
  class CoupledFitCheck {
  public:
    using ind_t = long;
    using ord_t = typename range_traits<typename Tensor::range_type>::ordinal_type;

    /// constructor for the base convergence test object
    /// \param[in] tol tolerance for ALS convergence
    explicit CoupledFitCheck(size_t lhs_dims, double tol = 1e-4) : tol_(tol), ndimL_(lhs_dims) {
    }

    ~CoupledFitCheck() = default;

    /// Function to check convergence of the ALS problem
    /// convergence when \f$ \|T - \hat{T}^{i+1}_n\|}{dim(A^{i}_n} \leq \epsilon \f$
    /// \param[in] btas_factors Current set of factor matrices
    bool operator () (const std::vector<Tensor> & btas_factors) {
      if (normTR_ < 0 || normTL_ < 0) BTAS_EXCEPTION("One must set the norm of the reference tensor");

      // First KRP (hadamard contract) out the first dimension of MtKRP using the last factor matrix
      // Need to do this for the right and the left side
      ord_t contract_size = btas_factors[ndimL_ - 1].extent(0);
      ord_t rank = btas_factors[0].extent(1);
      Tensor tempL(rank), tempR(rank);
      tempL.fill(0.0);
      tempR.fill(0.0);

      {
        auto &A = btas_factors[ndimL_ - 1];
        for (ord_t i = 0; i < contract_size; ++i) {
          auto *ptr_A = A.data() + i * rank;
          auto *ptr_MtKRP = MtKRPL_.data() + i * rank;
          for (ord_t r = 0; r < rank; ++r) {
            *(tempL.data() + r) += *(ptr_A + r) * *(ptr_MtKRP + r);
          }
        }
      }

      {
        auto n = btas_factors.size() - 2;
        contract_size = btas_factors[n].extent(0);
        auto &A = btas_factors[n];
        for (ord_t i = 0; i < contract_size; ++i) {
          auto *ptr_A = A.data() + i * rank;
          auto *ptr_MtKRP = MtKRPR_.data() + i * rank;
          for (ord_t r = 0; r < rank; ++r) {
            *(tempR.data() + r) += *(ptr_A + r) * *(ptr_MtKRP + r);
          }
        }
      }

      // Scale the final product by lambdas
      // These are the innerproducts of left and right tensors with their factors
      double iprodL = 0.0;
      double iprodR = 0.0;
      auto n = btas_factors.size() - 1;
      {
        auto * ptr_A = btas_factors[n].data();
        auto * ptr_temp = tempL.data();
        for (ord_t i = 0; i < rank; ++i) {
          iprodL += *(ptr_temp + i) * *(ptr_A + i);
        }
      }
      {
        auto * ptr_A = btas_factors[n].data();
        auto * ptr_temp = tempR.data();
        for (ord_t i = 0; i < rank; ++i) {
          iprodR += *(ptr_temp + i) * *(ptr_A + i);
        }
      }

      // Take the inner product of the factors <[[A,B,C..]], [[A,B,C,...]]>
      std::vector<Tensor> tensors_left;
      std::vector<Tensor> tensors_right;
      tensors_left.push_back(btas_factors[0]);
      tensors_right.push_back(btas_factors[0]);
      for (size_t i = 1; i < ndimL_; ++i) {
        tensors_left.push_back(btas_factors[i]);
      }
      for (size_t i = ndimL_; i < n + 1; ++i) {
        tensors_right.push_back(btas_factors[i]);
      }
      tensors_left.push_back(btas_factors[n]);

      double normFactorsL = norm(tensors_left);
      double normFactorsR = norm(tensors_right);

      // Find the residual sqrt(<T,T>  + <[[A,B,C...]],[[A,B,C,...]]> - 2 * <T, [[A,B,C,...]]>)
      double normResidualL = sqrt(abs(normTL_ * normTL_ + normFactorsL * normFactorsL - 2 * iprodL));
      double normResidualR = sqrt(abs(normTR_ * normTR_ + normFactorsR * normFactorsR - 2 * iprodR));
      //double fit = 1 - ((normResidualL + normResidualR) / (normTL_ + normTR_));
      double fit = 1 - ((normResidualR + normResidualL) / (normTR_ + normTL_));

      double fitChange = abs(fitOld_ - fit);
      fitOld_ = fit;
      if(fitChange < tol_) {
        iter_ = 0;
        final_fit_ = fitOld_;
        fitOld_ = -1.0;
        return true;
      }

      ++iter_;
      return false;
    }

    void set_norm(double normTL, double normTR){
      normTL_ = normTL;
      normTR_ = normTR;
    }

    void set_MtKRPL(Tensor & MtKRPL){
      MtKRPL_ = MtKRPL;
    }

    void set_MtKRPR(Tensor & MtKRPR){
      MtKRPR_ = MtKRPR;
    }

    double get_fit(){
      return final_fit_;
    }

  private:
    double tol_;
    double fitOld_ = -1.0;
    double final_fit_ = 0.0;
    double normTL_ = -1.0, normTR_ = -1.0;
    size_t iter_ = 0;
    Tensor MtKRPL_, MtKRPR_;
    size_t ndimL_;

    double norm(const std::vector<Tensor> & btas_factors) {
      ord_t rank = btas_factors[0].extent(1);
      auto n = btas_factors.size() - 1;
      Tensor coeffMat(rank, rank);
      auto temp = btas_factors[n];
      temp.resize(Range{Range1{rank}, Range1{1}});
      gemm(CblasNoTrans, CblasTrans, 1.0, temp, temp, 0.0, coeffMat);

      auto rank2 = rank * rank;
      for (size_t i = 0; i < n; ++i) {
        Tensor temp(rank, rank);
        gemm(CblasTrans, CblasNoTrans, 1.0, btas_factors[i], btas_factors[i], 0.0, temp);
        auto *ptr_coeff = coeffMat.data();
        auto *ptr_temp = temp.data();
        for (ord_t j = 0; j < rank2; ++j) {
          *(ptr_coeff + j) *= *(ptr_temp + j);
        }
      }

      auto nrm = 0.0;
      for(auto & i: coeffMat){
        nrm += i;
      }
      return sqrt(abs(nrm));
    }
  };
} //namespace btas
#endif  // BTAS_GENERIC_CONV_BASE_CLASS
