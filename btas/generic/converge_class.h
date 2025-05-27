#ifndef BTAS_GENERIC_CONV_BASE_CLASS
#define BTAS_GENERIC_CONV_BASE_CLASS

#include <vector>
#include <iomanip>
#include <btas/generic/dot_impl.h>
#include <btas/generic/contract.h>
#include <btas/generic/reconstruct.h>
#include <btas/generic/scal_impl.h>
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
    using ind_t = typename Tensor::range_type::index_type::value_type;
    using ord_t = typename range_traits<typename Tensor::range_type>::ordinal_type;

  public:
    /// constructor for the base convergence test object
    /// \param[in] tol tolerance for ALS convergence
    explicit NormCheck(double tol = 1e-3) : tol_(tol), iter_(0){
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

      if (verbose_) {
        std::cout << rank_ << "\t" << iter_ << "\t" << std::setprecision(16) << diff << std::endl;
      }
      if (diff < this->tol_) {
        return true;
      }
      ++iter_;

      return false;
    }

    /// Option to print fit and change in fit in the () operator call
    /// \param[in] verb bool which turns off/on fit printing.
    void verbose(bool verb) {
      verbose_ = verb;
    }

    double get_fit(bool hit_max_iters = false){

    }

  private:
    double tol_;
    std::vector<Tensor> prev;     // Set of previous factor matrices
    size_t ndim;                     // Number of factor matrices
    ind_t rank_;               // Rank of the CP problem
    bool verbose_ = false;
    size_t iter_;
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
    using ind_t = typename Tensor::range_type::index_type::value_type;
    using dtype = typename Tensor::value_type;
    using ord_t = typename range_traits<typename Tensor::range_type>::ordinal_type;
    using RT = real_type_t<dtype>;
    using RTensor = rebind_tensor_t<Tensor, RT>;
    /// constructor for the base convergence test object
    /// \param[in] tol tolerance for ALS convergence default = 1e-4
    explicit FitCheck(double tol = 1e-4): tol_(tol){
    }

    ~FitCheck() = default;

    /// Function to check convergence of the ALS problem
    /// convergence when \f$ 1 - \frac{\|X-full(M)\|}{\|X\|} \leq \epsilon \f$
    /// \param[in] btas_factors Current set of factor matrices
    /// \param[in] V Partial grammian matrices (rank x rank matricies from \f$ V^{i} = A^{iT} A^{i} \f$
    /// default = std::vector<Tensor>();
    bool operator()(const std::vector<Tensor> &btas_factors,
                    const std::vector<Tensor> & V = std::vector<Tensor>()) {
      if (normT_ < 0.0) BTAS_EXCEPTION("One must set the norm of the reference tensor");
      auto n = btas_factors.size() - 2;
      ord_t size = btas_factors[n].size();
      ind_t rank = btas_factors[n].extent(1);
      auto *ptr_A = btas_factors[n].data();
      auto *ptr_MtKRP = MtKRP_.data();
      auto lam_ptr = btas_factors[n + 1].data();
      dtype iprod = 0.0;
      for (ord_t i = 0; i < size; ++i) {
        iprod += *(ptr_MtKRP + i) * btas::impl::conj(*(ptr_A + i)) * *(lam_ptr + i % rank);
      }

      double normFactors = norm(btas_factors, V);
      double normResidual = sqrt(abs(normT_ * normT_ + normFactors - 2 * abs(iprod)));
      double fit = 1. - (normResidual / normT_);

      double fitChange = abs(fitOld_ - fit);
      fitOld_ = fit;
      if (verbose_) {
        std::cout << MtKRP_.extent(1) << "\t" << iter_ << "\t" << std::setprecision(16) << fit << "\t" << fitChange << std::endl;
      }
      if (fitChange < tol_) {
        converged_num++;
        if (converged_num == 2) {
          iter_ = 0;
          converged_num = 0;
          final_fit_ = fitOld_;
          fitOld_ = 1.0;
          return true;
        }
      }

      ++iter_;
      return false;
    }

    /// Set the norm of the reference tensor T
    /// \param[in] normT Norm of the reference tensor;
    void set_norm(double normT){
      normT_ = normT;
    }

    /// Set the current iteration's matricized tensor times KRP
    /// \f$ MtKRP = X_{n} * A^{1} \odot A^{2} \odot \dots \odot A^{n-1} \odot A^{n+1} \odot  \dots \odot A^{N} \f$
    /// Where N is the number of modes in the reference tensor X and \f$X_{n} \f$ is the nth mode
    /// matricization of X.
    /// \param[in] MtKRP matricized reference tensor times KRP
    void set_MtKRP(Tensor & MtKRP){
      MtKRP_ = MtKRP;
    }

    /// Returns the fit of the CP approximation, \f$ 1 - \frac{\|X - full{M}\|}{\|T\|} \f$
    /// from the previous () operator call.
    /// Where \f$ \hat{T} \f$ is the CP approximation of T
    /// \param[in] hit_max_iters bool, if CP_ALS strategy didn't converge hit_max_iters = true
    /// will return fitOld_ and reset the object, else return final_fit_;
    /// \returns fit of the CP approximation
    double get_fit(bool hit_max_iters = false){
      if(hit_max_iters){
        iter_ = 0;
        converged_num = 0;
        final_fit_ = fitOld_;
        fitOld_ = 1.0;
      }
      return final_fit_;
    }

    /// Option to print fit and change in fit in the () operator call
    /// \param[in] verb bool which turns off/on fit printing.
    void verbose(bool verb) {
      verbose_ = verb;
    }

  protected:
    double tol_;
    double fitOld_ = 1.0;
    double normT_ = -1.0;
    double final_fit_ = 0.0;
    size_t iter_ = 0;
    size_t converged_num = 0;
    Tensor MtKRP_;
    bool verbose_ = false;

    /// Function to compute the L2 norm of a tensors computed from the \c btas_factors
    /// \param[in] btas_factors Current set of factor matrices
    /// \param[in] V Partial grammian matrices (rank x rank matricies from \f$ V^{i} = A^{iT} A^{i} \f$
    double norm(const std::vector<Tensor> &btas_factors,
                const std::vector<Tensor> & V) {
      ind_t rank = btas_factors[0].extent(1);
      auto n = btas_factors.size() - 1;
      Tensor coeffMat;
      auto &temp1 = btas_factors[n];
      typename Tensor::value_type one = 1.0;
      ger(one, temp1.conj(), temp1, coeffMat);

      auto rank2 = rank * (ord_t)rank;
      Tensor temp(rank, rank);

      auto *ptr_coeff = coeffMat.data();
      if (V.empty()) {
        for (size_t i = 0; i < n; ++i) {
          gemm(blas::Op::Trans, blas::Op::NoTrans, 1.0, btas_factors[i].conj(), btas_factors[i], 0.0, temp);
          auto *ptr_temp = temp.data();
          for (ord_t j = 0; j < rank2; ++j) {
            *(ptr_coeff + j) *= *(ptr_temp + j);
          }
        }
      } else {
        for (size_t i = 0; i < n; ++i) {
          auto *ptr_V = V[i].data();
          for (ord_t j = 0; j < rank2; ++j) {
            *(ptr_coeff + j) *= *(ptr_V + j);
          }
        }
      }

      RT nrm = 0.0;
      for (auto &i : coeffMat) {
        nrm += std::real(i);
      }
      return nrm;
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
    using ind_t = typename Tensor::range_type::index_type::value_type;
    using ord_t = typename range_traits<typename Tensor::range_type>::ordinal_type;

    /// constructor for the base convergence test object
    /// \param[in] tol tolerance for ALS convergence
    explicit CoupledFitCheck(size_t lhs_dims, double tol = 1e-4) : tol_(tol), ndimL_(lhs_dims) {
    }

    ~CoupledFitCheck() = default;

    /// Function to check convergence of the ALS problem
    /// convergence when \f$ \|T - \frac{\hat{T}^{i+1}_n\|}{dim(A^{i}_n} \leq \epsilon \f$
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
      if (verbose_) {
        std::cout << MtKRPL_.extent(1) << "\t" << iter_ << "\t" << std::setprecision(16) << fit << "\t" << fitChange << std::endl;
      }
      if(fitChange < tol_) {
        iter_ = 0;
        final_fit_ = fitOld_;
        fitOld_ = 1.0;
        return true;
      }

      ++iter_;
      return false;
    }

    /// Set the norm of the reference tensors Tleft and Tright.
    /// \param[in] normTL Norm of the left reference tensor
    /// \param[in] normTR Norm of the right reference tensor
    void set_norm(double normTL, double normTR){
      normTL_ = normTL;
      normTR_ = normTR;
    }

    /// Set the current iteration's matricized tensor times KRP
    /// \f$ MtKRP = T_{n} * A^{1} \odot A^{2} \odot \dots \odot A^{n-1} \odot A^{n+1} \odot  \dots \odot A^{N} \f$
    /// Where N is the number of modes in the left reference tensor Tleft \f$T_{n} \f$ is the nth mode
    /// matricization of Tleft.
    /// \param[in] MtKRPl matricized left reference tensor times KRP
    void set_MtKRPL(Tensor & MtKRPL){
      MtKRPL_ = MtKRPL;
    }

    /// Set the current iteration's matricized tensor times KRP
    /// \f$ MtKRP = T_{n} * A^{1} \odot A^{2} \odot \dots \odot A^{n-1} \odot A^{n+1} \odot  \dots \odot A^{N} \f$
    /// Where N is the number of modes in the right reference tensor Tright \f$T_{n} \f$ is the nth mode
    /// matricization of Tright.
    /// \param[in] MtKRPr matricized right reference tensor times KRP
    void set_MtKRPR(Tensor & MtKRPR){
      MtKRPR_ = MtKRPR;
    }

    /// Returns the fit of the CP approximation, \f$ 1 - \frac{\|T - \hat{T}\|}{\|T\|} \f$
    /// from the previous () operator call
    /// Where \f$ T = T_{left}^T  T_{right} \f$ and \f$ \hat{T} \f$ is the CP approximation of T
    /// \param[in] hit_max_iters bool, if CP_ALS strategy didn't converge hit_max_iters = true
    /// will return fitOld_ and reset the object, else return final_fit_;
    /// \returns fit of the CP approximation
    double get_fit(bool hit_max_iters = false){
      if(hit_max_iters) {
        iter_ = 0;
        final_fit_ = fitOld_;
        fitOld_ = 1.0;
      }
      return final_fit_;
    }

    // returns the L2 norm of of the tensor generated by the CP
    // factor matrices.
    double get_norm(const std::vector<Tensor> & btas_array){
      return norm(btas_array);
    }

    void verbose(bool verb) { verbose_ = verb; }
  private:
    double tol_;
    double fitOld_ = 1.0;
    double final_fit_ = 0.0;
    double normTL_ = -1.0, normTR_ = -1.0;
    size_t iter_ = 0;
    Tensor MtKRPL_, MtKRPR_;
    size_t ndimL_;
    bool verbose_ = false;

    /// Function to compute the L2 norm of a tensors computed from the \c btas_factors
    /// \param[in] btas_factors Current set of factor matrices
    /// \param[in] V Partial grammian matrices (rank x rank matricies from \f$ V^{i} = A^{iT} A^{i} \f$
    double norm(const std::vector<Tensor> & btas_factors) {
      ord_t rank = btas_factors[0].extent(1);
      auto n = btas_factors.size() - 1;
      Tensor coeffMat(rank, rank);
      auto temp = btas_factors[n];
      temp.resize(Range{Range1{rank}, Range1{1}});
      gemm(blas::Op::NoTrans, blas::Op::Trans, 1.0, temp, temp, 0.0, coeffMat);

      auto rank2 = rank * rank;
      for (size_t i = 0; i < n; ++i) {
        Tensor temp(rank, rank);
        gemm(blas::Op::Trans, blas::Op::NoTrans, 1.0, btas_factors[i], btas_factors[i], 0.0, temp);
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
   \breif Class used to decide when ALS problem is converged.
   The fit is not computed and the optimization just runs until nALS is
   reached.
   **/
  template <typename Tensor>
  class NoCheck {
    using ind_t = typename Tensor::range_type::index_type::value_type;
    using ord_t = typename range_traits<typename Tensor::range_type>::ordinal_type;

   public:
    /// constructor for the base convergence test object
    /// \param[in] tol tolerance for ALS convergence
    explicit NoCheck(double tol = 1e-3) : iter_(0){
    }

    ~NoCheck() = default;

    /// Function to check convergence of the ALS problem
    /// convergence when \f$ \sum_n^{ndim} \frac{\|A^{i}_n - A^{i+1}_n\|}{dim(A^{i}_n} \leq \epsilon \f$
    /// \param[in] btas_factors Current set of factor matrices
    bool operator () (const std::vector<Tensor> &btas_factors,
                    const std::vector<Tensor> & V = std::vector<Tensor>()){
      auto rank_ = btas_factors[1].extent(1);
      if (verbose_) {
        std::cout << rank_ << "\t" << iter_ << std::endl;
      }
      ++iter_;

      return false;
    }

    /// Option to print fit and change in fit in the () operator call
    /// \param[in] verb bool which turns off/on fit printing.
    void verbose(bool verb) {
      verbose_ = verb;
    }

    double get_fit(bool hit_max_iters = false){

    }

   private:
    double tol_;
    bool verbose_ = false;
    size_t iter_;
    Tensor prevT_;
  };

  /// This class is going to take a tensor approximation
  /// and compare it to the previous tensor approximation
  /// Skipping the total fit and directly computing the relative fit
  template <typename Tensor>
  class ApproxFitCheck{
    using ind_t = typename Tensor::range_type::index_type::value_type;
    using ord_t = typename range_traits<typename Tensor::range_type>::ordinal_type;

   public:
    /// constructor for the base convergence test object
    /// \param[in] tol tolerance for ALS convergence
    explicit ApproxFitCheck(double tol = 1e-3) : iter_(0), tol_(tol){
    }

    ~ApproxFitCheck() = default;

    /// Function to check convergence of the ALS problem
    /// convergence when \f$ \sum_n^{ndim} \frac{\|A^{i}_n - A^{i+1}_n\|}{dim(A^{i}_n} \leq \epsilon \f$
    /// \param[in] btas_factors Current set of factor matrices

    bool operator () (std::vector<Tensor> & btas_factors,
                    const std::vector<Tensor> & V = std::vector<Tensor>()) {
      auto rank_ = btas_factors[1].extent(1);

      auto fit = 0.0;
      if(iter_ == 0) {
        fit_prev_ = (norm(btas_factors, btas_factors, rank_));
        norm_prev_ = sqrt(fit_prev_);
        prev_factors = btas_factors;
//        diff = reconstruct(btas_factors, orders);
        if (verbose_) {
          std::cout << rank_ << "\t" << iter_ << "\t" << 1.0 << std::endl;
        }
        ++iter_;
        return false;
      }

      auto curr_norm = norm(btas_factors, btas_factors, rank_);
      fit = sqrt(fit_prev_ - 2 * norm(prev_factors, btas_factors, rank_) + curr_norm) / norm_prev_;
//      fit = norm(diff);
//      diff = tnew;
      fit_prev_ = curr_norm;
      norm_prev_ = sqrt(curr_norm);
      prev_factors = btas_factors;

      if (verbose_) {
        std::cout << rank_ << "\t" << iter_ << "\t" << fit << std::endl;
      }
      ++iter_;
      if (fit < tol_) {
        ++converged_num;
        if(converged_num > 1) {
          iter_ = 0;
          return true;
        }
      }
      return false;
    }

    void verbose(bool verb){
      verbose_ = verb;
    }

   private:
    double tol_;
    bool verbose_ = false;
    double fit_prev_, norm_prev_;
    std::vector<size_t> orders;
    std::vector<Tensor> prev_factors;
//    Tensor diff;
    size_t converged_num = 0;
    size_t iter_;

    double norm(Tensor& a){
      auto n = 0.0;
      for(auto & i : a)
        n += i * i;
      return sqrt(n);
    }

    double norm(std::vector<Tensor> & facs1, std::vector<Tensor>& facs2, ind_t rank_){
      BTAS_ASSERT(facs1.size() == facs2.size());
      ind_t num_factors = facs1.size();
      Tensor RRp;
      Tensor T1 = facs1[0], T2 = facs2[0];
      auto lam_ptr1 = facs1[num_factors - 1].data(),
           lam_ptr2 = facs2[num_factors - 1].data();
      for (ind_t i = 0; i < rank_; i++) {
        scal(T1.extent(0), *(lam_ptr1 + i), std::begin(T1) + i, rank_);
        scal(T2.extent(0), *(lam_ptr2 + i), std::begin(T2) + i, rank_);
      }

      contract(1.0, T1, {1,2}, T2, {1,3}, 0.0, RRp, {2,3});

      for (ind_t i = 0; i < rank_; i++) {
        auto val1 = *(lam_ptr1 + i),
             val2 = *(lam_ptr2 + i);
        scal(T1.extent(0), (abs(val1) > 1e-12 ? 1.0/val1 : 1.0), std::begin(T1) + i, rank_);
        scal(T2.extent(0), (abs(val2) > 1e-12 ? 1.0/val2 : 1.0), std::begin(T2) + i, rank_);
      }

      auto * ptr_RRp = RRp.data();
      for (ind_t i = 1; i < num_factors - 3; ++i) {
        Tensor temp;
        contract(1.0, facs1[i], {1,2}, facs2[i], {1,3}, 0.0, temp, {2,3});
        auto * ptr_temp = temp.data();
        for(ord_t r = 0; r < rank_ * rank_; ++r)
          *(ptr_RRp + r) *= *(ptr_temp + r);
      }
      Tensor temp;
      auto last = num_factors - 2;
      contract(1.0, facs1[last], {1,2}, facs2[last], {1,3}, 0.0, temp, {2,3});
      return btas::dot(RRp, temp);
    }

  };

  /**
  \breif This is a class that computes the difference in two fits
   /| T - T^{i} \|^2 - \| T - T^{i + 1}\|^2 = T^{i}^2 - 2 TT^{i} + 2 TT^{i+1} - T^{i+1}^2
  **/
  template <typename Tensor>
  class DiffFitCheck{
    using ind_t = typename Tensor::range_type::index_type::value_type;
    using ord_t = typename range_traits<typename Tensor::range_type>::ordinal_type;
    using dtype = typename Tensor::value_type;

   public:
    /// constructor for the base convergence test object
    /// \param[in] tol tolerance for ALS convergence
    explicit DiffFitCheck(double tol = 1e-3) : iter_(0), tol_(tol){
    }

    ~DiffFitCheck() = default;

    /// Function to check convergence of the ALS problem
    /// convergence when \f$ \sum_n^{ndim} \frac{\|A^{i}_n - A^{i+1}_n\|}{dim(A^{i}_n} \leq \epsilon \f$
    /// \param[in] btas_factors Current set of factor matrices

    bool operator () (std::vector<Tensor> & btas_factors,
                      const std::vector<Tensor> & V = std::vector<Tensor>()) {
      auto rank_ = btas_factors[1].extent(1);
      auto n = btas_factors.size() - 1;
      auto & lambda = btas_factors[n];
      auto fit = 0.0;
      if(iter_ == 0) {
        fit_prev_ = sqrt(abs(norm(V, lambda, rank_) - 2.0 * abs(compute_inner_product(btas_factors[n - 1], lambda))));
        if (verbose_) {
          std::cout << rank_ << "\t" << iter_ << "\t" << 1.0 << std::endl;
        }
        ++iter_;
        return false;
      }

      auto curr_norm = sqrt(abs(norm(V, lambda, rank_) - 2.0 * abs(compute_inner_product(btas_factors[n - 1], lambda))));
      fit = sqrt(abs(fit_prev_ * fit_prev_ - curr_norm * curr_norm));
      fit_prev_ = curr_norm;

      if (verbose_) {
        std::cout << rank_ << "\t" << iter_ << "\t" << fit << std::endl;
      }
      ++iter_;
      if (fit < tol_) {
        ++converged_num;
        if(converged_num > 1) {
          return true;
        }
      }
      return false;
    }

    void verbose(bool verb){
      verbose_ = verb;
    }

    void set_MtKRP(Tensor & MtKRP){
      MtKRP_ = MtKRP;
    }

   private:
    double tol_;
    bool verbose_ = false;
    double fit_prev_;
    Tensor MtKRP_;
    size_t converged_num = 0;
    size_t iter_;

    dtype compute_inner_product(Tensor &last_factor, Tensor& lambda){
      ord_t size = last_factor.size();
      ind_t rank = last_factor.extent(1);
      auto *ptr_A = last_factor.data();
      auto *ptr_MtKRP = MtKRP_.data();
      auto lam_ptr = lambda.data();
      dtype iprod = 0.0;
      for (ord_t i = 0; i < size; ++i) {
        iprod += *(ptr_MtKRP + i) * btas::impl::conj(*(ptr_A + i)) * *(lam_ptr + i % rank);
      }
      return iprod;
    }

    double norm(const std::vector<Tensor> &V, Tensor & lambda, ind_t rank_) {
      auto n = V.size();
      Tensor coeffMat;
      typename Tensor::value_type one = 1.0;
      ger(one, lambda.conj(), lambda, coeffMat);

      auto rank2 = rank_ * (ord_t)rank_;
      Tensor temp(rank_, rank_);

      auto *ptr_coeff = coeffMat.data();
      for (size_t i = 0; i < n; ++i) {
        auto *ptr_V = V[i].data();
        for (ord_t j = 0; j < rank2; ++j) {
          *(ptr_coeff + j) *= *(ptr_V + j);
        }
      }

      dtype nrm = 0.0;
      for (auto &i : coeffMat) {
        nrm += i;
      }
      return nrm;
    }

    double norm(std::vector<Tensor> & facs1, std::vector<Tensor>& facs2, ind_t rank_){
      BTAS_ASSERT(facs1.size() == facs2.size());
      ind_t num_factors = facs1.size();
      Tensor RRp;
      Tensor T1 = facs1[0], T2 = facs2[0];
      auto lam_ptr1 = facs1[num_factors - 1].data(),
           lam_ptr2 = facs2[num_factors - 1].data();
      for (ind_t i = 0; i < rank_; i++) {
        scal(T1.extent(0), *(lam_ptr1 + i), std::begin(T1) + i, rank_);
        scal(T2.extent(0), *(lam_ptr2 + i), std::begin(T2) + i, rank_);
      }

      contract(1.0, T1, {1,2}, T2, {1,3}, 0.0, RRp, {2,3});

      for (ind_t i = 0; i < rank_; i++) {
        auto val1 = *(lam_ptr1 + i),
             val2 = *(lam_ptr2 + i);
        scal(T1.extent(0), (abs(val1) > 1e-12 ? 1.0/val1 : 1.0), std::begin(T1) + i, rank_);
        scal(T2.extent(0), (abs(val2) > 1e-12 ? 1.0/val2 : 1.0), std::begin(T2) + i, rank_);
      }

      auto * ptr_RRp = RRp.data();
      for (ind_t i = 1; i < num_factors - 3; ++i) {
        Tensor temp;
        contract(1.0, facs1[i], {1,2}, facs2[i], {1,3}, 0.0, temp, {2,3});
        auto * ptr_temp = temp.data();
        for(ord_t r = 0; r < rank_ * rank_; ++r)
          *(ptr_RRp + r) *= *(ptr_temp + r);
      }
      Tensor temp;
      auto last = num_factors - 2;
      contract(1.0, facs1[last], {1,2}, facs2[last], {1,3}, 0.0, temp, {2,3});
      return btas::dot(RRp, temp);
    }

  };
} //namespace btas
#endif  // BTAS_GENERIC_CONV_BASE_CLASS
