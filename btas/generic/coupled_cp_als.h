//
// Created by Karl Pierce on 7/24/19.
//

#ifndef BTAS_GENERIC_COUPLED_CP_ALS_H
#define BTAS_GENERIC_COUPLED_CP_ALS_H

#include <btas/btas.h>
#include <btas/generic/converge_class.h>
#include <btas/generic/cp.h>

namespace btas{

  template <typename Tensor, typename ConvClass = NormCheck<Tensor>>
  class COUPLED_CP_ALS : public CP<Tensor, ConvClass> {
  public:
    using CP<Tensor,ConvClass>::A;
    using CP<Tensor,ConvClass>::ndim;
    using CP<Tensor,ConvClass>::pseudoInverse;
    using CP<Tensor,ConvClass>::normCol;
    using CP<Tensor,ConvClass>::generate_KRP;
    using CP<Tensor,ConvClass>::generate_V;
    using CP<Tensor,ConvClass>::norm;

    COUPLED_CP_ALS(Tensor& left, Tensor& right) :
            CP<Tensor, ConvClass>(left.rank() + right.rank() - 1),
            tensor_ref_left(left), tensor_ref_right(right), ndimL(left.rank())
    {
      for(int i = 0; i < ndim; ++i){
        symm_dims.push_back(i);
      }
    }

    COUPLED_CP_ALS(Tensor& left, Tensor& right, std::vector<int>& symm):
            CP<Tensor, ConvClass>(left.rank() + right.rank()),
            tensor_ref_left(left), tensor_ref_right(right), ndimL(left.rank()), symm_dims(symm)
    {
      if (symm_dims.size() < ndim) {
        BTAS_EXCEPTION("Symmetry must be defined for all dimensions");
      }
    }

#ifdef _HAS_INTEL_MKL
    double compute_PALS(std::vector<ConvClass> & converge_list, double RankStep = 0.5, int panels = 4, bool symm = false,
                        int max_als = 20,bool fast_pI = true, bool calculate_epsilon = false, bool direct = true) override{
      return 0;
    }
#endif //_HAS_INTEL_MKL

  protected:
    Tensor& tensor_ref_left;
    Tensor& tensor_ref_right;
    int ndimL;
    std::vector<int> symm_dims;

    void build(int rank, ConvClass &converge_test, bool direct, int max_als, bool calculate_epsilon, int step, double &epsilon,
               bool SVD_initial_guess, int SVD_rank, bool & fast_pI, bool symm) override{
      // If its the first time into build and SVD_initial_guess
      // build and optimize the initial guess based on the left
      // singular vectors of the reference tensor.
#ifdef _HAS_INTEL_MKL
      if (A.empty() && SVD_initial_guess) {
        if (SVD_rank == 0) BTAS_EXCEPTION("Must specify the rank of the initial approximation using SVD");

        std::vector<int> modes_w_dim_LT_svd;
        A = std::vector<Tensor>(ndim);

        // Determine which factor matrices one can fill using SVD initial guess
        // start with left then do right
        auto ndimR = tensor_ref_right.rank();
        for(int i = 0; i < ndimL; i++){
          if(tensor_ref_left.extent(i) < SVD_rank){
            modes_w_dim_LT_svd.push_back(i);
          }
        }
        for(int i = 1; i < ndimR; i++){
          if(tensor_ref_right.extent(i) < SVD_rank){
            modes_w_dim_LT_svd.push_back(i + ndimL - 1);
          }
        }

        for(int tensor = 0; tensor < 2; ++tensor){
          auto & tensor_ref = tensor == 0 ? tensor_ref_left : tensor_ref_right;
          auto ndim_curr = tensor_ref.rank();
          // Fill all factor matrices with their singular vectors,
          // because we contract X X^T (where X is reference tensor) to make finding
          // singular vectors an eigenvalue problem some factor matrices will not be
          // full rank;
          bool left = tensor == 0;
          for(int i = left ? 0 : 1; i < ndim_curr; i++){
            int R = tensor_ref.extent(i);
            Tensor S(R,R), lambda(R);

            // Contract refrence tensor to make it square matrix of mode i
            gemm(CblasNoTrans, CblasTrans, 1.0, flatten(tensor_ref, i), flatten(tensor_ref, i), 0.0, S);

            // Find the Singular vectors of the matrix using eigenvalue decomposition
            auto info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', R, S.data(), R, lambda.data());
            if (info) BTAS_EXCEPTION("Error in computing the SVD initial guess");

            // Fill a factor matrix with the singular vectors with the largest corresponding singular
            // values
            lambda = Tensor(R, SVD_rank);
            lambda.fill(0.0);
            auto lower_bound = {0,0};
            auto upper_bound = {R, ((R > SVD_rank) ? SVD_rank : R)};
            auto view = make_view(S.range().slice(lower_bound, upper_bound), S.storage());
            auto l_iter = lambda.begin();
            for(auto iter = view.begin(); iter != view.end(); ++iter, ++l_iter){
              *(l_iter) = *(iter);
            }

            int A_dim = left ? i : i + ndimL - 1;
            A[A_dim] = lambda;
          }
        }

        //srand(3);
        std::mt19937 generator(random_seed_accessor());
        std::uniform_real_distribution<> distribution(-1.0, 1.0);
        // Fill the remaining columns in the set of factor matrices with dimension < SVD_rank with random numbers
        for(auto& i: modes_w_dim_LT_svd){
          auto dim = i < ndimL ? i : i - ndimL +1;
          auto & tensor_ref = i < ndimL ? tensor_ref_left : tensor_ref_right;
          int R = tensor_ref.extent(dim);
          auto lower_bound = {0, R};
          auto upper_bound = {R, SVD_rank};
          auto view = make_view(A[i].range().slice(lower_bound, upper_bound), A[i].storage());
          for(auto iter = view.begin(); iter != view.end(); ++iter){
            *(iter) = distribution(generator);
          }
        }

        // Normalize the columns of the factor matrices and
        // set the values al lambda, the weigt of each order 1 tensor
        Tensor lambda(Range{Range1{SVD_rank}});
        A.push_back(lambda);
        for(auto i = 0; i < ndim; ++i){
          this->normCol(A[i]);
        }

        // Optimize this initial guess.
        ALS(SVD_rank, converge_test, direct, max_als, calculate_epsilon, epsilon, fast_pI, symm);
      }
#else  // _HAS_INTEL_MKL
      if (SVD_initial_guess) BTAS_EXCEPTION("Computing the SVD requires LAPACK");
#endif // _HAS_INTEL_MKL
      // This loop keeps track of column dimension
      for (auto i = (A.empty()) ? 0 : A.at(0).extent(1); i < rank; i += step) {
        // This loop walks through the factor matrices
        for (auto j = 0; j < ndim; ++j) {  // select a factor matrix
          // If no factor matrices exists, make a set of factor matrices
          // and fill them with random numbers that are column normalized
          // and create the weighting vector lambda
          auto left = (j < ndimL);
          auto & tensor_ref = left ? tensor_ref_left : tensor_ref_right;
          if (i == 0) {
            Tensor a(Range{tensor_ref.range((left ? j : j - ndimL + 1)), Range1{i + 1}});
            a.fill(rand());
            A.push_back(a);
            this->normCol(j);
            if (j  == ndim - 1) {
              Tensor lam(Range{Range1{i + 1}});
              A.push_back(lam);
            }
          }

            // If the factor matrices have memory allocated, rebuild each matrix
            // with new column dimension col_dimension_old + skip
            // fill the new columns with random numbers and normalize the columns
          else {
            int row_extent = A[0].extent(0), rank_old = A[0].extent(1);
            Tensor b(Range{A[0].range(0), Range1{i + 1}});

            {
              auto lower_old = {0, 0}, upper_old = {row_extent, rank_old};
              auto old_view = make_view(b.range().slice(lower_old, upper_old), b.storage());
              auto A_itr = A[0].begin();
              for(auto iter = old_view.begin(); iter != old_view.end(); ++iter, ++A_itr){
                *(iter) = *(A_itr);
              }
            }

            {
              auto lower_new = {0, rank_old}, upper_new = {row_extent, (int) i+1};
              auto new_view = make_view(b.range().slice(lower_new, upper_new), b.storage());
              std::mt19937 generator(random_seed_accessor());
              std::uniform_real_distribution<> distribution(-1.0, 1.0);
              for(auto iter = new_view.begin(); iter != new_view.end(); ++iter){
                *(iter) = distribution(generator);
              }
            }

            A.erase(A.begin());
            A.push_back(b);
            if (j + 1 == ndim) {
              b.resize(Range{Range1{i + 1}});
              for (int k = 0; k < A[0].extent(0); k++) b(k) = A[0](k);
              A.erase(A.begin());
              A.push_back(b);
            }
          }
        }
        // compute the ALS of factor matrices with rank = i + 1.
        ALS(i + 1, converge_test, direct, max_als, calculate_epsilon, epsilon, fast_pI, symm);
      }
    }

    void build_random(int rank, ConvClass & converge_test, bool direct, int max_als, bool calculate_epsilon, double &epsilon,
                      bool & fast_pI, bool symm) override{
      return;
    }

    /// performs the ALS method to minimize the loss function for a single rank
    /// \param[in] rank The rank of the CP decomposition.
    /// \param[in] converge_test Test to see if ALS is converged
    /// \param[in] dir The CP decomposition be computed without calculating the
    /// Khatri-Rao product?
    /// \param[in] max_als If CP decomposition is to finite
    /// error, max_als is the highest rank approximation computed before giving up
    /// on CP-ALS. Default = 1e5.
    /// \param[in] calculate_epsilon Should the 2-norm
    /// error be calculated ||T_exact - T_approx|| = epsilon.
    /// \param[in] tcutALS
    /// How small difference in factor matrices must be to consider ALS of a
    /// single rank converged. Default = 0.1.
    /// \param[in, out] epsilon The 2-norm
    /// error between the exact and approximated reference tensor
    /// \param[in] fast_pI Should the pseudo inverse be computed using a fast cholesky decomposition
    /// \param[in] symm is \c tensor is symmetric in the last two dimension?
    void ALS(int rank, ConvClass & converge_test, bool dir, int max_als, bool calculate_epsilon, double &epsilon, bool & fast_pI, bool symm) {

      auto count = 0;
      // Until either the initial guess is converged or it runs out of iterations
      // update the factor matrices with or without Khatri-Rao product
      // intermediate
      bool is_converged = false;
      bool matlab = fast_pI;
      while(count < max_als && !is_converged){
        count++;
        this->num_ALS++;
        for (auto i = 0; i < ndim; i++) {
          if(symm_dims[i] == i) {
            direct(i, rank, symm, fast_pI, matlab, converge_test);
          } else {
            A[i] = A[symm_dims[i]];
          }
        }
        detail::get_fit(converge_test, epsilon);
        is_converged = converge_test(A);
      }
    }

    // In this we are minimizing a sum of two norms.  If n = coupled dimension then we have to minimize
    // f = ||B^{X}_{abcd...} - \hat{B}^X_{abcd...} || + || B^{X}_{ijkl...} - \hat{B}^X_{ijkl...}||
    // where X is the coupled dimension. otherwise we just minimize one of the two terms.
    void direct(int n, int rank, bool symm, bool & fast_pI, bool & matlab, ConvClass & converge_test){
      if(n == 0){
        // Start by computing (B^{X}_{abcd...} C^{-X}_{abcd...}) + B^{X}_{ijkl...} C^{-X}_{ijkl...}) = K
        // where C^{-X}_{abcd...} = C^{a} \odot C^{b} \odot C^{c} \odot C^{d} \dots ( the khatri-rao
        // product without the factor matrix C^X
        auto coupled_dim = tensor_ref_left.extent(0);
        Tensor K(coupled_dim, rank);
        K.fill(0.0);

        for(int tensor = 0; tensor < 2; ++tensor) {
          auto left = tensor == 0;
          auto & tensor_ref = left ? tensor_ref_left : tensor_ref_right;


          int ndim_curr = tensor_ref.rank(),
                  contract_size = tensor_ref.extent(ndim_curr - 1),
                  LHSsize = tensor_ref.size() / contract_size,
                  A_dim = left ? ndimL - 1: this->ndim - 1;
          auto R = tensor_ref.range();

          Tensor contract_tensor(LHSsize, rank);
          tensor_ref.resize(Range{Range1{LHSsize}, Range1{contract_size}});
          gemm(CblasNoTrans, CblasNoTrans, 1.0, tensor_ref, A[A_dim], 0.0, contract_tensor);
          tensor_ref.resize(R);
          --A_dim;
          for(int contract_dim = ndim_curr - 2; contract_dim > 0; --contract_dim, --A_dim){
            contract_size = tensor_ref.extent(contract_dim);
            LHSsize /= contract_size;

            contract_tensor.resize(Range{Range1{LHSsize}, Range1{contract_size}, Range1{rank}});
            Tensor temp(LHSsize, rank); temp.fill(0.0);
            for(int i = 0; i < LHSsize; ++i){
              for(int k = 0; k < contract_size; ++k){
                for(int r = 0; r < rank; ++r){
                  temp(i,r) += contract_tensor(i,k,r) * A[A_dim](k,r);
                }
              }
            }
            contract_tensor = temp;
          }
          K += contract_tensor;
        }
//         std::cout << "K: " << K << std::endl;
//         this->print(K);

        // Next form the Hadamard product sum
        // J = (C^{a\quadT} C^a * C^{b\quadT} C^b * \dots + C^{i\quadT} C^i * C^{j\quadT} C^j + \dots
        Tensor J1(rank, rank);
        J1.fill(1.0);
        {
          for (int i = 1; i < ndimL; ++i) {
            Tensor temp(rank, rank);
            gemm(CblasTrans, CblasNoTrans, 1.0, A[i], A[i], 0.0, temp);
            for (int j = 0; j < rank * rank; ++j) {
              *(J1.data() + j) *= *(temp.data() + j);
            }
          }
          Tensor J2(rank, rank);
          J2.fill(1.0);
          for (int i = ndimL; i < ndim; ++i) {
            Tensor temp(rank, rank);
            gemm(CblasTrans, CblasNoTrans, 1.0, A[i], A[i], 0.0, temp);
            for (int j = 0; j < rank * rank; ++j) {
              *(J2.data() + j) *= *(temp.data() + j);
            }
          }
          J1 += J2;
        }
        // Finally Form the product of K * J^\dagger
        Tensor a0(coupled_dim, rank);
        gemm(CblasNoTrans, CblasNoTrans, 1.0, K, pseudoInverse(J1), 0.0, a0);
        this->normCol(a0);
        A[0] = a0;
      }
      else{
        bool left = n < ndimL;
        Tensor &tensor_ref = left ? tensor_ref_left : tensor_ref_right;

        int ndim_curr = tensor_ref.rank(),
                contract_size = tensor_ref.extent(0),
                LHSsize = tensor_ref.size() / contract_size,
                A_dim = 0,
                pseudo_rank = rank,
                skip_dim = A[n].extent(0);
        auto R = tensor_ref.range();

        tensor_ref.resize(Range{Range1{contract_size}, Range1{LHSsize}});
        Tensor contract_tensor(LHSsize, rank);
        gemm(CblasTrans, CblasNoTrans, 1.0, tensor_ref, A[A_dim], 0.0, contract_tensor);

        tensor_ref.resize(R);
        A_dim = left ? ndimL - 1 : ndim - 1;
        for(int contract_dim = ndim_curr - 1; contract_dim > 0; --contract_dim, --A_dim){
          contract_size = tensor_ref.extent(contract_dim);
          LHSsize /= contract_size;
          contract_tensor.resize(Range{Range1{LHSsize}, Range1{contract_size}, Range1{pseudo_rank}});
          if(A_dim == n){
            pseudo_rank *= contract_size;
          }
          else if(A_dim > n){
            Tensor temp(LHSsize, pseudo_rank);
            temp.fill(0.0);
            auto & currA = A[A_dim];
            for(int i = 0; i < LHSsize; ++i){
              for(int j = 0; j < contract_size; ++j){
                for(int r = 0; r < rank; ++r){
                  temp(i,r) += contract_tensor(i,j,r) * currA(j,r);
                }
              }
            }
            contract_tensor = temp;
          }
          else{
            Tensor temp(LHSsize, pseudo_rank);
            temp.fill(0.0);
            auto & currA = A[A_dim];
            for(int i = 0; i < LHSsize; ++i){
              for(int j = 0; j < contract_size; ++j){
                for(int k = 0; k < skip_dim; ++k){
                  for(int r = 0; r < rank; ++r){
                    temp(i, r + k * rank) += contract_tensor(i,j,r +k * rank) * currA(j,r);
                  }
                }
              }
            }
            contract_tensor = temp;
          }
        }
        contract_tensor.resize(Range{Range1{skip_dim}, Range1{rank}});

        Tensor G(rank,rank);
        gemm(CblasTrans, CblasNoTrans, 1.0, A[0], A[0], 0.0, G);
        for(int i = (left ? 1 : ndimL); i < (left ? ndimL: ndim); ++i){
          if(i != n) {
            Tensor temp(rank, rank);
            gemm(CblasTrans, CblasNoTrans, 1.0, A[i], A[i], 0.0, temp);
            for (int j = 0; j < rank * rank; ++j) {
              *(G.data() + j) *= *(temp.data() + j);
            }
          }
        }
        if(n == ndimL - 1)
          detail::set_MtKRPL(converge_test, contract_tensor);
        else if(n == this->ndim - 1)
          detail::set_MtKRPR(converge_test, contract_tensor);
        Tensor an(skip_dim, rank);
        gemm(CblasNoTrans, CblasNoTrans, 1.0, contract_tensor, pseudoInverse(G), 0.0, an);
        this->normCol(an);
        A[n] = an;
      }
    }

  };

} // namespace btas

#endif //BTAS_GENERIC_COUPLED_CP_ALS_H
