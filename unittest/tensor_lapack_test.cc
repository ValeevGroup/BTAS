#include "test.h"

#include <ctime>
#include <iostream>
#include <random>

#include "btas/tensor.h"
#include "btas/tensor_func.h"
#include "btas/generic/gesvd_impl.h"
#include "btas/generic/contract.h"

using std::cout;
using std::endl;
using namespace btas;

template <typename T>
std::ostream& 
operator<<(std::ostream& s, const Tensor<T>& X)
    {
    for(auto i : X.range()) s << i << " " << X(i) << "\n";
    return s;
    }

template <typename T>
T
randomReal()
    {
    static std::mt19937 rng(std::time(NULL));
    static auto dist = std::uniform_real_distribution<T>{0., 1.};
    return dist(rng);
    }

template <typename T>
std::complex<T>
randomCplx()
    {
    return std::complex<T>(randomReal<T>(),randomReal<T>());
    }

const auto eps_double = 1.e4 * std::numeric_limits<double>::epsilon();
const auto eps_float = 1.e4 * std::numeric_limits<float>::epsilon();

TEST_CASE("Tensor SVD") {

  SECTION("Double-precision SVD") {

    auto do_test = [](auto v) {
      using ztype = decltype(v);
      using zT = Tensor<ztype>;
      using dT = Tensor<double>;
      const size_t M = 5, N = 6;
      const auto R = std::min(M, N);
      zT A(M, N);
      A.generate([]() {
        if constexpr (std::is_same_v<ztype,double>)
          return randomReal<double>();
        else
          return randomCplx<double>();
      });
      zT U(M, M);
      zT Vt(N, N);
      dT Sigma(R);
      auto Acopy = A;
      gesvd(lapack::Job::AllVec, lapack::Job::AllVec, A, Sigma, U, Vt);

      ztype one(1.0);
      ztype zero(0.0);
      zT A_(M, N);
      zT S(M, N);
      S.fill(zero);
      for (auto i = 0; i < R; i++) {
        S(i, i) = Sigma(i);
      }
      zT temp1(M, N);
      contract(one, U, {'i', 'j'}, S, {'j', 'k'}, zero, temp1, {'i', 'k'});
      contract(one, temp1, {'i', 'j'}, Vt, {'j', 'k'}, zero, A_, {'i', 'k'});
      double res = 0;
      for (auto i = 0; i < A.extent(0); i++) {
        for (auto j = 0; j < A.extent(1); j++) {
          res += pow(abs(Acopy(i, j) - A_(i, j)), 2);
        }
      }
      CHECK(res < eps_double);
    };

    do_test(double{});
    do_test(std::complex<double>{});

  }
  SECTION("Double precision 3D SVD"){

    auto do_test = [](auto v) {
      using ztype = decltype(v);
      using zT = Tensor<ztype>;
      using dT = Tensor<double>;
      const size_t M = 2, N = 3, O = 5;
      zT A(M, N, O);
      A.generate([]() {
        if constexpr (std::is_same_v<ztype,double>)
          return randomReal<double>();
        else
          return randomCplx<double>();
      });
      zT U(M, N, M*N);
      zT Vt(5, 5);
      dT Sigma(5);
      auto Acopy = A;
      gesvd(lapack::Job::AllVec, lapack::Job::AllVec, A, Sigma, U, Vt);
      zT A_(2, 3, 5);
      zT S(6, 5);
      S.fill(0.0);
      for (auto i = 0; i < 5; i++) {
        S(i, i) = Sigma(i);
      }
      ztype one(1.0);
      ztype zero(0.0);
      zT temp1(2, 3, 5);
      contract(one, U, {'i', 'j', 'k'}, S, {'k', 'l'}, zero, temp1, {'i', 'j', 'l'});
      contract(one, temp1, {'i', 'j', 'l'}, Vt, {'l', 'm'},zero, A_, {'i', 'j', 'm'});
      double res = 0;
      for (auto i = 0; i < A.extent(0); i++) {
        for (auto j = 0; j < A.extent(1); j++) {
          for (auto k = 0; k < A.extent(2); k++) {
            res += abs(Acopy(i, j, k) - A_(i, j, k));
          }
        }
      }
      CHECK(res<eps_double);
    };
    do_test(double{});
    do_test(std::complex<double>{});
  }
}
