#include "test.h"
#include <random>

#include "btas/tensor.h"
#include "btas/generic/dot_impl.h"

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

TEST_CASE("Tensor Dot")
    {

    SECTION("Double Dot")
        {
        Tensor<double> Td(4,2,6,5);
        Td.generate([](){ return randomReal<double>(); });
        const auto dres = dot(Td,Td);
        double dcheck = 0.;
        for(const auto& el : Td) dcheck += el*el;
        CHECK(fabs(dcheck-dres) < 1E-10);
        }

    SECTION("Float Dot")
        {
        Tensor<float> Tf(2,3,7,3);
        Tf.generate([](){ return randomReal<float>(); });
        const auto fres = dot(Tf,Tf);
        float fcheck = 0.;
        for(const auto& el : Tf) fcheck += el*el;
        CHECK(fabs(fcheck-fres) < 1E-10);

        Tensor<float> Uf(7,2,9);
        REQUIRE(Uf.size() == Tf.size());
        Uf.generate([](){ return randomReal<float>(); });
        const auto res = dot(Tf,Uf);
        fcheck = 0.;
        for(unsigned long i = 0; i < Tf.size(); ++i)
            {
            fcheck += Tf[i]*Uf[i];
            }
        CHECK(fabs(fcheck-res) < 1E-10);
        }

    SECTION("Complex Double Dot")
        {
        Tensor<std::complex<double>> Tc(8,9,4);
        Tc.generate([](){ return randomCplx<double>(); });
        const auto cres = dot(Tc,Tc);
        std::complex<double> ccheck = 0.;
        for(const auto& el : Tc) ccheck += std::conj(el)*el;
        CHECK(std::abs(ccheck-cres) < 1E-10);
        }

    SECTION("Complex Float Dot")
        {
        Tensor<std::complex<float>> Tc(8,9,4);
        Tc.generate([](){ return randomCplx<float>(); });
        const auto cres = dot(Tc,Tc);
        std::complex<float> ccheck = 0.;
        for(const auto& el : Tc) ccheck += std::conj(el)*el;
        CHECK(std::abs(ccheck-cres) < 1E-10);
        }

    }
