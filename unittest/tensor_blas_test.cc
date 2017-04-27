#include "test.h"

#include <iostream>
#include <random>

#include "btas/tensor.h"
#include "btas/tensor_func.h"
#include "btas/generic/dot_impl.h"
#include "btas/generic/scal_impl.h"
#include "btas/generic/axpy_impl.h"
#include "btas/generic/ger_impl.h"
#include "btas/generic/gemv_impl.h"
#include "btas/generic/gemm_impl.h"
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

TEST_CASE("Tensor Dot")
    {

    SECTION("Double Dot")
        {
        Tensor<double> Td(4,2,6,5);
        Td.generate([](){ return randomReal<double>(); });
        const auto dres = dot(Td,Td);
        double dcheck = 0.;
        for(const auto& el : Td) dcheck += el*el;
        CHECK(std::abs(dcheck-dres) < eps_double);
        }

    SECTION("Float Dot")
        {
        Tensor<float> Tf(2,3,7,3);
        Tf.generate([](){ return randomReal<float>(); });
        const auto fres = dot(Tf,Tf);
        float fcheck = 0.;
        for(const auto& el : Tf) fcheck += el*el;
        CHECK(fabs(fcheck-fres) < eps_float);

        Tensor<float> Uf(7,2,9);
        REQUIRE(Uf.size() == Tf.size());
        Uf.generate([](){ return randomReal<float>(); });
        const auto res = dot(Tf,Uf);
        fcheck = 0.;
        for(unsigned long i = 0; i < Tf.size(); ++i)
            {
            fcheck += Tf[i]*Uf[i];
            }
        CHECK(std::abs(fcheck-res) < eps_float);
        }

    SECTION("Complex Double Dot")
        {
        Tensor<std::complex<double>> Tc(8,9,4);
        Tc.generate([](){ return randomCplx<double>(); });
        const auto cres = dot(Tc,Tc);
        std::complex<double> ccheck = 0.;
        for(const auto& el : Tc) ccheck += std::conj(el)*el;
        CHECK(std::abs(ccheck-cres) < eps_double);
        }

    SECTION("Complex Float Dot")
        {
        Tensor<std::complex<float>> Tc(8,9,4);
        Tc.generate([](){ return randomCplx<float>(); });
        const auto cres = dot(Tc,Tc);
        std::complex<float> ccheck = 0.;
        for(const auto& el : Tc) ccheck += std::conj(el)*el;
        CHECK(std::abs(ccheck-cres) < eps_float);
        }

    }

TEST_CASE("Tensor Scal")
    {

    SECTION("Double Scal")
        {
        Tensor<double> T(4,2,6,5);
        T.generate([](){ return randomReal<double>(); });
        Tensor<double> Tbak=T;
        double d = randomReal<double>();
        scal(d,T);
        double res=0;
        for(auto i : T.range()) res+=std::abs(T(i)-Tbak(i)*d);
        CHECK(res < eps_double);
        }

    SECTION("Float Scal")
        {
        Tensor<float> T(4,2,6,5);
        T.generate([](){ return randomReal<float>(); });
        Tensor<float> Tbak=T;
        float d = randomReal<float>();
        scal(d,T);
        double res=0;
        for(auto i : T.range()) res+=std::abs(T(i)-Tbak(i)*d);
        CHECK(res < eps_float);
        }

    SECTION("Complex Double Scal")
        {
        Tensor<std::complex<double>> T(4,2,6,5);
        T.generate([](){ return randomCplx<double>(); });
        Tensor<std::complex<double>> Tbak=T;
        std::complex<double> d = randomCplx<double>();
        scal(d,T);
        double res=0;
        for(auto i : T.range()) res+=std::abs(T(i)-Tbak(i)*d);
        CHECK(res < eps_double);
        }

    SECTION("Complex Float scal")
        {
        Tensor<std::complex<float>> T(4,2,6,5);
        T.generate([](){ return randomCplx<float>(); });
        Tensor<std::complex<float>> Tbak=T;
        std::complex<float> d = randomCplx<float>();
        scal(d,T);
        double res=0;
        for(auto i : T.range()) res+=std::abs(T(i)-Tbak(i)*d);
        CHECK(res < eps_float);
        }
    } 

TEST_CASE("Tensor Axpy")
    {

    SECTION("Double Axpy")
        {
        Tensor<double> X(4,2,6,5);
        Tensor<double> Y(4,2,6,5);
        X.generate([](){ return randomReal<double>(); });
        Y.generate([](){ return randomReal<double>(); });
        Tensor<double> Ybak=Y;
        double alpha = randomReal<double>();
        axpy(alpha,X,Y);
        double res=0;
        for(auto i : Y.range()) res+=std::abs(Ybak(i)+X(i)*alpha-Y(i));
        CHECK(res < eps_double);
        }

    SECTION("Float Axpy")
        {
        Tensor<float> X(4,2,6,5);
        Tensor<float> Y(4,2,6,5);
        X.generate([](){ return randomReal<float>(); });
        Y.generate([](){ return randomReal<float>(); });
        Tensor<float> Ybak=Y;
        float alpha = randomReal<float>();
        axpy(alpha,X,Y);
        double res=0;
        for(auto i : X.range()) res+=std::abs(Ybak(i)+X(i)*alpha-Y(i));
        CHECK(res < eps_float);
        }

    SECTION("Complex Double Axpy")
        {
        Tensor<std::complex<double>> X(4,2,6,5);
        Tensor<std::complex<double>> Y(4,2,6,5);
        X.generate([](){ return randomCplx<double>(); });
        Y.generate([](){ return randomCplx<double>(); });
        Tensor<std::complex<double>> Ybak=Y;
        std::complex<double> alpha = randomReal<double>();
        axpy(alpha,X,Y);
        double res=0;
        for(auto i : X.range()) res+=std::abs(Ybak(i)+X(i)*alpha-Y(i));
        CHECK(res < eps_double);
        }

    SECTION("Complex Float Axpy")
        {
        Tensor<std::complex<float>> X(4,2,6,5);
        Tensor<std::complex<float>> Y(4,2,6,5);
        X.generate([](){ return randomCplx<float>(); });
        Y.generate([](){ return randomCplx<float>(); });
        Tensor<std::complex<float>> Ybak=Y;
        std::complex<float> alpha = randomReal<float>();
        axpy(alpha,X,Y);
        double res=0;
        for(auto i : X.range()) res+=std::abs(Ybak(i)+X(i)*alpha-Y(i));
        CHECK(res < eps_float);
        }
    }

TEST_CASE("Tensor Ger")
    {

    SECTION("Double Ger")
        {
        Tensor<double> A(4,2,6,5);
        Tensor<double> X(4,2);
        Tensor<double> Y(6,5);
        A.generate([](){ return randomReal<double>(); });
        X.generate([](){ return randomReal<double>(); });
        Y.generate([](){ return randomReal<double>(); });
        Tensor<double> Abak=A;
        double a = randomReal<double>();
        ger(a,X,Y,A);
        double res=0;
        for(auto i : A.range()) res+=std::abs(a*X(i[0],i[1])*Y(i[2],i[3])+Abak(i)-A(i));
        CHECK(res < eps_double);
        }

    SECTION("Float Ger")
        {
        Tensor<float> A(4,2,6,5);
        Tensor<float> X(4,2);
        Tensor<float> Y(6,5);
        A.generate([](){ return randomReal<float>(); });
        X.generate([](){ return randomReal<float>(); });
        Y.generate([](){ return randomReal<float>(); });
        Tensor<float> Abak=A;
        float a = randomReal<float>();
        ger(a,X,Y,A);
        double res=0;
        for(auto i : A.range()) res+=std::abs(a*X(i[0],i[1])*Y(i[2],i[3])+Abak(i)-A(i));
        CHECK(res < eps_float);
        }

    SECTION("Complex Double Ger")
        {
        Tensor<std::complex<double>> A(4,2,6,5);
        Tensor<std::complex<double>> X(4,2);
        Tensor<std::complex<double>> Y(6,5);
        A.generate([](){ return randomCplx<double>(); });
        X.generate([](){ return randomCplx<double>(); });
        Y.generate([](){ return randomCplx<double>(); });
        Tensor<std::complex<double>> Abak=A;
        std::complex<double> a = randomCplx<double>();
        ger(a,X,Y,A);
        double res=0;
        for(auto i : A.range()) res+=std::abs(a*X(i[0],i[1])*Y(i[2],i[3])+Abak(i)-A(i));
        CHECK(res < eps_double);
        }

    SECTION("Complex Float Ger")
        {
        Tensor<std::complex<float>> A(4,2,6,5);
        Tensor<std::complex<float>> X(4,2);
        Tensor<std::complex<float>> Y(6,5);
        A.generate([](){ return randomCplx<float>(); });
        X.generate([](){ return randomCplx<float>(); });
        Y.generate([](){ return randomCplx<float>(); });
        Tensor<std::complex<float>> Abak=A;
        std::complex<float> a = randomCplx<float>();
        ger(a,X,Y,A);
        double res=0;
        for(auto i : A.range()) res+=std::abs(a*X(i[0],i[1])*Y(i[2],i[3])+Abak(i)-A(i));
        CHECK(res < eps_float);
        }
    }

TEST_CASE("Tensor Gemv")
    {

    SECTION("Double Gemv --- NoTrans")
        {
        Tensor<double> A(4,2,6,5);
        Tensor<double> X(6,5);
        Tensor<double> Y(4,2);
        A.generate([](){ return randomReal<double>(); });
        X.generate([](){ return randomReal<double>(); });
        Y.generate([](){ return randomReal<double>(); });
        double alpha = randomReal<double>();
        double beta = randomReal<double>();
        Tensor<double> Ytest=Y;
        scal(beta,Ytest);
        for(long i=0;i<A.extent(0);i++)
        for(long j=0;j<A.extent(1);j++)
        for(long k=0;k<A.extent(2);k++)
        for(long l=0;l<A.extent(3);l++){
            Ytest(i,j)+=alpha*A(i,j,k,l)*X(k,l);
        }
        gemv(CblasNoTrans,alpha,A,X,beta,Y);
        double res=0;
        for(auto i : Y.range()) res+= std::abs(Ytest(i)- Y(i));
        CHECK(res < eps_double);
        }

    SECTION("Double Gemv --- Trans")
        {
        Tensor<double> A(6,5,4,2);
        Tensor<double> X(6,5);
        Tensor<double> Y(4,2);
        A.generate([](){ return randomReal<double>(); });
        X.generate([](){ return randomReal<double>(); });
        Y.generate([](){ return randomReal<double>(); });
        double alpha = randomReal<double>();
        double beta = randomReal<double>();
        Tensor<double> Ytest=Y;
        scal(beta,Ytest);
        for(long i=0;i<A.extent(2);i++)
        for(long j=0;j<A.extent(3);j++)
        for(long k=0;k<A.extent(0);k++)
        for(long l=0;l<A.extent(1);l++){
            Ytest(i,j)+=alpha*A(k,l,i,j)*X(k,l);
        }
        gemv(CblasTrans,alpha,A,X,beta,Y);
        double res=0;
        for(auto i : Y.range()) res+= std::abs(Ytest(i)- Y(i));
        CHECK(res < eps_double);
        }

    SECTION("Float Gemv --- NoTrans")
        {
        Tensor<float> A(4,2,6,5);
        Tensor<float> X(6,5);
        Tensor<float> Y(4,2);
        A.generate([](){ return randomReal<float>(); });
        X.generate([](){ return randomReal<float>(); });
        Y.generate([](){ return randomReal<float>(); });
        float alpha = randomReal<double>();
        float beta = randomReal<double>();
        Tensor<float> Ytest=Y;
        scal(beta,Ytest);
        for(long i=0;i<A.extent(0);i++)
        for(long j=0;j<A.extent(1);j++)
        for(long k=0;k<A.extent(2);k++)
        for(long l=0;l<A.extent(3);l++){
            Ytest(i,j)+=alpha*A(i,j,k,l)*X(k,l);
        }
        gemv(CblasNoTrans,alpha,A,X,beta,Y);
        double res=0;
        for(auto i : Y.range()) res+= std::abs(Ytest(i)- Y(i));
        CHECK(res < eps_float);
        }

    SECTION("Complex Double Gemv --- NoTrans")
        {
        Tensor<std::complex<double>> A(4,2,6,5);
        Tensor<std::complex<double>> X(6,5);
        Tensor<std::complex<double>> Y(4,2);
        A.generate([](){ return randomCplx<double>(); });
        X.generate([](){ return randomCplx<double>(); });
        Y.generate([](){ return randomCplx<double>(); });
        std::complex<double> alpha = randomCplx<double>();
        std::complex<double> beta = randomCplx<double>();
        Tensor<std::complex<double>> Ytest=Y;
        scal(beta,Ytest);
        for(long i=0;i<A.extent(0);i++)
        for(long j=0;j<A.extent(1);j++)
        for(long k=0;k<A.extent(2);k++)
        for(long l=0;l<A.extent(3);l++){
            Ytest(i,j)+=alpha*A(i,j,k,l)*X(k,l);
        }
        gemv(CblasNoTrans,alpha,A,X,beta,Y);
        double res=0;
        for(auto i : Y.range()) res+= std::abs(Ytest(i)- Y(i));
        CHECK(res < eps_double);
        }

    SECTION("Complex Double Gemv --- Trans")
        {
        Tensor<std::complex<double>> A(6,5,4,2);
        Tensor<std::complex<double>> X(6,5);
        Tensor<std::complex<double>> Y(4,2);
        A.generate([](){ return randomCplx<double>(); });
        X.generate([](){ return randomCplx<double>(); });
        Y.generate([](){ return randomCplx<double>(); });
        std::complex<double> alpha = randomCplx<double>();
        std::complex<double> beta = randomCplx<double>();
        Tensor<std::complex<double>> Ytest=Y;
        scal(beta,Ytest);
        for(long i=0;i<A.extent(2);i++)
        for(long j=0;j<A.extent(3);j++)
        for(long k=0;k<A.extent(0);k++)
        for(long l=0;l<A.extent(1);l++){
            Ytest(i,j)+=alpha*A(k,l,i,j)*X(k,l);
        }
        gemv(CblasTrans,alpha,A,X,beta,Y);
        double res=0;
        for(auto i : Y.range()) res+= std::abs(Ytest(i)- Y(i));
        CHECK(res < eps_double);
        }

    SECTION("Complex Double Gemv --- ConjTrans")
        {
        Tensor<std::complex<double>> A(6,5,4,2);
        Tensor<std::complex<double>> X(6,5);
        Tensor<std::complex<double>> Y(4,2);
        A.generate([](){ return randomCplx<double>(); });
        X.generate([](){ return randomCplx<double>(); });
        Y.generate([](){ return randomCplx<double>(); });
        std::complex<double> alpha = randomCplx<double>();
        std::complex<double> beta = randomCplx<double>();
        Tensor<std::complex<double>> Ytest=Y;
        scal(beta,Ytest);
        for(long i=0;i<A.extent(2);i++)
        for(long j=0;j<A.extent(3);j++)
        for(long k=0;k<A.extent(0);k++)
        for(long l=0;l<A.extent(1);l++){
            //FIXME
            //Why no conj
            //There may be a bug.
            Ytest(i,j)+=alpha*conj(A(k,l,i,j))*X(k,l);
            //Ytest(i,j)+=alpha*A(k,l,i,j)*X(k,l);
        }
        gemv(CblasConjTrans,alpha,A,X,beta,Y);
        double res=0;
        for(auto i : Y.range()) res+= std::abs(Ytest(i)- Y(i));
        CHECK(res < eps_double);
        }

    SECTION("Complex Float Gemv --- NoTrans")
        {
        Tensor<std::complex<float>> A(4,2,6,5);
        Tensor<std::complex<float>> X(6,5);
        Tensor<std::complex<float>> Y(4,2);
        A.generate([](){ return randomCplx<float>(); });
        X.generate([](){ return randomCplx<float>(); });
        Y.generate([](){ return randomCplx<float>(); });
        std::complex<float> alpha = randomCplx<float>();
        std::complex<float> beta = randomCplx<float>();
        Tensor<std::complex<float>> Ytest=Y;
        scal(beta,Ytest);
        for(long i=0;i<A.extent(0);i++)
        for(long j=0;j<A.extent(1);j++)
        for(long k=0;k<A.extent(2);k++)
        for(long l=0;l<A.extent(3);l++){
            Ytest(i,j)+=alpha*A(i,j,k,l)*X(k,l);
        }
        gemv(CblasNoTrans,alpha,A,X,beta,Y);
        double res=0;
        for(auto i : Y.range()) res+= std::abs(Ytest(i)- Y(i));
        CHECK(res < eps_float);
        }


    }

TEST_CASE("Tensor Gemm")
    {

    SECTION("Double Gemm --- NoTrans")
        {
        Tensor<double> A(4,2,6,5);
        Tensor<double> B(6,5,7);
        Tensor<double> C(4,2,7);
        A.generate([](){ return randomReal<double>(); });
        B.generate([](){ return randomReal<double>(); });
        C.generate([](){ return randomReal<double>(); });
        double alpha = randomReal<double>();
        double beta = randomReal<double>();
        Tensor<double> Ctest=C;
        scal(beta,Ctest);
        for(long i=0;i<A.extent(0);i++)
        for(long j=0;j<A.extent(1);j++)
        for(long k=0;k<A.extent(2);k++)
        for(long l=0;l<A.extent(3);l++)
        for(long m=0;m<B.extent(2);m++){
            Ctest(i,j,m)+=alpha*A(i,j,k,l)*B(k,l,m);
        }
        gemm(CblasNoTrans,CblasNoTrans,alpha,A,B,beta,C);
        double res=0;
        for(auto i : C.range()) res+= std::abs(Ctest(i)- C(i));
        CHECK(res < eps_double);
        }

    SECTION("Double Gemm --- Trans")
        {
        //Tensor<double> A(4,2,6,5);
        Tensor<double> A(6,5,4,2);
        Tensor<double> B(6,5,7);
        Tensor<double> C(4,2,7);
        A.generate([](){ return randomReal<double>(); });
        B.generate([](){ return randomReal<double>(); });
        C.generate([](){ return randomReal<double>(); });
        double alpha = randomReal<double>();
        double beta = randomReal<double>();
        Tensor<double> Ctest=C;
        scal(beta,Ctest);
        for(long i=0;i<A.extent(2);i++)
        for(long j=0;j<A.extent(3);j++)
        for(long k=0;k<A.extent(0);k++)
        for(long l=0;l<A.extent(1);l++)
        for(long m=0;m<B.extent(2);m++){
            Ctest(i,j,m)+=alpha*A(k,l,i,j)*B(k,l,m);
        }
        gemm(CblasTrans,CblasNoTrans,alpha,A,B,beta,C);
        double res=0;
        for(auto i : C.range()) res+= std::abs(Ctest(i)- C(i));
        CHECK(res < eps_double);
        }

    SECTION("Float Gemm --- NoTrans")
        {
        Tensor<float> A(4,2,6,5);
        Tensor<float> B(6,5,7);
        Tensor<float> C(4,2,7);
        A.generate([](){ return randomReal<float>(); });
        B.generate([](){ return randomReal<float>(); });
        C.generate([](){ return randomReal<float>(); });
        float alpha = randomReal<float>();
        float beta = randomReal<float>();
        Tensor<float> Ctest=C;
        scal(beta,Ctest);
        for(long i=0;i<A.extent(0);i++)
        for(long j=0;j<A.extent(1);j++)
        for(long k=0;k<A.extent(2);k++)
        for(long l=0;l<A.extent(3);l++)
        for(long m=0;m<B.extent(2);m++){
            Ctest(i,j,m)+=alpha*A(i,j,k,l)*B(k,l,m);
        }
        gemm(CblasNoTrans,CblasNoTrans,alpha,A,B,beta,C);
        double res=0;
        for(auto i : C.range()) res+= std::abs(Ctest(i)- C(i));
        CHECK(res < eps_float);
        }

    SECTION("Float Gemm --- Trans")
        {
        //Tensor<float> A(4,2,6,5);
        Tensor<float> A(6,5,4,2);
        Tensor<float> B(6,5,7);
        Tensor<float> C(4,2,7);
        A.generate([](){ return randomReal<float>(); });
        B.generate([](){ return randomReal<float>(); });
        C.generate([](){ return randomReal<float>(); });
        float alpha = randomReal<float>();
        float beta = randomReal<float>();
        Tensor<float> Ctest=C;
        scal(beta,Ctest);
        for(long i=0;i<A.extent(2);i++)
        for(long j=0;j<A.extent(3);j++)
        for(long k=0;k<A.extent(0);k++)
        for(long l=0;l<A.extent(1);l++)
        for(long m=0;m<B.extent(2);m++){
            Ctest(i,j,m)+=alpha*A(k,l,i,j)*B(k,l,m);
        }
        gemm(CblasTrans,CblasNoTrans,alpha,A,B,beta,C);
        double res=0;
        for(auto i : C.range()) res+= std::abs(Ctest(i)- C(i));
        CHECK(res < eps_float);
        }

    SECTION("Complex Double Gemm --- NoTrans")
        {
        Tensor<std::complex<double>> A(4,2,6,5);
        Tensor<std::complex<double>> B(6,5,7);
        Tensor<std::complex<double>> C(4,2,7);
        A.generate([](){ return randomCplx<double>(); });
        B.generate([](){ return randomCplx<double>(); });
        C.generate([](){ return randomCplx<double>(); });
        auto  alpha = randomCplx<double>();
        auto  beta = randomCplx<double>();
        Tensor<std::complex<double>> Ctest=C;
        scal(beta,Ctest);
        for(long i=0;i<A.extent(0);i++)
        for(long j=0;j<A.extent(1);j++)
        for(long k=0;k<A.extent(2);k++)
        for(long l=0;l<A.extent(3);l++)
        for(long m=0;m<B.extent(2);m++){
            Ctest(i,j,m)+=alpha*A(i,j,k,l)*B(k,l,m);
        }
        gemm(CblasNoTrans,CblasNoTrans,alpha,A,B,beta,C);
        double res=0;
        for(auto i : C.range()) res+= std::abs(Ctest(i)- C(i));
        CHECK(res < eps_double);
        }

    SECTION("Complex Double Gemm --- Trans")
        {
        Tensor<std::complex<double>> A(6,5,4,2);
        Tensor<std::complex<double>> B(6,5,7);
        Tensor<std::complex<double>> C(4,2,7);
        A.generate([](){ return randomCplx<double>(); });
        B.generate([](){ return randomCplx<double>(); });
        C.generate([](){ return randomCplx<double>(); });
        auto  alpha = randomCplx<double>();
        auto  beta = randomCplx<double>();
        Tensor<std::complex<double>> Ctest=C;
        scal(beta,Ctest);
        for(long i=0;i<A.extent(2);i++)
        for(long j=0;j<A.extent(3);j++)
        for(long k=0;k<A.extent(0);k++)
        for(long l=0;l<A.extent(1);l++)
        for(long m=0;m<B.extent(2);m++){
            Ctest(i,j,m)+=alpha*A(k,l,i,j)*B(k,l,m);
        }
        gemm(CblasTrans,CblasNoTrans,alpha,A,B,beta,C);
        double res=0;
        for(auto i : C.range()) res+= std::abs(Ctest(i)- C(i));
        CHECK(res < eps_double);
        }

    SECTION("Complex Double Gemm --- ConjTrans")
        {
        Tensor<std::complex<double>> A(6,5,4,2);
        Tensor<std::complex<double>> B(6,5,7);
        Tensor<std::complex<double>> C(4,2,7);
        A.generate([](){ return randomCplx<double>(); });
        B.generate([](){ return randomCplx<double>(); });
        C.generate([](){ return randomCplx<double>(); });
        auto  alpha = randomCplx<double>();
        auto  beta = randomCplx<double>();
        Tensor<std::complex<double>> Ctest=C;
        scal(beta,Ctest);
        for(long i=0;i<A.extent(2);i++)
        for(long j=0;j<A.extent(3);j++)
        for(long k=0;k<A.extent(0);k++)
        for(long l=0;l<A.extent(1);l++)
        for(long m=0;m<B.extent(2);m++){
            Ctest(i,j,m)+=alpha*conj(A(k,l,i,j))*B(k,l,m);
        }
        gemm(CblasConjTrans,CblasNoTrans,alpha,A,B,beta,C);
        double res=0;
        for(auto i : C.range()) res+= std::abs(Ctest(i)- C(i));
        CHECK(res < eps_double);
        }
    
    SECTION("Complex Float Gemm --- NoTrans")
        {
        Tensor<std::complex<float>> A(4,2,6,5);
        Tensor<std::complex<float>> B(6,5,7);
        Tensor<std::complex<float>> C(4,2,7);
        A.generate([](){ return randomCplx<float>(); });
        B.generate([](){ return randomCplx<float>(); });
        C.generate([](){ return randomCplx<float>(); });
        auto  alpha = randomCplx<float>();
        auto  beta = randomCplx<float>();
        Tensor<std::complex<float>> Ctest=C;
        scal(beta,Ctest);
        for(long i=0;i<A.extent(0);i++)
        for(long j=0;j<A.extent(1);j++)
        for(long k=0;k<A.extent(2);k++)
        for(long l=0;l<A.extent(3);l++)
        for(long m=0;m<B.extent(2);m++){
            Ctest(i,j,m)+=alpha*A(i,j,k,l)*B(k,l,m);
        }
        gemm(CblasNoTrans,CblasNoTrans,alpha,A,B,beta,C);
        double res=0;
        for(auto i : C.range()) res+= std::abs(Ctest(i)- C(i));
        CHECK(res < eps_float);
        }
    }

TEST_CASE("Contraction")
    {
    SECTION("Double Contraction")
        {
        Tensor<double> A(2,3,5);
        Tensor<double> B(5,3,6,4);
        Tensor<double> C(2,4,6);
        Tensor<double> Ctest(2,4,6);
        //enum {i,j,k,l,m,n};
        A.generate([](){ return randomReal<double>(); });
        B.generate([](){ return randomReal<double>(); });
        C.generate([](){ return randomReal<double>(); });
        double alpha = randomReal<double>();
        double beta = randomReal<double>();
        Ctest=C;
        scal(beta,Ctest);
        contract(alpha,A,{'i','j','k'},B,{'k','j','l','m'},beta,C,{'i','m','l'});
        for(long i=0;i<A.extent(0);i++)
        for(long j=0;j<A.extent(1);j++)
        for(long k=0;k<A.extent(2);k++)
        for(long l=0;l<B.extent(2);l++)
        for(long m=0;m<B.extent(3);m++){
            Ctest(i,m,l)+=alpha*A(i,j,k)*B(k,j,l,m);
        }
        double res=0;
        for(auto i : C.range()) res+=std::abs(C(i)-Ctest(i));
        CHECK(res < eps_double);

        Tensor<double> D;
        Tensor<double> Dtest(2,4,6);
        Dtest.fill(0.0);
        contract(alpha,A,{'i','j','k'},B,{'k','j','l','m'},0.0,D,{'i','m','l'});
        for(long i=0;i<A.extent(0);i++)
        for(long j=0;j<A.extent(1);j++)
        for(long k=0;k<A.extent(2);k++)
        for(long l=0;l<B.extent(2);l++)
        for(long m=0;m<B.extent(3);m++){
            Dtest(i,m,l)+=alpha*A(i,j,k)*B(k,j,l,m);
        }
        double res1=0;
        for(auto i : D.range()) res1+=std::abs(D(i)-Dtest(i));
        CHECK(res1 < eps_double);
        }

    SECTION("Float Contraction")
        {
        Tensor<float> A(2,3,5);
        Tensor<float> B(5,3,6,4);
        Tensor<float> C;
        Tensor<float> Ctest(2,4,6);
        Ctest.fill(0.0);
        //enum {i,j,k,l,m,n};
        A.generate([](){ return randomReal<float>(); });
        B.generate([](){ return randomReal<float>(); });
        float alpha = randomReal<float>();
        contract(alpha,A,{'i','j','k'},B,{'k','j','l','m'},0.0f,C,{'i','m','l'});
        for(long i=0;i<A.extent(0);i++)
        for(long j=0;j<A.extent(1);j++)
        for(long k=0;k<A.extent(2);k++)
        for(long l=0;l<B.extent(2);l++)
        for(long m=0;m<B.extent(3);m++){
            Ctest(i,m,l)+=alpha*A(i,j,k)*B(k,j,l,m);
        }
        float res=0;
        for(auto i : C.range()) res+=std::abs(C(i)-Ctest(i));
        CHECK(res < eps_float);
        }

    SECTION("Complex Double Contraction")
        {
        Tensor<std::complex<double>> A(2,3,5);
        Tensor<std::complex<double>> B(5,3,6,4);
        Tensor<std::complex<double>> C;
        Tensor<std::complex<double>> Ctest(2,4,6);
        Ctest.fill(0.0);
        //enum {i,j,k,l,m,n};
        A.generate([](){ return randomCplx<double>(); });
        B.generate([](){ return randomCplx<double>(); });
        std::complex<double> alpha = randomCplx<double>();
        std::complex<double> beta = 0.0;
        contract(alpha,A,{'i','j','k'},B,{'k','j','l','m'},beta,C,{'i','m','l'});
        for(long i=0;i<A.extent(0);i++)
        for(long j=0;j<A.extent(1);j++)
        for(long k=0;k<A.extent(2);k++)
        for(long l=0;l<B.extent(2);l++)
        for(long m=0;m<B.extent(3);m++){
            Ctest(i,m,l)+=alpha*A(i,j,k)*B(k,j,l,m);
        }
        double res=0;
        for(auto i : C.range()) res+=std::abs(C(i)-Ctest(i));
        CHECK(res < eps_double);
        }

    SECTION("Complex Float Contraction")
        {
        Tensor<std::complex<float>> A(2,3,5);
        Tensor<std::complex<float>> B(5,3,6,4);
        Tensor<std::complex<float>> C;
        Tensor<std::complex<float>> Ctest(2,4,6);
        Ctest.fill(0.0);
        //enum {i,j,k,l,m,n};
        A.generate([](){ return randomCplx<float>(); });
        B.generate([](){ return randomCplx<float>(); });
        std::complex<float> alpha = randomCplx<float>();
        std::complex<float> beta = 0.0;
        contract(alpha,A,{'i','j','k'},B,{'k','j','l','m'},beta,C,{'i','m','l'});
        for(long i=0;i<A.extent(0);i++)
        for(long j=0;j<A.extent(1);j++)
        for(long k=0;k<A.extent(2);k++)
        for(long l=0;l<B.extent(2);l++)
        for(long m=0;m<B.extent(3);m++){
            Ctest(i,m,l)+=alpha*A(i,j,k)*B(k,j,l,m);
        }
        float res=0;
        for(auto i : C.range()) res+=std::abs(C(i)-Ctest(i));
        CHECK(res < eps_float);
        }

    }
