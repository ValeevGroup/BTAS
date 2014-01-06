/*
 * synopsis.C
 *
 *  Created on: Jan 3, 2014
 *      Author: evaleev
 */

#include <btas/btas.h>
#include <btas/varray/allocators.h>

using namespace btas;

template <typename T> using myvec = std::vector<T, stack_allocator<T>>;

int main(int argc, char **argv) {

  //! [Synopsis]
  auto O = Range1{2};              // [0, 2)
  auto U = Range1{2, 5};           // [2, 5)
  auto P = merge(O, U);            // [0, 5)
  auto OOUU = Range{O, O, U, U};   // direct product range
  auto PPPP = Range{P, P, P, P};

  typedef Tensor<double> DTensor;
  DTensor t2(OOUU);
  std::fill(t2.begin(), t2.end(), 1.0); // fill t2 with ones
  DTensor v(PPPP);
  std::fill(v.begin(), v.end(), 2.0); // fill v with twos

  // Tensor slices
  auto v_uuuu = v.slice({U, U, U, U});

  // Tensor contraction
  auto t2v = DTensor{};
  contract(1.0,
           t2, {'i','j','a','b'}, DTensor{v_uuuu}, {'a','b','c','d'},  // v_uuuu is copied to temporary
                                                                       // SOON: TensorViews will be usable interchangeably with Tensors
           0.0, t2v, {'i','j','c','d'});

  // Tensor is highly configurable. A few examples are shown below. See doc/examples/synopsis.C for more examples.
  {
    // usually one wants to deal with Tensors whose rank is known as compile time.
    typedef Tensor<double,
                   RangeNd<CblasRowMajor, std::array<long, 2> >,
                   std::vector<double> > DTensor2;               // rank-2 tensor (matrix) of doubles
    typedef Tensor<double,
                   RangeNd<CblasRowMajor, std::array<long, 3> >,
                   std::vector<double> > DTensor3;               // rank-3 tensor of doubles

    DTensor2 a(Range{O,U}, 0.0);   // OK
//  DTensor2 b(OOUU, 0.0);         // runtime error! rank-2 tensor initialized with rank-4 range
    DTensor3 c(Range{O,U,U}, 0.0); // OK

    // sometimes one wants to deal with tensors of fixed rank AND dimension
    typedef Tensor<double,
                   RangeNd<CblasRowMajor, std::array<long, 2> >,
                   std::array<double, 9> > DMatrix9;               // a 9-element matrix
    DMatrix9 d(Range{U,U}, 0.0);   // OK: Range.area == 9
//  DMatrix9 e(Range{P,U}, 0.0);   // runtime error: Range.area > 9
  }

  //! [Synopsis]

  // The default Tensor is a "dense" box of values, with internally managed storage
  // some users want to manually manage memory; here's how simple stack memory management can be done
  {
    // allocate 10^6 byte buffer for holding Tensors
    const auto bufsize = 1000000ul;
    auto mybuf = std::make_shared<stack_arena>(new char[bufsize], bufsize);
    auto dalloc = stack_allocator<double>              (mybuf);
    auto falloc = stack_allocator<float>               (mybuf);
    auto zalloc = stack_allocator<std::complex<double>>(mybuf);

    typedef std::vector<double, stack_allocator<double>>        dvec;    // vector of doubles allocated in the buffer
    typedef std::vector<float, stack_allocator<float>>          fvec;    // vector of floats allocated in the buffer
    typedef std::vector<std::complex<double>,
                        stack_allocator<std::complex<double>>>  zvec;    // vector of complex doubles allocated in the buffer
    typedef Tensor<double, Range,  dvec> DTensor; // tensor of doubles that uses memory in the buffer
    typedef Tensor<float,  Range,  fvec> FTensor; // tensor of floats that uses memory in the buffer
    typedef Tensor<std::complex<double>,
                           Range,  zvec> ZTensor; // tensor of complex doubles that uses memory in the buffer

    DTensor v_oouu(OOUU, dvec(dalloc));
    ZTensor v_pppp(PPPP, zvec(zalloc));
    FTensor r_oouu(OOUU, fvec(falloc));

    auto v_uuuu = v_pppp.slice({U, U, U, U});
  }

  return 0;
}



