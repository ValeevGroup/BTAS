/*
 * synopsis.C
 *
 *  Created on: Jan 3, 2014
 *      Author: evaleev
 */

#include <btas/tensor.h>
#include <btas/generic/contract.h>

using namespace btas;

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

  // The default Tensor is a "dense" box of values, with internally managed storage
  // But Tensor is highly configurable
  // For example, it can be used with externally-managed storage
  {
    auto mybuf = std::vector<double>{OOUU.area()};
    typedef Tensor<double, Range, StorageRef<std::vector<double> >  > DTensor;
    auto v_oouu = DTensor(OOUU, ref);
  }
  //! [Synopsis]

  return 0;
}



