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

  auto O = Range1{2};              // [0, 2)
  auto U = Range1{2, 5};           // [2, 5)
  auto P = merge(O, U);            // [0, 5)
  auto OOUU = Range{O, O, U, U};   // direct product range
  auto PPPP = Range{P, P, P, P};

  // "dense" Tensor
  typedef Tensor<double> DTensor;
  // make a few Tensors
  DTensor t2(OOUU);
  std::fill(t2.begin(), t2.end(), 1.0); // fill t2 with ones
  DTensor v(PPPP);
  std::fill(v.begin(), v.end(), 2.0); // fill v with twos

  // Tensor slices
  auto v_uuuu = v.slice({U, U, U, U});

  // Tensor contraction
  auto t2v = DTensor{};
  // contract(1.0, t2, {'i','j','a','b'}, v_uuuu, {'a','b','c','d'}, 1.0, t2v, {'i','j','c','d'}); // error, can't pass TensorViews yet
  contract(1.0, t2, {'i','j','a','b'}, DTensor{v_uuuu}, {'a','b','c','d'}, 1.0, t2v, {'i','j','c','d'});

  return 0;
}



