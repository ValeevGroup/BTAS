//
// Created by Karl Pierce on 5/16/19.
//

#ifndef BTAS_GENERIC_DEFAULT_RANDOM_SEED_H
#define BTAS_GENERIC_DEFAULT_RANDOM_SEED_H
namespace btas{
  static unsigned int& random_seed_accessor(){
    static unsigned int value = 3;
    return value;
  }
} //namespace btas
#endif //BTAS_GENERIC_DEFAULT_RANDOM_SEED_H
