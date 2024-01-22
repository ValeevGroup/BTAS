//
// Created by Karl Pierce on 1/22/24.
//

#ifndef BTAS_GENERIC_HDF5_READ_WRITE_H
#define BTAS_GENERIC_HDF5_READ_WRITE_H

#ifdef BTAS_HAS_HighFive
#include <highfive/highfive.hpp>
namespace btas {
  template<typename Tensor>
  Tensor read_hdf5_to_tensor(std::string& filename){
    auto File = HighFive::File(filename);


  }
} // namespace btas
#endif //BTAS_HAS_HighFive

#endif  // BTAS_GENERIC_HDF5_READ_WRITE_H
