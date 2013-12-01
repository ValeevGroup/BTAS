/*
 * test_ta.C
 *
 *  Created on: Nov 25, 2013
 *      Author: evaleev
 */

#include <btas/varray.h>
#include <btas/tensor.h>

#include <tiled_array.h>

namespace TA = TiledArray;

namespace madness {
    namespace archive {

    template <class Archive, typename T>
    struct ArchiveLoadImpl<Archive,btas::varray<T> > {
        static inline void load(const Archive& ar, btas::varray<T>& x) {
          typename btas::varray<T>::size_type n; ar & n;
          x.resize(n);
          for (typename btas::varray<T>::value_type& xi : x)
            ar & xi;
        }
    };

    template <class Archive, typename T>
    struct ArchiveStoreImpl<Archive,btas::varray<T> > {
        static inline void store(const Archive& ar, btas::varray<T>& x) {
          ar & x.size();
          for (const typename btas::varray<T>::value_type& xi : x)
            ar & xi;
        }
    };

    template <class Archive,
              typename _T,
              class _Container,
              class _Shape>
    struct ArchiveSerializeImpl<Archive, btas::Tensor<_T,_Container,_Shape> > {
        static inline void serialize(const Archive& ar, btas::Tensor<_T,_Container,_Shape>& t) {
        }
    };

    }
}


int main(int argc, char **argv) {

  // Initialize runtime
  madness::World& world = madness::initialize(argc, argv);

  // Construct TiledRange
  std::vector<unsigned int> blocking;
  const size_t num_blocks = 5;
  const size_t dimension_size = 100;
  const size_t block_size = dimension_size / num_blocks;
  blocking.reserve(num_blocks + 1);
  for(std::size_t i = 0; i <= dimension_size; i += block_size)
    blocking.push_back(i);

  std::vector<TA::TiledRange1> blocking2(2,
      TA::TiledRange1(blocking.begin(), blocking.end()));

  TA::TiledRange
    trange(blocking2.begin(), blocking2.end());

//  typedef btas::Tensor<double, btas::varray<double>, TA::Range > DenseTensor;
  typedef btas::Tensor<double, btas::varray<double> > DenseTensor;
//  typedef TA::Array<double, 2, btas::varray<double> > TArray;
  typedef TA::Array<double, 2, DenseTensor > TArray;

  TArray a(world, trange);
  TArray b(world, trange);
  TArray c(world, trange);
  a.set_all_local(1.0);
#if 0
  b.set_all_local(1.0);

  // Start clock
  world.gop.fence();
  const double wall_time_start = madness::wall_time();

  // Do matrix multiplication
  {
    c("m,n") = a("m,k") * b("k,n");
    world.gop.fence();
    if(world.rank() == 0)
      std::cout << "done\n";
  }
#endif

  madness::finalize();
  return 0;

}
