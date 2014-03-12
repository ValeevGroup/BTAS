/*
 * test_ta.C
 *
 *  Created on: Nov 25, 2013
 *      Author: evaleev
 */

#include <btas/varray/varray.h>
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
              class _Range,
              class _Store>
    struct ArchiveSerializeImpl<Archive, btas::Tensor<_T,_Range,_Store> > {
        static inline void serialize(const Archive& ar, btas::Tensor<_T,_Range,_Store>& t) {
        }
    };

    }
}

namespace TiledArray {
  template<typename _T,
           class _Range,
           class _Store>
  struct expression_traits<btas::Tensor<_T,_Range,_Store> > {
      typedef btas::Tensor<_T,_Range,_Store> eval_type;
  };
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

  typedef btas::Tensor<double, TA::Range, btas::varray<double>> DenseTensor;
  typedef TA::Array<double, 2, DenseTensor > TArray;

  TArray a(world, trange);
  TArray b(world, trange);
  TArray c(world, trange);
  a.set_all_local(1.0);
  b.set_all_local(1.0);

  // Start clock
  world.gop.fence();
  const double wall_time_start = madness::wall_time();

  // Do matrix multiplication
  {
    // currently TA ContractionResult is hardwired to use TA::Tensor ... is likely fixed on expressions branch
    //c("m,n") = a("m,k") * b("k,n");
    auto c = a("m,k") * b("k,n"); // does not actually compute c!
    world.gop.fence();
    if(world.rank() == 0)
      std::cout << "done\n";
  }

  const double wall_time_finish = madness::wall_time();

  std::cout << "elapsed time = " << wall_time_finish - wall_time_start << " sec" << std::endl;

  madness::finalize();
  return 0;

}
