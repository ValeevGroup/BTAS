#include "test.h"

#include <iostream>

#include "btas/util/mohndle.h"
#include "btas/varray/varray.h"

#include <fstream>

#ifdef BTAS_HAS_BOOST_SERIALIZATION
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#endif  // BTAS_HAS_BOOST_SERIALIZATION

using std::cout;
using std::endl;
using namespace btas;

template <typename T, typename Enabler = void>
constexpr inline bool storage_is_movable = true;
template <typename T, std::size_t N>
constexpr inline bool storage_is_movable<std::array<T, N>> = false;
template <typename T, std::size_t N>
constexpr inline bool storage_is_movable<const std::array<T, N>> = false;

TEST_CASE("mohndle")
{

  // test storage_traits
  static_assert(std::is_same_v<typename storage_traits<mohndle<std::vector<double>>>::template rebind_t<float>, mohndle<std::vector<float>>>);

  SECTION("core")
  {
    auto test = [](auto&& storage) {
      using storage_t = std::remove_reference_t<decltype(storage)>;
      using mohndle_t = mohndle<storage_t>;

      CHECK_NOTHROW(mohndle_t{});
      mohndle_t v1;
      CHECK(!v1);
      CHECK(!(v1.is_owner()));

      CHECK_NOTHROW(mohndle_t{std::ref(storage)});
      mohndle_t v2{std::ref(storage)};
      CHECK(static_cast<bool>(v2));
      CHECK(!v2.is_owner());
      CHECK(v2.begin() == storage.begin());
      CHECK(v2.end() == storage.end());
      CHECK(v2.data() == storage.data());
      CHECK(v2.size() == storage.size());

      CHECK_NOTHROW(mohndle_t{&storage});
      mohndle_t v3{&storage};
      CHECK(static_cast<bool>(v3));
      CHECK(!v3.is_owner());
      CHECK(v3.begin() == storage.begin());
      CHECK(v3.end() == storage.end());
      CHECK(v3.data() == storage.data());
      CHECK(v3.size() == storage.size());

      auto uptr_to_copy = std::make_unique<storage_t>(storage);
      const auto v4_begin_ref = uptr_to_copy->begin();
      const auto v4_end_ref = uptr_to_copy->end();
      const auto v4_data_ref = uptr_to_copy->data();
      const auto v4_size_ref = uptr_to_copy->size();
      CHECK_NOTHROW(mohndle_t{std::make_unique<storage_t>(storage)});
      mohndle_t v4{std::move(uptr_to_copy)};
      CHECK(static_cast<bool>(v4));
      CHECK(v4.is_owner());
      CHECK(v4.begin() == v4_begin_ref);
      CHECK(v4.end() == v4_end_ref);
      CHECK(v4.data() == v4_data_ref);
      CHECK(v4.size() == v4_size_ref);

      auto sptr_to_copy = std::make_shared<storage_t>(storage);
      CHECK_NOTHROW(mohndle_t{sptr_to_copy});
      mohndle_t v5{sptr_to_copy};
      CHECK(static_cast<bool>(v5));
      CHECK(v5.is_owner());
      CHECK(v5.begin() == sptr_to_copy->begin());
      CHECK(v5.end() == sptr_to_copy->end());
      CHECK(v5.data() == sptr_to_copy->data());
      CHECK(v5.size() == sptr_to_copy->size());

      auto storage_copy = storage;
      const auto v6_begin_ref = storage_copy.begin();
      const auto v6_end_ref = storage_copy.end();
      const auto v6_data_ref = storage_copy.data();
      const auto v6_size_ref = storage_copy.size();
      CHECK_NOTHROW(mohndle_t{std::move(storage)});
      mohndle_t v6{std::move(storage_copy)};
      CHECK(static_cast<bool>(v6));
      CHECK(v6.is_owner());
      if constexpr (storage_is_movable<storage_t>) {
        CHECK(v6.begin() == v6_begin_ref);
        CHECK(v6.end() == v6_end_ref);
        CHECK(v6.data() == v6_data_ref);
      }
      else {
        CHECK(v6.begin() != v6_begin_ref);
        CHECK(v6.end() != v6_end_ref);
        CHECK(v6.data() != v6_data_ref);
      }
      CHECK(v6.size() == v6_size_ref);

      if constexpr (!std::is_const_v<storage_t>) {
        CHECK_NOTHROW(swap(v1, v6));
        CHECK(!v6);
        CHECK(!v6.is_owner());
        CHECK(static_cast<bool>(v1));
        CHECK(v1.is_owner());
        if constexpr (storage_is_movable<storage_t>) {
          CHECK(v1.begin() == v6_begin_ref);
          CHECK(v1.end() == v6_end_ref);
          CHECK(v1.data() == v6_data_ref);
        }
        CHECK(v1.size() == v6_size_ref);
      }
    };

    test(varray<double>{5});
    test(std::vector<double>{5});
    test(std::array<double, 5>{});

    varray<double> v0{5};
    test(static_cast<const varray<double>&>(v0));
    std::vector<double> v1{5};
    test(static_cast<const std::vector<double>&>(v1));
    std::array<double, 5> v2{};
    test(static_cast<const std::array<double, 5>&>(v2));
  }

#ifdef BTAS_HAS_BOOST_SERIALIZATION
  SECTION("Serialization") {
    const auto archive_fname = "mohndle.serialization.archive";

    using stdvecshr_t = btas::mohndle<std::vector<double>, btas::Handle::shared_ptr>;
    using varrayshr_t = btas::mohndle<btas::varray<double>, btas::Handle::shared_ptr>;
    stdvecshr_t m0(std::vector<double>{1., 2., 3.});
    varrayshr_t m1(btas::varray<double>{-1., -2.});
    // write
    {
      std::ofstream os(archive_fname);
      assert(os.good());
      boost::archive::xml_oarchive ar(os);
      CHECK_NOTHROW(ar << BOOST_SERIALIZATION_NVP(m0));
      CHECK_NOTHROW(ar << BOOST_SERIALIZATION_NVP(m1));
    }
    // read
    {
      std::ifstream is(archive_fname);
      assert(is.good());
      boost::archive::xml_iarchive ar(is);

      stdvecshr_t m0copy;
      CHECK_NOTHROW(ar >> BOOST_SERIALIZATION_NVP(m0copy));
      CHECK(m0 == m0copy);

      varrayshr_t m1copy;
      CHECK_NOTHROW(ar >> BOOST_SERIALIZATION_NVP(m1copy));
      CHECK(m1 == m1copy);

    }
    std::remove(archive_fname);
  }
#endif  // BTAS_HAS_BOOST_SERIALIZATION

}
