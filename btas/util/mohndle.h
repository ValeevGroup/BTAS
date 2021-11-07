//
// Created by Eduard Valeyev on 3/29/21.
//

#ifndef BTAS_UTIL_MOHNDLE_H
#define BTAS_UTIL_MOHNDLE_H

#include <btas/storage_traits.h>
#include <btas/array_adaptor.h>

#include <btas/serialization.h>
#ifdef BTAS_HAS_BOOST_SERIALIZATION
# include <boost/serialization/unique_ptr.hpp>
# include <boost/serialization/shared_ptr.hpp>
# include <boost/archive/binary_iarchive.hpp>
# include <boost/archive/binary_oarchive.hpp>
#endif

#include <variant>
#include <functional>

namespace btas {

  /// describes handle types that can be used for default/direct construction of mohndle
  enum class Handle {
    invalid,
    value,
    unique_ptr,
    shared_ptr,
    ptr
  };

  /// @brief Maybe Owning HaNDLE (`mohndle`) to @c Storage

  /// @tparam Storage a type that meets the TWG.Storage concept
  /// @tparam DefaultHandle the handle type to use when default constructing, or constructing storage object directly from a pack
  /// Enacpsulates a value, a reference, or a pointer to (bare, unique, or shared) to a contiguous storage
  template <typename Storage, Handle DefaultHandle = Handle::invalid, typename = std::enable_if_t<is_storage<Storage>::value>>
  class mohndle : std::variant<std::monostate, Storage, std::unique_ptr<Storage>, std::shared_ptr<Storage>,
                               std::reference_wrapper<Storage>, Storage*> {
   public:
    using base_type = std::variant<std::monostate, Storage, std::unique_ptr<Storage>, std::shared_ptr<Storage>,
                                   std::reference_wrapper<Storage>, Storage*>;

    // using base_type::base_type;

    typedef typename storage_traits<Storage>::value_type value_type;
    typedef typename storage_traits<Storage>::pointer pointer;
    typedef typename storage_traits<Storage>::const_pointer const_pointer;
    typedef typename storage_traits<Storage>::reference reference;
    typedef typename storage_traits<Storage>::const_reference const_reference;
    typedef typename storage_traits<Storage>::size_type size_type;
    typedef typename storage_traits<Storage>::difference_type difference_type;

    typedef typename storage_traits<Storage>::iterator iterator;
    typedef typename storage_traits<Storage>::const_iterator const_iterator;

    mohndle() = default;

    /// constructs mohndle from a handle

    /// @param handle a handle object
    template <typename Handle, typename = std::enable_if_t<std::is_constructible_v<base_type, Handle&&>>>
    explicit mohndle(Handle&& handle) : base_type(std::forward<Handle>(handle)) {}

    mohndle(const mohndle& other)
        : base_type(std::visit(
              [](auto&& v) -> base_type {
                using v_t = std::remove_reference_t<decltype(v)>;
                if constexpr (std::is_same_v<v_t, Storage> || std::is_same_v<v_t, Storage const> ||
                              std::is_same_v<v_t, std::reference_wrapper<Storage>> ||
                              std::is_same_v<v_t, std::reference_wrapper<Storage> const> ||
                              std::is_same_v<v_t, Storage*> || std::is_same_v<v_t, Storage* const> ||
                              std::is_same_v<v_t, std::shared_ptr<Storage>> ||
                              std::is_same_v<v_t, std::shared_ptr<Storage> const>) {
                  return v;
                } else if constexpr (std::is_same_v<v_t, std::unique_ptr<Storage>> ||
                                     std::is_same_v<v_t, std::unique_ptr<Storage> const>) {
                  return std::make_unique<Storage>(*(v.get()));
                } else if constexpr (std::is_same_v<v_t, std::monostate> ||
                                     std::is_same_v<v_t, std::monostate const>) {
                  return {};
                } else
                  abort();
              },
              other.base())) {}

    mohndle(mohndle&&) = default;

    mohndle& operator=(const mohndle& other) {
      std::swap(this->base(), mohndle(other).base());
      return *this;
    }

    mohndle& operator=(mohndle&&) = default;
    ~mohndle() = default;

    /// constructs a mohndle of type given by DefaultHandle directly from zero or more arguments
    template <typename... Args, typename = std::enable_if_t<std::is_constructible_v<Storage, Args&&...>>>
    explicit mohndle(Args&&... args) {
      if constexpr (DefaultHandle == Handle::value)
        this->base().template emplace<Storage>(std::forward<Args>(args)...);
      else if constexpr (DefaultHandle == Handle::ptr)
        this->base().template emplace<Storage*>(new Storage(std::forward<Args>(args)...));
      else if constexpr (DefaultHandle == Handle::unique_ptr)
        this->base().template emplace<std::unique_ptr<Storage>>(std::make_unique<Storage>(std::forward<Args>(args)...));
      else if constexpr (DefaultHandle == Handle::shared_ptr)
        this->base().template emplace<std::shared_ptr<Storage>>(std::make_shared<Storage>(std::forward<Args>(args)...));
      else  // if constexpr (DefaultHandle == Handle::invalid)
        abort();
    }

    explicit operator bool() const { return this->index() != 0; }

    bool is_owner() const {
      const auto idx = this->index();
      return idx > 0 && idx < 4;
    }

    template <typename S = Storage>
    std::enable_if_t<has_nonmember_begin_v<S> && !std::is_const_v<S>, iterator> begin() {
      using std::begin;
      return begin(*(this->get()));
    }

    template <typename S = Storage>
    std::enable_if_t<has_nonmember_end_v<S> && !std::is_const_v<S>, iterator> end() {
      using std::end;
      return end(*(this->get()));
    }

    template <typename S = Storage>
    std::enable_if_t<has_nonmember_begin_v<S>, const_iterator> begin() const {
      using std::begin;
      return begin(*(this->get()));
    }

    template <typename S = Storage>
    std::enable_if_t<has_nonmember_end_v<S>, const_iterator> end() const {
      using std::end;
      return end(*(this->get()));
    }

    template <typename S = Storage>
    std::enable_if_t<has_nonmember_begin_v<S>, const_iterator> cbegin() const {
      return this->begin();
    }

    template <typename S = Storage>
    std::enable_if_t<has_nonmember_end_v<S>, const_iterator> cend() const {
      return this->end();
    }

    template <typename S = Storage>
    std::enable_if_t<has_nonmember_data_v<S> && !std::is_const_v<S>, pointer> data() {
      using std::data;
      return data(*(this->get()));
    }

    template <typename S = Storage>
    std::enable_if_t<has_nonmember_data_v<S>, const_pointer> data() const {
      using std::data;
      return data(*(this->get()));
    }

    template <typename S = Storage>
    std::enable_if_t<has_nonmember_size_v<S>, std::size_t> size() const {
      using std::size;
      return size(*(this->get()));
    }

    template <typename S = Storage>
    void resize(std::size_t new_size) {
      if (this->base().index() == 0 && new_size > 0)
        *this = mohndle(new_size);
      else
        array_adaptor<Storage>::resize(*(this->get()), new_size);
    }

    template <typename S = Storage>
    std::enable_if_t<has_squarebraket_v<S> && !std::is_const_v<S>, reference> operator[](std::size_t ord) {
      return (*(this->get()))[ord];
    }

    template <typename S = Storage>
    std::enable_if_t<has_squarebraket_v<S>, const_reference> operator[](std::size_t ord) const {
      return (*(this->get()))[ord];
    }

    template <typename S, typename Enabler>
    friend void swap(mohndle<S>& first, mohndle<S>& second);

    const Storage* get() const {
      return std::visit(
          [](auto&& v) -> const Storage* {
            using v_t = std::remove_reference_t<decltype(v)>;
            if constexpr (std::is_same_v<v_t, std::monostate> || std::is_same_v<v_t, std::monostate const>) {
              return &null_storage_;
            } else if constexpr (std::is_same_v<v_t, Storage>) {
              return &v;
            } else if constexpr (std::is_same_v<v_t, Storage const>) {
              return &v;
            } else if constexpr (std::is_same_v<v_t, std::reference_wrapper<Storage>>) {
              return &(v.get());
            } else if constexpr (std::is_same_v<v_t, std::reference_wrapper<Storage> const>) {
              return &(v.get());
            } else if constexpr (std::is_same_v<v_t, Storage*>) {
              assert(v);
              return v;
            } else if constexpr (std::is_same_v<v_t, Storage* const>) {
              assert(v);
              return v;
            } else if constexpr (std::is_same_v<v_t, std::unique_ptr<Storage>>) {
              assert(v);
              return v.get();
            } else if constexpr (std::is_same_v<v_t, std::unique_ptr<Storage> const>) {
              assert(v);
              return v.get();
            } else if constexpr (std::is_same_v<v_t, std::shared_ptr<Storage>>) {
              assert(v);
              return v.get();
            } else if constexpr (std::is_same_v<v_t, std::shared_ptr<Storage> const>) {
              assert(v);
              return v.get();
            } else
              abort();
          },
          this->base());
    }
    Storage* get() { return const_cast<Storage*>(const_cast<const mohndle*>(this)->get()); }

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int /* version */) {
      constexpr bool writing = std::is_base_of_v<boost::archive::detail::basic_oarchive, Archive>;
      constexpr auto serializable_index = std::index_sequence<0, 1, 2, 3, 5>{};

      auto index = this->base().index();

      // abort if trying to store an unsupported case
      if constexpr (writing) {
        if (std::holds_alternative<std::reference_wrapper<Storage>>(this->base()))
          abort();
      }
      ar& BOOST_SERIALIZATION_NVP(index);
      if constexpr (writing)
        std::visit(
            [&ar](const auto& value) -> void {
              using v_t = std::decay_t<decltype(value)>;
              // - can't read reference_wrapper
              // - no need to write monostate
              if constexpr (!std::is_same_v<v_t, std::reference_wrapper<Storage>> && !std::is_same_v<v_t, std::monostate>)
                ar & BOOST_SERIALIZATION_NVP(value);
            },
            this->base());
      else
        variant_load_impl(ar, this->base(), index, serializable_index);
    }

    auto& base() { return static_cast<base_type&>(*this); }
    const auto& base() const { return static_cast<const base_type&>(*this); }

    bool operator==(const mohndle& other) const {
      return (*this && other) || (!*this && !other && *(this->get()) == *(other.get()));
    }

   private:
    template <typename Storage_, typename>
    friend void swap(mohndle<Storage_>& first, mohndle<Storage_>& second);

    // utility for serializing select members of variant
    template <typename Archive, typename... Ts, std::size_t I0, std::size_t... Is>
    static Archive& variant_load_impl(Archive& ar, std::variant<Ts...>& v, std::size_t which, std::index_sequence<I0, Is...>) {
      constexpr bool writing = std::is_base_of_v<boost::archive::detail::basic_oarchive, Archive>;
      static_assert(!writing);
      if (which == I0) {
        using type = std::variant_alternative_t<I0, std::variant<Ts...>>;
        if constexpr (!std::is_same_v<type, std::monostate>) {
          type value;
          ar& BOOST_SERIALIZATION_NVP(value);
          v.template emplace<I0>(std::move(value));
        }
        else if constexpr (std::is_same_v<type, std::monostate>)
          v = {};
      } else {
        if constexpr (sizeof...(Is) == 0)
          throw std::logic_error("btas::mohndle::variant_load_impl(ar,v,idx,idxs): idx is not present in idxs");
        else
          return variant_load_impl(ar, v, which, std::index_sequence<Is...>{});
      }
      return ar;
    }

    inline static Storage null_storage_ = {};  // used if this is null
  };

  template <typename Storage, typename = std::enable_if_t<!std::is_const_v<Storage>>>
  void swap(mohndle<Storage>& first, mohndle<Storage>& second) {
    using std::swap;
    swap(first.base(), second.base());
  }

  /// mohndle can have shallow copy semantics
  template <typename S, Handle H>
  constexpr inline bool is_deep_copy_v<mohndle<S,H>> = false;

}  // namespace btas

// serialization to/fro MADNESS archive (github.com/m-a-d-n-e-s-s/madness)
namespace madness::archive {

  template <class Archive, typename Storage, btas::Handle DefaultHandle>
  struct ArchiveLoadImpl<Archive, btas::mohndle<Storage, DefaultHandle>> {
    static inline void load(const Archive& ar, btas::mohndle<Storage, DefaultHandle>& t) {
      constexpr auto serializable_index = std::index_sequence<0, 1, 2, 3, 5>{};
      auto index = t.base().index();
      ar& index;
      variant_load_impl(ar, t.base(), index, serializable_index);
    }

    template <typename T>
    struct value_type {
      using type = T;
    };
    template <typename T>
    struct value_type<T*> {
      using type = T;
    };
    template <typename T>
    struct value_type<std::unique_ptr<T>> {
      using type = T;
    };
    template <typename T>
    struct value_type<std::shared_ptr<T>> {
      using type = T;
    };

    // utility for serializing select members of variant
    template <typename... Ts, std::size_t I0, std::size_t... Is>
    static const Archive& variant_load_impl(const Archive& ar, std::variant<Ts...>& v, std::size_t which,
                                      std::index_sequence<I0, Is...>) {
      if (which == I0) {
        using type = std::variant_alternative_t<I0, std::variant<Ts...>>;
        if constexpr (!std::is_same_v<type, std::monostate>) {
          type value;
          if constexpr (std::is_same_v<type, Storage>) {
            ar& value;
          } else {  // bare or smart ptr
            using v_t = typename value_type<type>::type;
            std::allocator<v_t> alloc;  // instead use the allocator associated with the archive?
            auto* buf = alloc.allocate(sizeof(v_t));
            v_t* ptr = new (buf) v_t;
            ar& *ptr;
            value = type(ptr);
          }
          v.template emplace<I0>(std::move(value));
        } else if constexpr (std::is_same_v<type, std::monostate>)
          v = {};
      } else {
        if constexpr (sizeof...(Is) == 0)
          throw std::logic_error("btas::mohndle::variant_load_impl(ar,v,idx,idxs): idx is not present in idxs");
        else
          return variant_load_impl(ar, v, which, std::index_sequence<Is...>{});
      }
      return ar;
    }
  };

  template <class Archive, typename Storage, btas::Handle DefaultHandle>
  struct ArchiveStoreImpl<Archive, btas::mohndle<Storage, DefaultHandle>> {
    static inline void store(const Archive& ar, const btas::mohndle<Storage, DefaultHandle>& t) {
      constexpr auto serializable_index = std::index_sequence<0, 1, 2, 3, 5>{};
      const auto index = t.base().index();

      // abort if trying to store an unsupported case
      if (std::holds_alternative<std::reference_wrapper<Storage>>(t.base())) abort();
      ar& index;
      std::visit(
          [&ar](const auto& v) -> void {
            using v_t = std::decay_t<decltype(v)>;
            // - can't read reference_wrapper
            // - no need to write monostate
            if constexpr (!std::is_same_v<v_t, std::reference_wrapper<Storage>> &&
                          !std::is_same_v<v_t, std::monostate>) {
              if constexpr (std::is_same_v<v_t, Storage>) {
                ar& v;
              } else {
                ar& *v;
              }
            }
          },
          t.base());
    }
  };

}  // namespace madness::archive


#endif  // BTAS_UTIL_MOHNDLE_H
