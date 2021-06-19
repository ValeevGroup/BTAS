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
  struct mohndle : std::variant<std::monostate, Storage, std::unique_ptr<Storage>, std::shared_ptr<Storage>,
                                std::reference_wrapper<Storage>, Storage*> {
    using base_type = std::variant<std::monostate, Storage, std::unique_ptr<Storage>, std::shared_ptr<Storage>,
                                   std::reference_wrapper<Storage>, Storage*>;

    //using base_type::base_type;

    typedef typename storage_traits<Storage>::value_type value_type;
    typedef typename storage_traits<Storage>::pointer pointer;
    typedef typename storage_traits<Storage>::const_pointer const_pointer;
    typedef typename storage_traits<Storage>::reference reference;
    typedef typename storage_traits<Storage>::const_reference const_reference;
    typedef typename storage_traits<Storage>::size_type size_type;
    typedef typename storage_traits<Storage>::difference_type difference_type;

    typedef typename storage_traits<Storage>::iterator iterator;
    typedef typename storage_traits<Storage>::const_iterator const_iterator;

    /// constructs mohndle from a handle

    /// @param handle a handle object
    template <typename Handle, typename = std::enable_if_t<std::is_constructible_v<base_type,Handle&&>>>
    explicit mohndle(Handle&& handle) : base_type(std::forward<Handle>(handle)) {}

    mohndle(const mohndle& other) : base_type(std::visit(
          [](auto&& v) -> base_type {
            using v_t = std::remove_reference_t<decltype(v)>;
            if constexpr (std::is_same_v<v_t, Storage> || std::is_same_v<v_t, Storage const> || std::is_same_v<v_t, std::reference_wrapper<Storage>> || std::is_same_v<v_t, std::reference_wrapper<Storage> const> || std::is_same_v<v_t, Storage*> || std::is_same_v<v_t, Storage* const> || std::is_same_v<v_t, std::shared_ptr<Storage>> || std::is_same_v<v_t, std::shared_ptr<Storage> const>) {
              return v;
            } else if constexpr (std::is_same_v<v_t, std::unique_ptr<Storage>> || std::is_same_v<v_t, std::unique_ptr<Storage> const>) {
              return std::make_unique<Storage>(*(v.get()));
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
    template <typename ... Args, typename = std::enable_if_t<std::is_constructible_v<Storage, Args&&...>>>
    explicit mohndle(Args&& ... args) {
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
    std::enable_if_t<has_nonmember_begin_v<S> && !std::is_const_v<S>, iterator> begin() { using std::begin; return begin(*(this->get())); }

    template <typename S = Storage>
    std::enable_if_t<has_nonmember_end_v<S> && !std::is_const_v<S>, iterator> end() { using std::end; return end(*(this->get())); }

    template <typename S = Storage>
    std::enable_if_t<has_nonmember_begin_v<S>, const_iterator> begin() const { using std::begin; return begin(*(this->get())); }

    template <typename S = Storage>
    std::enable_if_t<has_nonmember_end_v<S>, const_iterator> end() const { using std::end; return end(*(this->get())); }

    template <typename S = Storage>
    std::enable_if_t<has_nonmember_data_v<S> && !std::is_const_v<S>, pointer> data() { using std::data; return data(*(this->get())); }

    template <typename S = Storage>
    std::enable_if_t<has_nonmember_data_v<S>, const_pointer> data() const { using std::data; return data(*(this->get())); }

    template <typename S = Storage>
    std::enable_if_t<has_nonmember_size_v<S>, std::size_t> size() const { using std::size; return size(*(this->get())); }

    template <typename S = Storage>
    void resize(std::size_t new_size) { return array_adaptor<Storage>::resize(*(this->get()), new_size); }

    template <typename S = Storage>
    std::enable_if_t<has_squarebraket_v<S> && !std::is_const_v<S>, reference> operator[](std::size_t ord) { return (*(this->get()))[ord]; }

    template <typename S = Storage>
    std::enable_if_t<has_squarebraket_v<S>, const_reference> operator[](std::size_t ord) const { return (*(this->get()))[ord]; }

    template <typename S, typename Enabler>
    friend void swap(mohndle<S>& first, mohndle<S>& second);

    const Storage* get() const {
      return std::visit(
          [](auto&& v) -> const Storage* {
            using v_t = std::remove_reference_t<decltype(v)>;
            if constexpr (std::is_same_v<v_t, Storage>) {
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
    void serialize(Archive& ar, const unsigned int = 0) {
      std::visit(
          [&ar](auto&& v) -> void {
            using v_t = std::remove_reference_t<decltype(v)>;
            if constexpr (std::is_same_v<v_t, Storage> || std::is_same_v<v_t, Storage const> ||
                          std::is_same_v<v_t, Storage*> || std::is_same_v<v_t, Storage* const> ||
                          std::is_same_v<v_t, std::unique_ptr<Storage>> ||
                          std::is_same_v<v_t, std::unique_ptr<Storage> const> ||
                          std::is_same_v<v_t, std::shared_ptr<Storage>> ||
                          std::is_same_v<v_t, std::shared_ptr<Storage> const>) {
              ar& v;
            } else if constexpr (std::is_same_v<v_t, std::reference_wrapper<Storage>> ||
                                 std::is_same_v<v_t, std::reference_wrapper<Storage> const>) {
              abort();
            } else
              abort();
          },
          this->base());
    }

   private:
    auto& base() { return static_cast<base_type&>(*this); }
    const auto& base() const { return static_cast<const base_type&>(*this); }

    template <typename Storage_, typename>
    friend void swap(mohndle<Storage_>& first, mohndle<Storage_>& second);
  };

  template <typename Storage, typename = std::enable_if_t<!std::is_const_v<Storage>>>
  void swap(mohndle<Storage>& first, mohndle<Storage>& second) {
    using std::swap;
    swap(first.base(), second.base());
  }
}

#endif  // BTAS_UTIL_MOHNDLE_H
