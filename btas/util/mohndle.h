//
// Created by Eduard Valeyev on 3/29/21.
//

#ifndef BTAS_UTIL_MOHNDLE_H
#define BTAS_UTIL_MOHNDLE_H

#include <btas/storage_traits.h>

#include <variant>
#include <functional>

namespace btas {

  /// @brief `mohndle` = Maybe Owning HaNDLE

  /// Enacpsulates a value, a reference, or a pointer to (bare, unique, or shared) to a contiguous storage
  template <typename Storage, typename = std::enable_if_t<is_storage<Storage>::value>>
  struct mohndle : std::variant<std::monostate, Storage, std::unique_ptr<Storage>, std::shared_ptr<Storage>,
                                  std::reference_wrapper<Storage>, Storage*> {
    using base_type = std::variant<std::monostate, Storage, std::unique_ptr<Storage>, std::shared_ptr<Storage>,
                                   std::reference_wrapper<Storage>, Storage*>;

    using base_type::base_type;

    typedef typename storage_traits<Storage>::value_type value_type;
    typedef typename storage_traits<Storage>::pointer pointer;
    typedef typename storage_traits<Storage>::const_pointer const_pointer;
    typedef typename storage_traits<Storage>::reference reference;
    typedef typename storage_traits<Storage>::const_reference const_reference;
    typedef typename storage_traits<Storage>::size_type size_type;
    typedef typename storage_traits<Storage>::difference_type difference_type;

    typedef typename storage_traits<Storage>::iterator iterator;
    typedef typename storage_traits<Storage>::const_iterator const_iterator;

    template <typename Arg>
    explicit mohndle(Arg&& arg) : base_type(std::forward<Arg>(arg)) {}
    mohndle() = default;
    mohndle(const mohndle&) = default;
    mohndle(mohndle&&) = default;
    mohndle& operator=(const mohndle&) = default;
    mohndle& operator=(mohndle&&) = default;
    ~mohndle() = default;

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
    std::enable_if_t<has_squarebraket_v<S> && !std::is_const_v<S>, reference> operator[](std::size_t ord) { return (*(this->get()))[ord]; }

    template <typename S = Storage>
    std::enable_if_t<has_squarebraket_v<S>, const_reference> operator[](std::size_t ord) const { return (*(this->get()))[ord]; }

    template <typename S, typename Enabler>
    friend void swap(mohndle<S>& first, mohndle<S>& second);

    const Storage* get() const {
      return std::visit(
          [](auto&& v) -> const Storage* {
            using v_t = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<v_t, Storage>) {
              return &v;
            } else if constexpr (std::is_same_v<v_t, std::reference_wrapper<Storage>>) {
              return &(v.get());
            } else if constexpr (std::is_same_v<v_t, Storage*>) {
              assert(v);
              return v;
            } else if constexpr (std::is_same_v<v_t, std::unique_ptr<Storage>>) {
              assert(v);
              return v.get();
            } else if constexpr (std::is_same_v<v_t, std::shared_ptr<Storage>>) {
              assert(v);
              return v.get();
            } else
              abort();
          },
          this->base());
    }
    Storage* get() { return const_cast<Storage*>(this->get()); }

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
