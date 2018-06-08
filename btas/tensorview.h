/*
 * tensorview.h
 *
 *  Created on: Dec 28, 2013
 *      Author: evaleev
 */

#ifndef BTAS_TENSORVIEW_H_
#define BTAS_TENSORVIEW_H_

#include <functional>

#include <btas/range.h>
#include <btas/storage_traits.h>
#include <btas/util/sequence_adaptor.h>
#include <btas/tensorview_iterator.h>
#include <btas/defaults.h>
#include <btas/util/functional.h>
#include <btas/error.h>

namespace btas {

  // \internal btas::TensorView has a policy that configures whether to  to support constexpr mutability (as part of the type), or trackable at runtime.
  // Such design allows to reduce code bloat due to the need to instantiate the code using views for const and non-const type variants; it also
  // makes it easier to use view by avoiding the need for metaprogramming when dealing with views with constexpr mutability. The
  // runtime tracking of mutability incurs extra storage overhead (TensorView will have an extra bool member) and extra performance overhead due to
  // extra runtime logic; the runtime overhead can be avoided by disabling BTAS_ASSERT after testing. Expert users can avoid storage overhead by
  // using constexpr mutability tracking.


  enum TensorViewPolicy_ConstnessPolicy {
    TensorViewPolicy_RuntimeConst = 1,
    TensorViewPolicy_CompiletimeConst = 0
  };

  /// TensorViewPolicy configures behavior of certain features of TensorView
  /// \tparam runtimeconst If true, constness of data access is checked at runtime. This involves
  ///      extra space overhead (enough to store a boolean readwrite flag). Non-const data access members
  ///      will also check whether readwrite is set using BTAS_ASSERT (hence runtime overhead can be eliminated after
  ///      testing. This feature is needed if you want to use a single TensorView<T,Range,Storage> type
  ///      for mutable (non-const) and immutable (const) views.
  template <TensorViewPolicy_ConstnessPolicy ConstnessPolicy = TensorViewPolicy_CompiletimeConst>
  struct TensorViewPolicy {
      /// true if constness tracked at runtime
      static constexpr bool runtimeconst = (ConstnessPolicy == TensorViewPolicy_RuntimeConst);
  };

  namespace detail {

  struct bool_wrapper {
    bool value;
    bool_wrapper() = default;
    bool_wrapper(const bool_wrapper& other) = default;
    bool_wrapper(bool_wrapper&& other) = default;
    inline bool_wrapper(bool b) : value(b) {}
    inline operator bool() const noexcept { return value; }
    inline bool operator()() const noexcept { return value; }
  };
  inline bool operator==(const bool_wrapper& one, const bool_wrapper& two) {
    return one.value == two.value;
  }

  /// Helper class to implement the constness logic, as well as to guarantee empty
  /// base optimization for constexpr constness policy.
  template <typename Policy, typename Storage>
  struct TensorViewMutabilityImpl
      : public std::conditional<
            std::is_same<Policy, TensorViewPolicy<
                                     TensorViewPolicy_RuntimeConst>>::value &&
                not std::is_const<Storage>::value,
            bool_wrapper,
            btas::detail::bool_type<not std::is_const<Storage>::value>>::type {
    using impl_type = typename std::conditional<
        std::is_same<Policy, TensorViewPolicy<
                                 TensorViewPolicy_RuntimeConst>>::value &&
            not std::is_const<Storage>::value,
        bool_wrapper,
        btas::detail::bool_type<not std::is_const<Storage>::value>>::type;

    /// By default, make TensorView mutable if Storage is mutable and Policy is constexpr.
    /// @return a default TensorViewWritable<Policy,Storage>
    TensorViewMutabilityImpl() : TensorViewMutabilityImpl(make_default()) {}

    TensorViewMutabilityImpl(TensorViewMutabilityImpl& other) : impl_type(other) {}
    TensorViewMutabilityImpl(const TensorViewMutabilityImpl&) : impl_type(false) {}
    TensorViewMutabilityImpl& operator=(TensorViewMutabilityImpl& other) {
      *this = other;
      return *this;
    }
    TensorViewMutabilityImpl& operator=(const TensorViewMutabilityImpl& other) {
      *this = impl_type(false);
      return *this;
    }
    TensorViewMutabilityImpl(TensorViewMutabilityImpl&&) = default;
    TensorViewMutabilityImpl& operator=(TensorViewMutabilityImpl&&) = default;

    TensorViewMutabilityImpl(bool is_mutable) : impl_type(is_mutable) {}

    static constexpr TensorViewMutabilityImpl make_default() {
      return TensorViewMutabilityImpl(std::is_same<Policy, TensorViewPolicy<
          TensorViewPolicy_RuntimeConst>>::value ? false : not std::is_const<Storage>::value);
    }
  };

  }  // namespace detail

  /// View (aka generalized slice) of a tensor

  /**
      @tparam _T apparent element type, TensorView will present tensor elements as values of this type
      @tparam _Range Range type
      @tparam _Storage Storage type
  */
  template<typename _T,
           class _Range = btas::DEFAULT::range,
           class _Storage = btas::DEFAULT::storage<_T>,
           class _Policy = btas::TensorViewPolicy<>
           >
  class TensorView : private detail::TensorViewMutabilityImpl<_Policy,_Storage> {

      typedef detail::TensorViewMutabilityImpl<_Policy,_Storage> mutability_impl_type;

    public:

      /// type of an element
      typedef _T value_type;

      /// type of an lvalue reference to an element
      typedef value_type& reference;

      /// type of a const lvalue reference to an element
      typedef const value_type& const_reference;

      /// type of Range
      typedef _Range range_type;

      /// type of index
      typedef typename _Range::index_type index_type;

      /// type of underlying data storage
      typedef _Storage storage_type;

      /// type of data storage reference
      typedef std::reference_wrapper<storage_type> storageref_type;

      /// size type
      typedef typename storage_traits<storage_type>::size_type size_type;

      /// element iterator
      typedef TensorViewIterator<range_type, storage_type> iterator;

      /// element iterator
      typedef TensorViewIterator<range_type, const storage_type> const_iterator;

    private:
      struct Enabler {};

      /// use this to disable non-const members
      static constexpr bool constexpr_is_writable() {
        return _Policy::runtimeconst || not std::is_const<storage_type>::value;
      }

    public:

      /// default constructor creates an uninitialized view
      TensorView() :
        range_(),
        storageref_(*((storage_type*)nullptr))
      {}

      /// destructor
      ~TensorView() = default;

      /// construct from \c range and \c storageref ; write access must be passed explicitly if \c _Policy requires
      template<class Range, class Storage, class Policy = _Policy, class = typename std::enable_if<not Policy::runtimeconst>::type>
      TensorView (Range&& range,
                  Storage&& storageref,
                  bool can_write = mutability_impl_type::make_default()) :
                  mutability_impl_type(can_write), range_(std::forward<Range>(range)), storageref_(std::forward<Storage>(storageref))
      {
      }

      /// conversion from const Tensor into TensorConstView
      template <
          class _Tensor, class Storage = _Storage,
          class = typename std::enable_if<
              is_boxtensor<_Tensor>::value && std::is_const<Storage>::value &&
              std::is_same<
                  typename std::decay<typename _Tensor::storage_type>::type,
                  typename std::decay<Storage>::type>::value>::type>
      TensorView(const _Tensor& x)
          : mutability_impl_type(false),
            range_(x.range()),
            storageref_(std::cref(x.storage())) {}

      /// conversion from const Tensor to non-const View only possible if \c
      /// Policy::runtimeconst is \c true
      /// \note this is not explicit to allow simple assignments like \code TensorView view = tensor; \endcode
      template <
          class _Tensor, class Storage = _Storage, class Policy = _Policy,
          class = typename std::enable_if<
              is_boxtensor<_Tensor>::value &&
              not std::is_const<Storage>::value && Policy::runtimeconst &&
              std::is_same<
                  typename std::decay<typename _Tensor::storage_type>::type,
                  typename std::decay<Storage>::type>::value>::type>
      TensorView(const _Tensor& x)
          : mutability_impl_type(false),
            range_(x.range()),
            storageref_(std::ref(const_cast<storage_type&>(x.storage()))) {}

      /// this constructor exists to generate a readable error upon conversion
      /// from const Tensor to compile-time non-const View
      template <class _Tensor, class Storage = _Storage, class Policy = _Policy>
      TensorView(
          const _Tensor& x,
          typename std::enable_if<
              is_boxtensor<_Tensor>::value &&
              not std::is_const<Storage>::value && not Policy::runtimeconst &&
              std::is_same<
                  typename std::decay<typename _Tensor::storage_type>::type,
                  typename std::decay<Storage>::type>::value>::type* = nullptr)
          : TensorView() {
        static_assert(!is_boxtensor<_Tensor>::value,
                      "attempt to create a compile-time-const TensorView from "
                      "a const Tensor");
      }

      /// conversion from non-const Tensor
      template <
          class _Tensor, class Storage = _Storage,
          class = typename std::enable_if<
              is_boxtensor<_Tensor>::value &&
              not std::is_const<_Tensor>::value &&
              std::is_same<
                  typename _Tensor::storage_type,
                  Storage>::value>::type>
      TensorView(_Tensor& x)
          : mutability_impl_type(true),
            range_(x.range()),
            storageref_(std::ref(x.storage())) {}

      /// conversion from non-const TensorView
      template <
          class __T, class __Range, class __Storage, class __Policy,
          class = typename std::enable_if<
              not std::is_const<__Storage>::value &&
              std::is_same<
                                __Storage,
                                _Storage>::value>::type>
      explicit TensorView(TensorView<__T, __Range, __Storage, __Policy>& x)
          : range_(x.range()),
            storageref_(std::ref(x.storage())),
            mutability_impl_type(x) {}

      TensorView (const TensorView& x) = default;
      TensorView& operator= (const TensorView& x) = default;
      TensorView (TensorView&& x) = default;
      TensorView& operator= (TensorView&& x) = default;

      /// number of indices (tensor rank)
      size_type
      rank () const
      {
        return range_.rank();
      }

      /// \return number of elements
      size_type
      size () const
      {
        return range_.area();
      }

      /// \return range object
      const range_type&
      range() const
      {
        return range_;
      }

      /// \param d dimension
      /// \return subrange for dimension \d
      const Range1d<typename index_type::value_type>
      range(size_t d) const
      {
        return range_.range(d);
      }

      /// \return range's extent object
      typename range_type::extent_type
      extent() const
      {
        return range_.extent();
      }

      /// \return extent of range along dimension \c d
      typename range_type::extent_type::value_type
      extent(size_t d) const
      {
        return range_.extent(d);
      }

      /// \return storage object
      const storage_type&
      storage() const
      {
        return storageref_.get();
      }

      /// \return storage object
      storage_type&
      storage()
      {
        assert_writable();
        return storageref_.get();
      }

      /// test whether TensorView is empty
      bool
      empty() const
      {
        return range_.area() == 0;
      }

      /// \return const iterator begin
      const_iterator
      begin() const
      {
        return cbegin();
      }

      /// \return begin iterator
      iterator
      begin()
      {
        assert_writable();
        return iterator(range().begin(), storage());
      }

      /// \return const end iterator
      const_iterator
      end() const
      {
        return cend();
      }

      /// \return const end iterator
      iterator
      end()
      {
        assert_writable();
        return iterator(range().end(), storageref_);
      }

      /// \return const iterator begin, even if this is not itself const
      const_iterator
      cbegin() const
      {
        return const_iterator(range().begin(), storage());
      }

      /// \return const iterator end, even if this is not itself const
      const_iterator
      cend() const
      {
        return const_iterator(range().end(), storage());
      }

      /// Immutable access to an element without range check.

      /// Available when \c value_type == \c storage_type::value_type.
      /// \return const reference to the element indexed by \c index
      template<typename ... Index>
      typename std::enable_if<not is_index<typename std::decay<Index>::type...>::value && std::is_same<value_type,typename storage_type::value_type>::value,
      const_reference>::type
      operator() (Index&& ... index) const
      {
        return storageref_.get()[ range_.ordinal(std::forward<Index>(index)...) ];
      }

      /// Immutable access to an element without range check.

      /// Available when \c value_type == \c storage_type::value_type
      /// \return const reference to the element indexed by \c index
      template <typename Index>
      typename std::enable_if<is_index<Index>::value &&
                              std::is_same<value_type,typename storage_type::value_type>::value,
                              const_reference
                             >::type
      operator() (const Index& index) const
      {
        return storageref_.get()[range_.ordinal(index)];
      }

      /// Mutable access to an element without range check.

      /// Available when \c value_type == \c storage_type::value_type.
      /// \return const reference to the element indexed by \c index
      template<typename ... Index>
      typename std::enable_if<not is_index<typename std::decay<Index>::type...>::value && std::is_same<value_type,typename storage_type::value_type>::value && TensorView::constexpr_is_writable(),
      reference>::type
      operator() (Index&& ... index)
      {
        assert_writable();
        return storageref_.get()[ range_.ordinal(std::forward<Index>(index)...) ];
      }

      /// Mutable access to an element without range check (rank() == general)

      /// Available when \c value_type == \c storag_type::value_type
      /// \return reference to the element indexed by \c index
      template <typename Index>
      typename std::enable_if<is_index<Index>::value &&
                              std::is_same<value_type,typename storage_type::value_type>::value && TensorView::constexpr_is_writable(),
                              reference
                             >::type
      operator() (const Index& index)
      {
        assert_writable();
        return storageref_.get()[range_.ordinal(index)];
      }

      /// Immutable access to an element without range check.

      /// Available when \c value_type != \c storage_type::value_type.
      /// \return value of the element indexed by \c index , converted to \c value_type
      template<typename ... Index>
      typename std::enable_if<not is_index<typename std::decay<Index>::type...>::value && not std::is_same<value_type,typename storage_type::value_type>::value,
      value_type>::type
      operator() (Index&& ... index) const
      {
        return static_cast<value_type>(storageref_.get()[ range_.ordinal(std::forward<Index>(index)...) ]);
      }

      /// Immutable access to an element without range check (rank() == general)

      /// Available when \c value_type != \c storage_type::value_type
      /// \return value of the element indexed by \c index , converted to \c value_type
      template <typename Index>
      typename std::enable_if<is_index<Index>::value &&
                              not std::is_same<value_type,typename storage_type::value_type>::value,
                              value_type
                             >::type
      operator() (const Index& index) const
      {
        return static_cast<value_type>(storageref_.get()[range_.ordinal(index)]);
      }

      /// Immutable access to an element with range check.

      /// Available when \c value_type == \c storage_type::value_type.
      /// \return const reference to the element indexed by \c index
      template<typename ... Index>
      typename std::enable_if<not is_index<typename std::decay<Index>::type...>::value && std::is_same<value_type,typename storage_type::value_type>::value,
      const_reference>::type
      at (Index&& ... index) const
      {
        BTAS_ASSERT( range_.includes(std::forward<Index>(index)...) );
        return this->operator()(std::forward<Index>(index)...);
      }

      /// Immutable access to an element with range check.

      /// Available when \c value_type == \c storage_type::value_type.
      /// \return const reference to the element indexed by \c index
      template <typename Index>
      typename std::enable_if<is_index<Index>::value &&
                              std::is_same<value_type,typename storage_type::value_type>::value,
                              const_reference>::type
      at (const Index& index) const
      {
        BTAS_ASSERT( range_.includes(index) );
        return this->operator()(index);
      }

      /// Mutable access to an element with range check.

      /// Available when \c value_type == \c storage_type::value_type.
      /// \return reference to the element indexed by \c index
      template<typename ... Index>
      typename std::enable_if<not is_index<typename std::decay<Index>::type...>::value && std::is_same<value_type,typename storage_type::value_type>::value && TensorView::constexpr_is_writable(),
      reference>::type
      at (Index&& ... index)
      {
        assert_writable();
        BTAS_ASSERT( range_.includes(std::forward<Index>(index)...) );
        return this->operator()(std::forward<Index>(index)...);
      }

      /// Mutable access to an element with range check.

      /// Available when \c value_type == \c storage_type::value_type.
      /// \return reference to the element indexed by \c index
      template <typename Index>
      typename std::enable_if<is_index<Index>::value &&
                              std::is_same<value_type,typename storage_type::value_type>::value &&
                              TensorView::constexpr_is_writable(),
                              reference>::type
      at (const Index& index)
      {
        assert_writable();
        BTAS_ASSERT( range_.includes(index) );
        return this->operator()(index);
      }

      /// Immutable access to an element with range check.

      /// Available when \c value_type != \c storage_type::value_type.
      /// \return the element value indexed by \c index , converted to \c value_type
      template<typename ... Index>
      typename std::enable_if<not is_index<typename std::decay<Index>::type...>::value && not std::is_same<value_type,typename storage_type::value_type>::value,
      value_type>::type
      at (Index&& ... index) const
      {
        BTAS_ASSERT( range_.includes(std::forward<Index>(index)...) );
        return this->operator()(std::forward<Index>(index)...);
      }

      /// Immutable access to an element with range check.

      /// Available when \c value_type != \c storage_type::value_type.
      /// \return the element value indexed by \c index , converted to \c value_type
      template <typename Index>
      typename std::enable_if<is_index<Index>::value &&
                              not std::is_same<value_type,typename storage_type::value_type>::value,
                              value_type>::type
      at (const Index& index) const
      {
        BTAS_ASSERT( range_.includes(index) );
        return this->operator()(index);
      }

      /// swap this and x
      void
      swap (TensorView& x) noexcept
      {
        using std::swap;
        swap(range_, x.range_);
        swap(storageref_, x.storageref_);
        swap(static_cast<mutability_impl_type&>(*this), static_cast<mutability_impl_type&>(x));
      }

      //  ========== Finished Public Interface and Its Reference Implementations ==========

      //
      //  Here come Non-Standard members (to be discussed)
      //
#if 0
      /// addition assignment
      TensorView&
      operator+= (const TensorView& x)
      {
        assert( std::equal(range_.begin(), range_.end(), x.range_.begin()) );
        std::transform(storageref_.begin(), storageref_.end(), x.storageref_.begin(), storageref_.begin(), std::plus<value_type>());
        return *this;
      }

      /// addition of tensors
      TensorView
      operator+ (const TensorView& x) const
      {
        TensorView y(*this); y += x;
        return y; /* automatically called move semantics */
      }

      /// subtraction assignment
      TensorView&
      operator-= (const TensorView& x)
      {
        assert(
            std::equal(range_.begin(), range_.end(), x.range_.begin()));
        std::transform(storageref_.begin(), storageref_.end(), x.storageref_.begin(), storageref_.begin(), std::minus<value_type>());
        return *this;
      }

      /// subtraction of tensors
      TensorView
      operator- (const TensorView& x) const
      {
        TensorView y(*this); y -= x;
        return y; /* automatically called move semantics */
      }

      /// fill all elements by val
      void
      fill (const value_type& val)
      {
        std::fill(storageref_.begin(), storageref_.end(), val);
      }

      /// generate all elements by gen()
      template<class Generator>
      void
      generate (Generator gen)
      {
          std::generate(storageref_.begin(), storageref_.end(), gen);
      }
#endif

      bool writable() const {
        return static_cast<bool>(static_cast<const mutability_impl_type&>(*this));
      }

    private:

      range_type range_;///< range object
      storageref_type storageref_;///< dataref
//      typedef typename std::conditional<_Policy::runtimeconst,
//                                        bool,
//                                        btas::detail::bool_type<not std::is_const<storage_type>::value>
//                                       >::type writable_type;
//      writable_type can_write_;

      /// use this in non-const members to assert writability if Policy calls for runtime const check
      void assert_writable() const {
        if (_Policy::runtimeconst)
          BTAS_ASSERT(writable());
      }

      /// construct from \c range and \c storage; pass \c can_write explicitly if needed
      explicit TensorView (range_type&& range, storage_type& storage,
                           bool can_write = mutability_impl_type::make_default()) :
          mutability_impl_type(can_write), range_(std::move(range)), storageref_(std::ref(storage)) {
      }

      template <typename T,
                typename Range,
                typename Storage,
                typename Policy>
      friend TensorView<T,
                 Range,
                 typename std::conditional<std::is_same<T,typename Storage::value_type>::value,
                                           Storage,
                                           typename std::add_const<Storage>::type
                                          >::type,
                 Policy>
      __make_view(Range&& range, Storage& storage,
                  Policy,
                  bool can_write);
      template <typename T,
                typename Range,
                typename Storage,
                typename Policy>
      friend TensorView<T, Range, const Storage, Policy> __make_cview(Range&& range, const Storage& storage, Policy);

      template <class __T,
                class __Range,
                class __Storage,
                class __Policy>
      friend class TensorView;
  }; // end of TensorView

  // N.B. The equality and inequality operators are implemented by the generic ops in tensor.h

  /// TensorConstView is a read-only variant of TensorView
  template <typename _T,
            class _Range   = btas::DEFAULT::range,
            class _Storage = btas::DEFAULT::storage<_T>,
            class _Policy  = btas::TensorViewPolicy<>
           >
  using TensorConstView = TensorView<_T, _Range, const _Storage, _Policy>;

  /// TensorRWView is a variant of TensorView with runtime write access check
  template <typename _T,
            class _Range   = btas::DEFAULT::range,
            class _Storage = btas::DEFAULT::storage<_T>,
            class _Policy  = btas::TensorViewPolicy<TensorViewPolicy_RuntimeConst>
           >
  using TensorRWView = TensorView<_T, _Range, typename std::remove_const<_Storage>::type, _Policy>;


  /// Helper function (friendly to TensorView) that constructs a view with an explicitly-specified element type of the view. Useful if need to
  /// view a tensor of floats as a tensor of complex floats.
  /// \tparam T the element type of the resulting view
  /// \tparam Range the range type
  /// \tparam Storage the storage type
  /// \tparam Policy the TensorViewPolicy type
  /// \param range the range object defining the view
  /// \param storage the storage object that will be viewed into
  /// \return TensorView into \c storage using \c range and policy \c Policy
  /// \attention use __make_cview if you must force a const view; this will provide const view, however, if \c storage is a const reference.
  template <typename T,
            typename Range,
            typename Storage,
            typename Policy>
  TensorView<T,
             Range,
             typename std::conditional<std::is_same<T,typename Storage::value_type>::value,
                                       Storage,
                                       typename std::add_const<Storage>::type
                                      >::type,
             Policy>
  __make_view(Range&& range, Storage& storage,
              Policy = Policy(),
              bool can_write = not Policy::runtimeconst
                               ? (not std::is_const<Storage>::value && std::is_same<T,typename Storage::value_type>::value)
                               : false)
  {
    typedef  TensorView<T,
        Range,
        typename std::conditional<std::is_same<T,typename Storage::value_type>::value,
                                  Storage,
                                  typename std::add_const<Storage>::type
                                 >::type,
        Policy> result_type;
    return result_type(std::move(range), storage, can_write);
  }

  /// Helper function (friendly to TensorView) that constructs a view, with an explicitly-specified element type of the view. Useful if need to
  /// view a tensor of floats as a tensor of complex floats. \sa TensorConstView
  /// \tparam T the element type of the resulting view
  /// \tparam Range the range type
  /// \tparam Storage the storage type
  /// \param range the range object defining the view
  /// \param storage the storage object that will be viewed into
  /// \return TensorView into \c storage using \c range and policy \c Policy
  template <typename T,
            typename Range,
            typename Storage,
            typename Policy>
  TensorView<T, Range, const Storage, Policy>
  __make_cview(Range&& range, const Storage& storage, Policy = Policy())
  {
    return TensorView<T, Range, const Storage, Policy>(std::move(range), storage, false);
  }

  /// Helper function that constructs TensorView.
  /// \tparam Range the range type
  /// \tparam Storage the storage type
  /// \tparam Policy the TensorViewPolicy type; if the Policy requires additional runtime parameters use __make_view instead
  /// \param range the range object defining the view
  /// \param storage the storage object that will be viewed into
  /// \return TensorView into \c storage using \c range, with policy \c Policy
  /// \attention use make_cview if you must force a const view; this will provide const view, however, if \c storage is a const reference.
  template <typename Range,
            typename Storage,
            typename Policy = TensorViewPolicy<TensorViewPolicy_CompiletimeConst>,
            class = typename std::enable_if<not std::is_reference<Range>::value>::type>
  TensorView<typename Storage::value_type, Range, Storage, Policy>
  make_view(const Range& range, Storage& storage, Policy = Policy())
  {
    return make_view<typename Storage::value_type, Range, Storage, Policy>(range, storage);
  }

  /// Helper function that constructs TensorView.
  /// \tparam Range the range type
  /// \tparam Storage the storage type
  /// \tparam Policy the TensorViewPolicy type
  /// \param range the range object defining the view
  /// \param storage the storage object that will be viewed into
  /// \return TensorView into \c storage using \c range, with policy \c Policy
  /// \attention use make_cview if you must force a const view; this will provide const view, however, if \c storage is a const reference.
  template <typename Range,
            typename Storage,
            typename Policy = TensorViewPolicy<TensorViewPolicy_CompiletimeConst>,
            class = typename std::enable_if<not std::is_reference<Range>::value>::type>
  TensorView<typename Storage::value_type, Range, Storage, Policy>
  make_view(Range&& range, Storage& storage, Policy = Policy())
  {
    return make_view<typename Storage::value_type, Range, Storage, Policy>(range, storage);
  }


  /// Helper function that constructs TensorView, with an explicitly-specified element type of the view. Useful if need to
  /// view a tensor of floats as a tensor of complex floats.
  /// \tparam T the element type of the resulting view
  /// \tparam Range the range type
  /// \tparam Storage the storage type
  /// \tparam Policy the TensorViewPolicy type
  /// \param range the range object defining the view
  /// \param storage the storage object that will be viewed into
  /// \return TensorView into \c storage using \c range, with policy \c Policy
  /// \attention use make_cview if you must force a const view; this will provide const view, however, if \c storage is a const reference.
  template <typename T,
            typename Range,
            typename Storage,
            typename Policy = TensorViewPolicy<TensorViewPolicy_CompiletimeConst>,
            class = typename std::enable_if<not std::is_reference<Range>::value>::type>
  auto
  make_view(const Range& range, Storage& storage, Policy = Policy()) -> decltype(__make_view<T, Range, Storage, Policy>(Range(range), storage))
  {
    return __make_view<T, Range, Storage, Policy>(Range(range), storage);
  }

  /// Helper function that constructs TensorView, with an explicitly-specified element type of the view. Useful if need to
  /// view a tensor of floats as a tensor of complex floats.
  /// \tparam T the element type of the resulting view
  /// \tparam Range the range type
  /// \tparam Storage the storage type
  /// \tparam Policy the TensorViewPolicy type
  /// \param range the range object defining the view
  /// \param storage the storage object that will be viewed into
  /// \return TensorView into \c storage using \c range, with policy \c Policy
  /// \attention use make_cview if you must force a const view; this will provide const view, however, if \c storage is a const reference.
  template <typename T,
            typename Range,
            typename Storage,
            typename Policy = TensorViewPolicy<TensorViewPolicy_CompiletimeConst>,
            class = typename std::enable_if<not std::is_reference<Range>::value>::type>
  auto
  make_view(Range&& range, Storage& storage, Policy = Policy()) -> decltype(__make_view<T, Range, Storage, Policy>(range, storage))
  {
    return __make_view<T, Range, Storage, Policy>(range, storage);
  }

  /// Helper function that constructs a full TensorView of a Tensor.
  /// \tparam Tensor the tensor type
  /// \param tensor the Tensor object
  /// \return TensorView, a full view of the \c tensor
  /// \attention use make_cview if you must force a const view; this will provide const view, however, if \c tensor is a const reference.
  /// \note Provided for completeness.
  template <typename Tensor,
            typename Policy = TensorViewPolicy<TensorViewPolicy_CompiletimeConst>,
            class = typename std::enable_if<is_boxtensor<Tensor>::value>::type>
  TensorView<typename Tensor::value_type,
             typename Tensor::range_type,
             typename Tensor::storage_type,
             Policy>
  make_view(Tensor& tensor, Policy = Policy())
  {
    return TensorView<typename Tensor::value_type,
                      typename Tensor::range_type,
                      typename Tensor::storage_type,
                      Policy>(tensor);
  }

  /// Helper function that constructs a full TensorView of a Tensor,
  /// with an explicitly-specified element type of the view. Useful if need to
  /// view a tensor of floats as a tensor of complex floats.
  /// \tparam T the element type of the resulting view
  /// \tparam Tensor the tensor type
  /// \param tensor the Tensor object
  /// \return TensorView, a full view of the \c tensor
  /// \attention use make_cview if you must force a const view; this will provide const view, however, if \c tensor is a const reference.
  /// \note Provided for completeness.
  template <typename T, typename Tensor,
            typename Policy = TensorViewPolicy<TensorViewPolicy_CompiletimeConst>,
            class = typename std::enable_if<is_boxtensor<Tensor>::value>::type>
  TensorView<T,
             typename Tensor::range_type,
             typename std::conditional<std::is_same<T,typename Tensor::storage_type::value_type>::value,
                                       typename Tensor::storage_type,
                                       typename std::add_const<typename Tensor::storage_type>::type
                                      >::type,
             Policy>
  make_view(Tensor& tensor, Policy = Policy())
  {
      typedef   TensorView<T,
          typename Tensor::range_type,
          typename std::conditional<std::is_same<T,typename Tensor::storage_type::value_type>::value,
                                    typename Tensor::storage_type,
                                    typename std::add_const<typename Tensor::storage_type>::type
                                   >::type,
          Policy> result_type;
    return result_type(tensor);
  }

  /// Helper function that constructs a constant TensorView. \sa TensorConstView
  /// \tparam Range the range type
  /// \tparam Storage the storage type
  /// \param range the range object defining the view
  /// \param storage the storage object that will be viewed into
  /// \return TensorView into \c storage using \c range
  template <typename Range,
            typename Storage,
            typename Policy = btas::TensorViewPolicy<TensorViewPolicy_CompiletimeConst>,
            class = typename std::enable_if<not std::is_reference<Range>::value>::type>
  TensorView<typename Storage::value_type, Range, const Storage, Policy>
  make_cview(const Range& range, const Storage& storage, Policy = Policy())
  {
    return make_cview<typename Storage::value_type, Range, Storage, Policy>(range, storage);
  }

  /// Helper function that constructs a constant TensorView, with an explicitly-specified element type of the view. Useful if need to
  /// view a tensor of floats as a tensor of complex floats. \sa TensorConstView
  /// \tparam T the element type of the resulting view
  /// \tparam Range the range type
  /// \tparam Storage the storage type
  /// \param range the range object defining the view
  /// \param storage the storage object that will be viewed into
  /// \return TensorView into \c storage using \c range
  template <typename T,
            typename Range,
            typename Storage,
            typename Policy = TensorViewPolicy<TensorViewPolicy_CompiletimeConst>,
            class = typename std::enable_if<not std::is_reference<Range>::value>::type>
  TensorView<T, Range, const Storage, Policy>
  make_cview(const Range& range, const Storage& storage, Policy = Policy())
  {
    return __make_cview<T, Range, const Storage, Policy>(Range(range), storage);
  }

  /// Helper function that constructs a full constant TensorView of a Tensor.
  /// \tparam Tensor the tensor type
  /// \param tensor the Tensor object
  /// \return TensorView, a full view of the \c tensor
  /// \note Provided for completeness.
  template <typename Tensor,
            typename Policy = TensorViewPolicy<TensorViewPolicy_CompiletimeConst>,
            class = typename std::enable_if<is_boxtensor<Tensor>::value>::type>
  TensorView<typename Tensor::value_type,
             typename Tensor::range_type,
             const typename Tensor::storage_type,
             Policy>
  make_cview(const Tensor& tensor)
  {
    return TensorView<typename Tensor::value_type,
                      typename Tensor::range_type,
                      const typename Tensor::storage_type,
                      Policy>(tensor);
  }

  /// Helper function that constructs a full constant TensorView of a Tensor,
  /// with an explicitly-specified element type of the view. Useful if need to
  /// view a tensor of floats as a tensor of complex floats.
  /// \tparam T the element type of the resulting view
  /// \tparam Tensor the tensor type
  /// \param tensor the Tensor object
  /// \return TensorView, a full view of the \c tensor
  /// \note Provided for completeness.
  template <typename T, typename Tensor,
            typename Policy = TensorViewPolicy<TensorViewPolicy_CompiletimeConst>,
            class = typename std::enable_if<is_boxtensor<Tensor>::value>::type>
  TensorView<T,
             typename Tensor::range_type,
             const typename Tensor::storage_type,
             Policy>
  make_cview(const Tensor& tensor)
  {
    return TensorView<T,
                      typename Tensor::range_type,
                      const typename Tensor::storage_type,
                      Policy>(tensor);
  }

  /// Helper function that constructs writable TensorView.
  /// \tparam Range the range type
  /// \tparam Storage the storage type
  /// \param range the range object defining the view
  /// \param storage the storage object that will be viewed into
  /// \return TensorView into \c storage using \c range
  /// \attention use make_cview if you must force a const view; this will provide const view, however, if \c storage is a const reference.
  template <typename Range,
            typename Storage,
            class = typename std::enable_if<not std::is_reference<Range>::value>::type>
  TensorRWView<typename Storage::value_type, Range, Storage>
  make_rwview(const Range& range,
              Storage& storage,
              bool can_write = not std::is_const<Storage>::value)
  {
    // enforce mutability
    can_write = can_write && (not std::is_const<Storage>::value);
    return make_rwview(Range(range), storage, can_write);
  }

  /// Helper function that constructs writable TensorView.
  /// \tparam Range the range type
  /// \tparam Storage the storage type
  /// \param range the range object defining the view
  /// \param storage the storage object that will be viewed into
  /// \return TensorView into \c storage using \c range
  /// \attention use make_cview if you must force a const view; this will provide const view, however, if \c storage is a const reference.
  template <typename Range,
            typename Storage,
            class = typename std::enable_if<not std::is_reference<Range>::value>::type>
  TensorRWView<typename Storage::value_type, Range, Storage>
  make_rwview(Range&& range,
              Storage& storage,
              bool can_write = not std::is_const<Storage>::value)
  {
    // enforce mutability
    can_write = can_write && (not std::is_const<Storage>::value);
    return make_rwview<typename Storage::value_type, Range, Storage>(std::move(range), storage, can_write);
  }

  /// Helper function that constructs writable TensorView, with an explicitly-specified element type of the view. Useful if need to
  /// view a tensor of floats as a tensor of complex floats.
  /// \tparam T the element type of the resulting view
  /// \tparam Range the range type
  /// \tparam Storage the storage type
  /// \param range the range object defining the view
  /// \param storage the storage object that will be viewed into
  /// \return TensorView into \c storage using \c range
  /// \attention use make_cview if you must force a const view; this will provide const view, however, if \c storage is a const reference.
  template <typename T,
            typename Range,
            typename Storage,
            class = typename std::enable_if<not std::is_reference<Range>::value>::type>
  TensorRWView<T, Range, Storage>
  make_rwview(const Range& range, Storage& storage,
              bool can_write = not std::is_const<Storage>::value &&
                               std::is_same<T,typename Storage::value_type>::value)
  {
    // enforce mutability
    can_write = can_write && (not std::is_const<Storage>::value &&
                              std::is_same<T,typename Storage::value_type>::value);
    return make_rwview(Range(range),
                       storage,
                       can_write);
  }

  /// Helper function that constructs writable TensorView, with an explicitly-specified element type of the view. Useful if need to
  /// view a tensor of floats as a tensor of complex floats.
  /// \tparam T the element type of the resulting view
  /// \tparam Range the range type
  /// \tparam Storage the storage type
  /// \param range the range object defining the view
  /// \param storage the storage object that will be viewed into
  /// \return TensorView into \c storage using \c range
  /// \attention use make_cview if you must force a const view; this will provide const view, however, if \c storage is a const reference.
  template <typename T,
            typename Range,
            typename Storage,
            class = typename std::enable_if<not std::is_reference<Range>::value>::type>
  TensorRWView<T, Range, Storage>
  make_rwview(Range&& range, Storage& storage,
              bool can_write = not std::is_const<Storage>::value &&
                               std::is_same<T,typename Storage::value_type>::value)
  {
    // enforce mutability
    can_write = can_write && (not std::is_const<Storage>::value &&
                              std::is_same<T,typename Storage::value_type>::value);
    return __make_view<T,
                       Range,
                       typename std::remove_const<Storage>::type,
                       TensorViewPolicy<TensorViewPolicy_RuntimeConst> >(std::move(range),
                          const_cast<typename std::remove_const<Storage>::type&>(storage),
                          TensorViewPolicy<TensorViewPolicy_RuntimeConst>(),
                          can_write);
  }

  /// Helper function that constructs a full writable TensorView of a Tensor.
  /// \tparam Tensor the tensor type
  /// \param tensor the Tensor object
  /// \return TensorView, a full view of the \c tensor
  /// \note Provided for completeness.
  template <typename Tensor, class = typename std::enable_if<is_boxtensor<Tensor>::value>::type>
  TensorRWView<typename Tensor::value_type,
               typename Tensor::range_type,
               typename Tensor::storage_type>
  make_rwview(Tensor& tensor,
              bool can_write = not std::is_const<Tensor>::value &&
                               not std::is_const<typename Tensor::storage_type>::value)
  {
      // enforce mutability
      can_write = can_write && (not std::is_const<Tensor>::value && not std::is_const<typename Tensor::storage_type>::value);
      return make_rwview(tensor.range(), tensor.storage(), can_write);
  }

  /// Helper function that constructs a full writable TensorView of a Tensor,
  /// with an explicitly-specified element type of the view. Useful if need to
  /// view a tensor of floats as a tensor of complex floats.
  /// \tparam T the element type of the resulting view
  /// \tparam Tensor the tensor type
  /// \param tensor the Tensor object
  /// \return TensorView, a full view of the \c tensor
  /// \note Provided for completeness.
  template <typename T, typename Tensor, class = typename std::enable_if<is_boxtensor<Tensor>::value>::type>
  TensorRWView<T,
               typename Tensor::range_type,
               typename Tensor::storage_type>
  make_rwview(Tensor& tensor,
              bool can_write = not std::is_const<Tensor>::value &&
                               not std::is_const<typename Tensor::storage_type>::value &&
                               std::is_same<T,typename Tensor::storage_type::value_type>::value)
  {
      // enforce mutability
      can_write = can_write &&
                  (not std::is_const<Tensor>::value &&
                   not std::is_const<typename Tensor::storage_type>::value &&
                   std::is_same<T,typename Tensor::storage_type::value_type>::value);
      return make_rwview(tensor.range(), tensor.storage(), can_write);
  }

  template <typename _T, typename _Range, typename _Storage>
  auto cbegin(const btas::TensorView<_T, _Range, _Storage>& x) -> decltype(x.cbegin()) {
    return x.cbegin();
  }
  template <typename _T, typename _Range, typename _Storage>
  auto cend(const btas::TensorView<_T, _Range, _Storage>& x) -> decltype(x.cbegin()) {
    return x.cend();
  }

  /// maps TensorView -> Range
  template <typename _T, typename _Range, typename _Storage>
  auto
  range (const btas::TensorView<_T, _Range, _Storage>& t) -> decltype(t.range()) {
    return t.range();
  }

  /// maps TensorView -> Range extent
  template <typename _T, typename _Range, typename _Storage>
  auto
  extent (const btas::TensorView<_T, _Range, _Storage>& t) -> decltype(t.range().extent()) {
    return t.range().extent();
  }

  /// TensorView stream output operator

  /// prints TensorView in row-major form. To be implemented elsewhere using slices.
  /// \param os The output stream that will be used to print \c t
  /// \param t The TensorView to be printed
  /// \return A reference to the output stream
  template <typename _T, typename _Range, typename _Storage>
  std::ostream& operator<<(std::ostream& os, const btas::TensorView<_T, _Range, _Storage>& t) {
    os << "TensorView:\n  Range: " << t.range() << std::endl;
    return os;
  }

  /// TensorMap views a sequence of values as a Tensor
  template <typename _T,
            class _Range = btas::DEFAULT::range>
  using TensorMap = TensorView<_T, _Range, btas::infinite_sequence_adaptor<_T*>>;
  /// TensorConstMap const-views a sequence of values as a Tensor
  template <typename _T,
            class _Range = btas::DEFAULT::range>
  using TensorConstMap = TensorView<const _T, _Range, const btas::infinite_sequence_adaptor<const _T*>>;

  /// Helper function that constructs TensorMap.
  /// \tparam T the element type returned by the view
  /// \tparam Range the range type
  /// \param range the range object defining the view
  /// \return TensorView into \c storage using \c range
  /// \attention use make_cmap if you must force a const view; this will provide const view, however, if \c storage is a const reference.
  template <typename T,
            typename Range>
  TensorMap<T, typename std::decay<Range>::type>
  make_map(T* data, Range&& range)
  {
    return TensorMap<T, typename std::decay<Range>::type>(std::forward<Range>(range),
                               std::ref(btas::infinite_sequence_adaptor<T*>(data)));
  }

  /// Helper function that constructs TensorConstMap.
  /// \tparam T the element type returned by the view
  /// \tparam Range the range type
  /// \param range the range object defining the view
  /// \return TensorView into \c storage using \c range
  /// \attention use make_cmap if you must force a const view; this will provide const view, however, if \c storage is a const reference.
  template <typename T,
            typename Range>
  TensorConstMap<T, typename std::decay<Range>::type>
  make_map(const T* data, Range&& range)
  {
    return TensorConstMap<T, typename std::decay<Range>::type>(std::forward<Range>(range),
                                    std::cref(btas::infinite_sequence_adaptor<const T*>(data)));
  }

  /// Helper function that constructs TensorConstMap.
  /// \tparam Range the range type
  /// \param range the range object defining the view
  /// \return TensorView into \c storage using \c range
  /// \attention use make_cmap if you must force a const view; this will provide const view, however, if \c storage is a const reference.
  template <typename T,
            typename Range>
  TensorConstMap<typename std::remove_const<T>::type, typename std::decay<Range>::type>
  make_cmap(T* data, Range&& range)
  {
    typedef typename std::remove_const<T>::type value_type;
    typedef TensorConstMap<value_type, typename std::decay<Range>::type> result_type;
    return result_type(std::forward<Range>(range),
                       std::cref(btas::infinite_sequence_adaptor<const T*>(const_cast<const T*>(data))));
  }

} // namespace btas

// serialization of TensorView is disabled
#if 0
namespace boost {
  namespace serialization {

    /// boost serialization
    template<class Archive,
             typename _T,
             class _Range,
             class _Storage,
             class _Policy>
    void serialize(Archive& ar,
                   btas::TensorView<_T,_Range,_Storage,_Policy>& tv,
                   const unsigned int version) {
      boost::serialization::split_free(ar, tv, version);
    }
    template<class Archive,
             typename _T,
             class _Range,
             class _Storage,
             class _Policy>
    void save(Archive& ar,
              const btas::TensorView<_T,_Range,_Storage,_Policy>& tv,
              const unsigned int version) {
      const auto& range = tv.range();
      const auto* storage_ptr = &tv.storage();
      bool writable = tv.writable();
      ar << BOOST_SERIALIZATION_NVP(range) << BOOST_SERIALIZATION_NVP(storage_ptr) << BOOST_SERIALIZATION_NVP(writable);
    }
    template<class Archive,
             typename _T,
             class _Range,
             class _Storage,
             class _Policy>
    void load(Archive& ar,
              btas::TensorView<_T,_Range,_Storage,_Policy>& tv,
              const unsigned int version) {
      _Range range;
      _Storage* storage_ptr;
      bool writable;
      ar >> BOOST_SERIALIZATION_NVP(range) >> BOOST_SERIALIZATION_NVP(storage_ptr) >> BOOST_SERIALIZATION_NVP(writable);
      std::reference_wrapper<_Storage> storage_ref(*storage_ptr);
      tv = btas::TensorView<_T,_Range,_Storage,_Policy>(std::move(range), std::move(storage_ref), writable);
    }

  } // namespace serialization
} // namespace boost
#endif // serialization of TensorView is disabled

#endif /* TENSORVIEW_H_ */
