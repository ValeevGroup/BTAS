#ifndef __BTAS_TARRAY_H
#define __BTAS_TARRAY_H 1

#include <array>
#include <vector>
#include <algorithm>
#include <type_traits>

#include <btas/types.h>
#include <btas/tensor_traits.h>

#include <btas/util/stride.h>
#include <btas/util/dot.h>

namespace btas {

/// tensor class which has fixed-size rank as a template parameter
template<typename _T,
         unsigned long _N,
         CBLAS_ORDER _Order = CblasRowMajor,
         class _Container = DEFAULT::storage<_T>>
class TArray {

public:

   //
   //  Type names ==================================================================================
   //

   /// value type
   typedef _T value_type;

   /// container type
   typedef _Container container;

   /// size_type
   typedef typename container::size_type size_type;

   /// iterator
   typedef typename container::iterator iterator;

   /// const iterator
   typedef typename container::const_iterator const_iterator;

   /// shape type
   typedef std::array<size_type, _N> shape_type;

   //
   //  rank() is of const expression ===============================================================
   //

   /// \return rank of array
   static constexpr const size_type rank() { return _N; }

   //
   //  Constructors ================================================================================
   //

   /// default constructor
   TArray()
   {
      shape_.fill(0);
      stride_.fill(0);
   }

   /// destructor
   ~TArray()
   { }

   /// construct from variadic arguments
   template<typename... Args>
   explicit
   TArray(size_type n, Args... rest)
   {
      resize(n, rest...);
   }

   /// construct from shape
   explicit
   TArray(const shape_type& shape)
   : shape_ (shape)
   {
      resize(shape);
   }

   /// construct from shape with init. value
   TArray(const shape_type& shape, const value_type& val)
   : shape_ (shape)
   {
      resize(shape, val);
   }

   /// copy constructor
   template<class _Tensor, class = typename std::enable_if<is_tensor<_Tensor>::value>::type>
   explicit
   TArray(const _Tensor& x)
   {
      // x is assumed to be dynamic (i.e. variable-rank) tensor
      assert(x.rank() == _N);
      // set shape_ and stride_
      std::copy(x.shape ().begin(), x.shape ().end(), shape_.begin());
      std::copy(x.stride().begin(), x.stride().end(), stride_.begin());
      // resize data_ and take deep copy from x
      data_.resize(x.size());
      std::copy(x.begin(), x.end(), data_.begin());
   }

   /// copy constructor (to avoid implicit deletion of copy constructor)
   explicit
   TArray(const TArray& x)
   : shape_ (x.shape_), stride_ (x.stride_)
   {
      data_.resize(x.data_.size());
      std::copy(x.data_.begin(), x.data_.end(), data_.begin());
   }

   /// move constructor
   TArray(const TArray&& x)
   : shape_ (x.shape_), stride_ (x.stride_), data_ (x.data_)
   { }

   //
   //  Assignment operators ========================================================================
   //

   /// copy assignment
   template<class _Tensor, class = typename std::enable_if<is_tensor<_Tensor>::value>::type>
   TArray& operator= (const _Tensor& x)
   {
      // x is assumed to be dynamic (i.e. variable-rank) tensor
      assert(x.rank() == _N);
      // set shape_ and stride_
      std::copy(x.shape ().begin(), x.shape ().end(), shape_.begin());
      std::copy(x.stride().begin(), x.stride().end(), stride_.begin());
      // resize data_ and take deep copy from x
      data_.resize(x.size());
      std::copy(x.begin(), x.end(), data_.begin());
      return *this;
   }

   /// copy assignment (to avoid implicit deletion of copy assignment)
   TArray& operator= (const TArray& x)
   {
      shape_  = x.shape_;
      stride_ = x.stride_;
      std::copy(x.data_.begin(), x.data_.end(), data_.begin());
      return *this;
   }

   /// move assignment
   TArray& operator= (TArray&& x)
   {
      shape_.swap(x.shape_);
      stride_.swap(x.stride_);
      data_.swap(x.data_);
      return *this;
   }

   //
   //  Element access ==============================================================================
   //

   /// element access without range check from variadic arguments
   template<typename... Args>
   const value_type& operator() (size_type n, Args... rest) const
   {
      return data_[__args_to_address<0>(n, rest...)];
   }

   /// element access without range check from variadic arguments
   template<typename... Args>
   value_type& operator() (size_type n, Args... rest)
   {
      return data_[__args_to_address<0>(n, rest...)];
   }

   /// element access without range check from index object
   const value_type& operator() (const shape_type& index) const
   {
      return data_[__index_to_address(index)];
   }

   /// element access without range check from index object
   value_type& operator() (const shape_type& index)
   {
      return data_[__index_to_address(index)];
   }

   /// element access with range check from variadic arguments
   template<typename... Args>
   const value_type& at(size_type n, Args... rest) const
   {
      return data_.at(__args_to_address<0>(n, rest...));
   }

   /// element access with range check from variadic arguments
   template<typename... Args>
   value_type& at(size_type n, Args... rest)
   {
      return data_.at(__args_to_address<0>(n, rest...));
   }

   /// element access with range check from index object
   const value_type& at(const shape_type& index) const
   {
      return data_.at(__index_to_address(index));
   }

   /// element access with range check from index object
   value_type& at(const shape_type& index)
   {
      return data_.at(__index_to_address(index));
   }

   //
   //  Iterators ===================================================================================
   //

   /// \return iterator to the first
   iterator begin() { return data_.begin(); }

   /// \return const_iterator to the first
   const_iterator begin() const { return data_.begin(); }

   /// \return const iterator to the first even if this is not itself const
   const_iterator cbegin() const { return data_.begin(); }

   /// \return iterator to the end
   iterator end() { return data_.end(); }

   /// \return const_iterator to the end
   const_iterator end() const { return data_.end(); }

   /// \return const iterator to the end even if this is not itself const
   const_iterator cend() const { return data_.end(); }

   //
   //  Size and Shape ==============================================================================
   //

   /// test whether storage is empty
   bool empty() const { return data_.empty(); }

   /// \return size of total elements
   size_type size() const { return data_.size(); }

   /// \return shape object
   const shape_type& shape() const { return shape_; }

   /// \return n-th shape
   const size_type& shape(size_type n) const { return shape_.at(n); }

   /// \return stride object
   const shape_type& stride() const { return stride_; }

   /// \return n-th stride
   const size_type& stride(size_type n) const { return stride_.at(n); }

   //
   //  Resize ======================================================================================
   //

   /// resize from variadic arguments
   template<typename... Args>
   void resize(size_type n, Args... rest)
   {
      __resize_by_args<0>(n, rest...);
   }

   /// resize from shape
   void resize(const shape_type& shape)
   {
      shape_ = shape;
//    __normal_stride<_Order>::set(shape_, stride_);
//    data_.resize(shape_[0]*stride_[0]);
      data_.resize(__normal_stride<_Order>::set(shape_, stride_));
   }

   /// resize from shape with init. value
   void resize(const shape_type& shape, const value_type& val)
   {
      resize(shape);
      fill(val);
   }

   //
   //  Others ======================================================================================
   //

   /// swap object
   void swap(TArray& x)
   {
      shape_.swap(x.shape_);
      stride_.swap(x.stride_);
      data_.swap(x.data_);
   }

   /// clear data storage
   void clear()
   {
      shape_.fill(0);
      stride_.fill(0);
      data_.clear();
   }

   //
   //  Non-standard member functions ===============================================================
   //

   /// return major order directive
   static constexpr CBLAS_ORDER order() { return _Order; }

   /// fill elements with const value
   void fill(const value_type& val)
   {
      std::fill(data_.begin(), data_.end(), val);
   }

   /// fill elements with gen()
   template<class _Generator>
   void generate(_Generator gen)
   {
      std::generate(data_.begin(), data_.end(), gen);
   }

   /// access to bare pointer
   value_type*
   data()
   {
      return data_.data();
   }

   /// access to bare pointer
   const value_type*
   data() const
   {
      return data_.data();
   }

private:

   //
   //  Supportive functions ========================================================================
   //

   /// resize by variadic arguments
   /// if \tparam i exceeds the rank, this gives an error at compile-time
   template<int _i, typename... Args, class = typename std::enable_if<(_i < _N)>::type>
   void __resize_by_args(size_type n, Args... rest)
   {
      shape_[_i] = n;
      __resize_by_args<_i+1>(rest...);
   }

   /// specialized for the last argument
   template<int _i, class = typename std::enable_if<(_i == _N)>::type>
   void __resize_by_args()
   {
//    __normal_stride<_Order>::set(shape_, stride_);
//    data_.resize(shape_[0]*stride_[0]);
      data_.resize(__normal_stride<_Order>::set(shape_, stride_));
   }

   /// specialized for the last argument (case having init. value)
   template<int _i, class = typename std::enable_if<(_i == _N)>::type>
   void __resize_by_args(const value_type& val)
   {
      __resize_by_args<_i>();
      fill(val);
   }

   /// calculate address from index arguments
   /// if \tparam i exceeds the rank, this gives an error at compile-time
   template<int _i, typename... Args, class = typename std::enable_if<(_i < _N-1)>::type>
   size_type __args_to_address(size_type n, Args... rest)
   {
      return n*stride_[_i]+__args_to_address<_i+1>(rest...);
   }

   /// specialized for the last argument
   template<int _i, class = typename std::enable_if<(_i == _N-1)>::type>
   size_type __args_to_address(size_type n)
   {
      return n*stride_[_i];
   }

   /// calculate address from index shape
   size_type __index_to_address(const shape_type& index)
   {
      return dot(stride_, index);
   }

   //
   //  Member variables ============================================================================
   //

   /// shape
   shape_type shape_;

   /// stride
   shape_type stride_;

   /// data
   container data_;

};

} // namespace btas

#endif // __BTAS_TARRAY_H
