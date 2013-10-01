#ifndef __BTAS_TENSOR_H
#define __BTAS_TENSOR_H

#include <vector>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

#include <blas_type.h>

namespace btas
{

//! Utility: dot b/w two std::vector
template <typename T>
T dot(const std::vector<T>& x, const std::vector<T>& y)
    {
        assert(x.size() == y.size());
        auto xt = x.begin();
        auto yt = y.begin();
        T xy = static_cast<T>(0);
        while(xt != x.end())
        {
            xy += (*xt) * (*yt);
            ++xt;
            ++yt;
        }
        return xy;
    }

//! Use std::vector as range object
typedef std::vector<unsigned int>
TensorRange;

// forward decl. SlicedTensor
template <typename T>
class SlicedTensor;

//! Reference implementation of dense tensor class
template <typename T>
class Tensor
    {
    public:

    //! element type (templated)
    typedef T 
    value_type;

    //! type of structure storing index ranges (sizes)
    typedef TensorRange
    range_type;

    //! type of data to be stored as 1D array
    typedef std::vector<T>
    data_type;

    //
    // Constructors
    //

    //! default constructor
    Tensor()
    {}

    //! constructor with index range, for ndim() == 1
    explicit 
    Tensor(int n01)
    {
        resize(n01);
    }

    //! constructor with index ranges, for ndim() == 2
    Tensor(int n01, int n02)
    {
        resize(n01, n02);
    }

    //! constructor with index ranges, for ndim() == 3
    Tensor(int n01, int n02, int n03)
    {
        resize(n01, n02, n03);
    }

    //! constructor with index ranges, for ndim() == 4
    Tensor(int n01, int n02, int n03, int n04)
    {
        resize(n01, n02, n03, n04);
    }

    //! constructor with index ranges, for ndim() == 5
    Tensor(int n01, int n02, int n03, int n04,
           int n05)
    {
        resize(n01, n02, n03, n04,
               n05);
    }

    //! constructor with index ranges, for ndim() == 6
    Tensor(int n01, int n02, int n03, int n04,
           int n05, int n06)
    {
        resize(n01, n02, n03, n04,
               n05, n06);
    }

    //! constructor with index ranges, for ndim() == 7
    Tensor(int n01, int n02, int n03, int n04,
           int n05, int n06, int n07)
    {
        resize(n01, n02, n03, n04,
               n05, n06, n07);
    }

    //! constructor with index ranges, for ndim() == 8
    Tensor(int n01, int n02, int n03, int n04,
           int n05, int n06, int n07, int n08)
    {
        resize(n01, n02, n03, n04,
               n05, n06, n07, n08);
    }

    //! constructor with index ranges, for ndim() == 9
    Tensor(int n01, int n02, int n03, int n04,
           int n05, int n06, int n07, int n08,
           int n09)
    {
        resize(n01, n02, n03, n04,
               n05, n06, n07, n08,
               n09);
    }

    //! constructor with index ranges, for ndim() == 10
    Tensor(int n01, int n02, int n03, int n04,
           int n05, int n06, int n07, int n08,
           int n09, int n10)
    {
        resize(n01, n02, n03, n04,
               n05, n06, n07, n08,
               n09, n10);
    }

    //! constructor with index ranges, for ndim() == 11
    Tensor(int n01, int n02, int n03, int n04,
           int n05, int n06, int n07, int n08,
           int n09, int n10, int n11)
    {
        resize(n01, n02, n03, n04,
               n05, n06, n07, n08,
               n09, n10, n11);
    }

    //! constructor with index ranges, for ndim() == 12
    Tensor(int n01, int n02, int n03, int n04,
           int n05, int n06, int n07, int n08,
           int n09, int n10, int n11, int n12)
    {
        resize(n01, n02, n03, n04,
               n05, n06, n07, n08,
               n09, n10, n11, n12);
    }

    //
    // Copy semantics
    //

    //! copy constructor
    explicit
    Tensor(const Tensor& x)
    {
        range_ = x.range_;
        stride_ = x.stride_;
#ifdef _HAS_BLAS
        data_.resize(x.size());
        BLAS_TYPE<T>::COPY(x.size(), x.data_.data(), 1, data_.data(), 1);
#else
        data_ = x.data_;
#endif
    }

    //! copy assignment operator
    Tensor& operator= (const Tensor& x)
    {
        range_ = x.range_;
        stride_ = x.stride_;
#ifdef _HAS_BLAS
        data_.resize(x.size());
        BLAS_TYPE<T>::COPY(x.size(), x.data_.data(), 1, data_.data(), 1);
#else
        data_ = x.data_;
#endif
        return *this;
    }

    //
    // Move semantics
    //

    //! move constructor
    explicit
    Tensor(Tensor&& x)
    {
        range_.swap(x.range_);
        stride_.swap(x.stride_);
        data_.swap(x.data_);
    }

    //! move assignment operator
    Tensor& operator= (Tensor&& x)
    {
        range_.swap(x.range_);
        stride_.swap(x.stride_);
        data_.swap(x.data_);
        return *this;
    }

    //
    // Accessors and iterators
    //

    //! number of indices (tensor rank)
    int 
    ndim() const
    { return range_.size(); }

    //! list of index sizes/ranges
    range_type 
    range() const
    { return range_; }

    //! returns true if default-constructed or clear() was called
    bool 
    empty() const
    {
        return data_.empty();
    }

    //
    // Overloaded operators
    //

    //! addition of tensors
    Tensor 
    operator+(const Tensor& x) const
    {
        assert(range_ == x.range_);
        Tensor y(*this);
#ifdef _HAS_BLAS
        BLAS_TYPE<T>::AXPY(x.size(), BLAS_TYPE<T>::ONE, x.data_.data(), 1, y.data_.data(), 1);
#else
        auto ix = x.begin();
        auto iy = y.begin();
        for(; ix != x.end(); ++ix, ++iy)
        {
            *iy += *ix;
        }
#endif
        return y; // calling move semantics automatically
    }

    //! subtraction of tensors
    Tensor 
    operator-(const Tensor& x) const
    {
        assert(range_ == x.range_);
        Tensor y(*this);
#ifdef _HAS_BLAS
        BLAS_TYPE<T>::AXPY(x.size(),-BLAS_TYPE<T>::ONE, x.data_.data(), 1, y.data_.data(), 1);
#else
        auto ix = x.begin();
        auto iy = y.begin();
        for(; ix != x.end(); ++ix, ++iy)
        {
            *iy -= *ix;
        }
#endif
        return y; // calling move semantics automatically
    }

    //! multiply by scalar
    void 
    operator*=(const T& value)
    {
#ifdef _HAS_BLAS
        BTAS_TYPE<T>::SCAL(value, data_.data(), 1);
#else
        for(auto i = data_.begin(); i != data_.end(); ++i)
        {
            *i *= value;
        }
#endif
    }

    //
    // Iteration and element access
    //

    //! begin const iteration over tensor elements
    const_iterator begin() const
    {
        return data_.begin();
    }

    //! end const iteration over tensor elements
    const_iterator end() const
    {
        return data_.end();
    }

    //! begin iteration over tensor elements
    iterator begin()
    {
        return data_.begin();
    }

    //! end iteration over tensor elements
    iterator end()
    {
        return data_.end();
    }

    //! return element i01 without range check (ndim()==1)
    const T& 
    operator()(int i01) const
    {
        range_type index = { i01 };
        return this->operator()(index);
    }

    //! return element i01,i02,... without range check (ndim()==2)
    const T& 
    operator()(int i01, int i02) const
    {
        range_type index = { i01, i02 };
        return this->operator()(index);
    }

    //! return element i01,i02,... without range check (ndim()==3)
    const T& 
    operator()(int i01, int i02, int i03) const
    {
        range_type index = { i01, i02, i03 };
        return this->operator()(index);
    }

    //! return element i01,i02,... without range check (ndim()==4)
    const T& 
    operator()(int i01, int i02, int i03, int i04) const
    {
        range_type index = { i01, i02, i03, i04 };
        return this->operator()(index);
    }

    //! return element i01,i02,... without range check (ndim()==5)
    const T& 
    operator()(int i01, int i02, int i03, int i04,
               int i05) const
    {
        range_type index = { i01, i02, i03, i04,
                             i05 };
        return this->operator()(index);
    }

    //! return element i01,i02,... without range check (ndim()==6)
    const T& 
    operator()(int i01, int i02, int i03, int i04,
               int i05, int i06) const
    {
        range_type index = { i01, i02, i03, i04,
                             i05, i06 };
        return this->operator()(index);
    }

    //! return element i01,i02,... without range check (ndim()==7)
    const T& 
    operator()(int i01, int i02, int i03, int i04,
               int i05, int i06, int i07) const
    {
        range_type index = { i01, i02, i03, i04,
                             i05, i06, i07 };
        return this->operator()(index);
    }

    //! return element i01,i02,... without range check (ndim()==8)
    const T& 
    operator()(int i01, int i02, int i03, int i04,
               int i05, int i06, int i07, int i08) const
    {
        range_type index = { i01, i02, i03, i04,
                             i05, i06, i07, i08 };
        return this->operator()(index);
    }

    //! return element i01,i02,... without range check (ndim()==9)
    const T& 
    operator()(int i01, int i02, int i03, int i04,
               int i05, int i06, int i07, int i08,
               int i09) const
    {
        range_type index = { i01, i02, i03, i04,
                             i05, i06, i07, i08,
                             i09 };
        return this->operator()(index);
    }

    //! return element i01,i02,... without range check (ndim()==10)
    const T& 
    operator()(int i01, int i02, int i03, int i04,
               int i05, int i06, int i07, int i08,
               int i09, int i10) const
    {
        range_type index = { i01, i02, i03, i04,
                             i05, i06, i07, i08,
                             i09, i10 };
        return this->operator()(index);
    }

    //! return element i01,i02,... without range check (ndim()==11)
    const T& 
    operator()(int i01, int i02, int i03, int i04,
               int i05, int i06, int i07, int i08,
               int i09, int i10, int i11) const
    {
        range_type index = { i01, i02, i03, i04,
                             i05, i06, i07, i08,
                             i09, i10, i11 };
        return this->operator()(index);
    }

    //! return element i01,i02,... without range check (ndim()==12)
    const T& 
    operator()(int i01, int i02, int i03, int i04,
               int i05, int i06, int i07, int i08,
               int i09, int i10, int i11, int i12) const
    {
        range_type index = { i01, i02, i03, i04,
                             i05, i06, i07, i08,
                             i09, i10, i11, i12 };
        return this->operator()(index);
    }

    //! return element without range check (ndim()==general)
    const T& 
    operator()(const range_type& index) const
    {
        assert(index.size() == this->ndim());
        return data_[dot(index, stride_)];
    }
    
    //! access element i01 without range check (ndim()==1)
    T& 
    operator()(int i01)
    {
        range_type index = { i01 };
        return this->operator()(index);
    }

    //! access element i01,i02,... without range check (ndim()==2)
    T& 
    operator()(int i01, int i02)
    {
        range_type index = { i01, i02 };
        return this->operator()(index);
    }

    //! access element i01,i02,... without range check (ndim()==3)
    T& 
    operator()(int i01, int i02, int i03)
    {
        range_type index = { i01, i02, i03 };
        return this->operator()(index);
    }

    //! access element i01,i02,... without range check (ndim()==4)
    T& 
    operator()(int i01, int i02, int i03, int i04)
    {
        range_type index = { i01, i02, i03, i04 };
        return this->operator()(index);
    }

    //! access element i01,i02,... without range check (ndim()==5)
    T& 
    operator()(int i01, int i02, int i03, int i04,
               int i05)
    {
        range_type index = { i01, i02, i03, i04,
                             i05 };
        return this->operator()(index);
    }

    //! access element i01,i02,... without range check (ndim()==6)
    T& 
    operator()(int i01, int i02, int i03, int i04,
               int i05, int i06)
    {
        range_type index = { i01, i02, i03, i04,
                             i05, i06 };
        return this->operator()(index);
    }

    //! access element i01,i02,... without range check (ndim()==7)
    T& 
    operator()(int i01, int i02, int i03, int i04,
               int i05, int i06, int i07)
    {
        range_type index = { i01, i02, i03, i04,
                             i05, i06, i07 };
        return this->operator()(index);
    }

    //! access element i01,i02,... without range check (ndim()==8)
    T& 
    operator()(int i01, int i02, int i03, int i04,
               int i05, int i06, int i07, int i08)
    {
        range_type index = { i01, i02, i03, i04,
                             i05, i06, i07, i08 };
        return this->operator()(index);
    }

    //! access element i01,i02,... without range check (ndim()==9)
    T& 
    operator()(int i01, int i02, int i03, int i04,
               int i05, int i06, int i07, int i08,
               int i09)
    {
        range_type index = { i01, i02, i03, i04,
                             i05, i06, i07, i08,
                             i09 };
        return this->operator()(index);
    }

    //! access element i01,i02,... without range check (ndim()==10)
    T& 
    operator()(int i01, int i02, int i03, int i04,
               int i05, int i06, int i07, int i08,
               int i09, int i10)
    {
        range_type index = { i01, i02, i03, i04,
                             i05, i06, i07, i08,
                             i09, i10 };
        return this->operator()(index);
    }

    //! access element i01,i02,... without range check (ndim()==11)
    T& 
    operator()(int i01, int i02, int i03, int i04,
               int i05, int i06, int i07, int i08,
               int i09, int i10, int i11)
    {
        range_type index = { i01, i02, i03, i04,
                             i05, i06, i07, i08,
                             i09, i10, i11 };
        return this->operator()(index);
    }

    //! access element i01,i02,... without range check (ndim()==12)
    T& 
    operator()(int i01, int i02, int i03, int i04,
               int i05, int i06, int i07, int i08,
               int i09, int i10, int i11, int i12)
    {
        range_type index = { i01, i02, i03, i04,
                             i05, i06, i07, i08,
                             i09, i10, i11, i12 };
        return this->operator()(index);
    }

    //! access element without range check (ndim()==general)
    T& 
    operator()(const range_type& index)
    {
        assert(index.size() == this->ndim());
        return data_[dot(index, stride_)];
    }
    
    //! return element i01 with range check (ndim()==1)
    const T& 
    at(int i01) const
    {
        range_type index = { i01 };
        return at(index);
    }

    //! return element i01,i02,... with range check (ndim()==2)
    const T& 
    at(int i01, int i02) const
    {
        range_type index = { i01, i02 };
        return at(index);
    }

    //! return element i01,i02,... with range check (ndim()==3)
    const T& 
    at(int i01, int i02, int i03) const
    {
        range_type index = { i01, i02, i03 };
        return at(index);
    }

    //! return element i01,i02,... with range check (ndim()==4)
    const T& 
    at(int i01, int i02, int i03, int i04) const
    {
        range_type index = { i01, i02, i03, i04 };
        return at(index);
    }

    //! return element i01,i02,... with range check (ndim()==5)
    const T& 
    at(int i01, int i02, int i03, int i04,
       int i05) const
    {
        range_type index = { i01, i02, i03, i04,
                             i05 };
        return at(index);
    }

    //! return element i01,i02,... with range check (ndim()==6)
    const T& 
    at(int i01, int i02, int i03, int i04,
       int i05, int i06) const
    {
        range_type index = { i01, i02, i03, i04,
                             i05, i06 };
        return at(index);
    }

    //! return element i01,i02,... with range check (ndim()==7)
    const T& 
    at(int i01, int i02, int i03, int i04,
       int i05, int i06, int i07) const
    {
        range_type index = { i01, i02, i03, i04,
                             i05, i06, i07 };
        return at(index);
    }

    //! return element i01,i02,... with range check (ndim()==8)
    const T& 
    at(int i01, int i02, int i03, int i04,
       int i05, int i06, int i07, int i08) const
    {
        range_type index = { i01, i02, i03, i04,
                             i05, i06, i07, i08 };
        return at(index);
    }

    //! return element i01,i02,... with range check (ndim()==9)
    const T& 
    at(int i01, int i02, int i03, int i04,
       int i05, int i06, int i07, int i08,
       int i09) const
    {
        range_type index = { i01, i02, i03, i04,
                             i05, i06, i07, i08,
                             i09 };
        return at(index);
    }

    //! return element i01,i02,... with range check (ndim()==10)
    const T& 
    at(int i01, int i02, int i03, int i04,
       int i05, int i06, int i07, int i08,
       int i09, int i10) const
    {
        range_type index = { i01, i02, i03, i04,
                             i05, i06, i07, i08,
                             i09, i10 };
        return at(index);
    }

    //! return element i01,i02,... with range check (ndim()==11)
    const T& 
    at(int i01, int i02, int i03, int i04,
       int i05, int i06, int i07, int i08,
       int i09, int i10, int i11) const
    {
        range_type index = { i01, i02, i03, i04,
                             i05, i06, i07, i08,
                             i09, i10, i11 };
        return at(index);
    }

    //! return element i01,i02,... with range check (ndim()==12)
    const T& 
    at(int i01, int i02, int i03, int i04,
       int i05, int i06, int i07, int i08,
       int i09, int i10, int i11, int i12) const
    {
        range_type index = { i01, i02, i03, i04,
                             i05, i06, i07, i08,
                             i09, i10, i11, i12 };
        return at(index);
    }

    //! return element with range check (ndim()==general)
    const T& 
    at(const range_type& index) const
    {
        assert(index.size() == this->ndim());
        return data_.at(dot(index, stride_));
    }
    
    //! access element i01 with range check (ndim()==1)
    T& 
    at(int i01)
    {
        range_type index = { i01 };
        return at(index);
    }

    //! access element i01,i02,... with range check (ndim()==2)
    T& 
    at(int i01, int i02)
    {
        range_type index = { i01, i02 };
        return at(index);
    }

    //! access element i01,i02,... with range check (ndim()==3)
    T& 
    at(int i01, int i02, int i03)
    {
        range_type index = { i01, i02, i03 };
        return at(index);
    }

    //! access element i01,i02,... with range check (ndim()==4)
    T& 
    at(int i01, int i02, int i03, int i04)
    {
        range_type index = { i01, i02, i03, i04 };
        return at(index);
    }

    //! access element i01,i02,... with range check (ndim()==5)
    T& 
    at(int i01, int i02, int i03, int i04,
       int i05)
    {
        range_type index = { i01, i02, i03, i04,
                             i05 };
        return at(index);
    }

    //! access element i01,i02,... with range check (ndim()==6)
    T& 
    at(int i01, int i02, int i03, int i04,
       int i05, int i06)
    {
        range_type index = { i01, i02, i03, i04,
                             i05, i06 };
        return at(index);
    }

    //! access element i01,i02,... with range check (ndim()==7)
    T& 
    at(int i01, int i02, int i03, int i04,
       int i05, int i06, int i07)
    {
        range_type index = { i01, i02, i03, i04,
                             i05, i06, i07 };
        return at(index);
    }

    //! access element i01,i02,... with range check (ndim()==8)
    T& 
    at(int i01, int i02, int i03, int i04,
       int i05, int i06, int i07, int i08)
    {
        range_type index = { i01, i02, i03, i04,
                             i05, i06, i07, i08 };
        return at(index);
    }

    //! access element i01,i02,... with range check (ndim()==9)
    T& 
    at(int i01, int i02, int i03, int i04,
       int i05, int i06, int i07, int i08,
       int i09)
    {
        range_type index = { i01, i02, i03, i04,
                             i05, i06, i07, i08,
                             i09 };
        return at(index);
    }

    //! access element i01,i02,... with range check (ndim()==10)
    T& 
    at(int i01, int i02, int i03, int i04,
       int i05, int i06, int i07, int i08,
       int i09, int i10)
    {
        range_type index = { i01, i02, i03, i04,
                             i05, i06, i07, i08,
                             i09, i10 };
        return at(index);
    }

    //! access element i01,i02,... with range check (ndim()==11)
    T& 
    at(int i01, int i02, int i03, int i04,
       int i05, int i06, int i07, int i08,
       int i09, int i10, int i11)
    {
        range_type index = { i01, i02, i03, i04,
                             i05, i06, i07, i08,
                             i09, i10, i11 };
        return at(index);
    }

    //! access element i01,i02,... with range check (ndim()==12)
    T& 
    at(int i01, int i02, int i03, int i04,
       int i05, int i06, int i07, int i08,
       int i09, int i10, int i11, int i12)
    {
        range_type index = { i01, i02, i03, i04,
                             i05, i06, i07, i08,
                             i09, i10, i11, i12 };
        return at(index);
    }

    //! access element with range check (ndim()==general)
    T& 
    at(const range_type& index)
    {
        assert(index.size() == this->ndim());
        return data_.at(dot(index, stride_));
    }
    
    //! return number of elements
    int
    size() const
    {
        return data_.size();
    }

    //! returns const pointer to start of elements
    const T* 
    data() const
    {
        // note that this is C++11 feature
        return data_.data();
    }

    //! returns pointer to start of elements
    T* 
    data()
    {
        // note that this is C++11 feature
        return data_.data();
    }

    //! resize array range, for ndim() == 1
    void 
    resize(int n01)
    {
        range_type r = { n01 };
        resize(r);
    }

    //! resize array range, for ndim() == 2
    void 
    resize(int n01, int n02)
    {
        range_type r = { n01, n02 };
        resize(r);
    }

    //! resize array range, for ndim() == 3
    void 
    resize(int n01, int n02, int n03)
    {
        range_type r = { n01, n02, n03 };
        resize(r);
    }

    //! resize array range, for ndim() == 4
    void 
    resize(int n01, int n02, int n03, int n04)
    {
        range_type r = { n01, n02, n03, n04 };
        resize(r);
    }

    //! resize array range, for ndim() == 5
    void 
    resize(int n01, int n02, int n03, int n04,
           int n05)
    {
        range_type r = { n01, n02, n03, n04,
                         n05 };
        resize(r);
    }

    //! resize array range, for ndim() == 6
    void 
    resize(int n01, int n02, int n03, int n04,
           int n05, int n06)
    {
        range_type r = { n01, n02, n03, n04,
                         n05, n06 };
        resize(r);
    }

    //! resize array range, for ndim() == 7
    void 
    resize(int n01, int n02, int n03, int n04,
           int n05, int n06, int n07)
    {
        range_type r = { n01, n02, n03, n04,
                         n05, n06, n07 };
        resize(r);
    }

    //! resize array range, for ndim() == 8
    void 
    resize(int n01, int n02, int n03, int n04,
           int n05, int n06, int n07, int n08)
    {
        range_type r = { n01, n02, n03, n04,
                         n05, n06, n07, n08 };
        resize(r);
    }

    //! resize array range, for ndim() == 9
    void 
    resize(int n01, int n02, int n03, int n04,
           int n05, int n06, int n07, int n08,
           int n09)
    {
        range_type r = { n01, n02, n03, n04,
                         n05, n06, n07, n08,
                         n09 };
        resize(r);
    }

    //! resize array range, for ndim() == 10
    void 
    resize(int n01, int n02, int n03, int n04,
           int n05, int n06, int n07, int n08,
           int n09, int n10)
    {
        range_type r = { n01, n02, n03, n04,
                         n05, n06, n07, n08,
                         n09, n10 };
        resize(r);
    }

    //! resize array range, for ndim() == 11
    void 
    resize(int n01, int n02, int n03, int n04,
           int n05, int n06, int n07, int n08,
           int n09, int n10, int n11)
    {
        range_type r = { n01, n02, n03, n04,
                         n05, n06, n07, n08,
                         n09, n10, n11 };
        resize(r);
    }

    //! resize array range, for ndim() == 12
    void 
    resize(int n01, int n02, int n03, int n04,
           int n05, int n06, int n07, int n08,
           int n09, int n10, int n11, int n12)
    {
        range_type r = { n01, n02, n03, n04,
                         n05, n06, n07, n08,
                         n09, n10, n11, n12 };
        resize(r);
    }

    //! resize array range, for general ndim
    void
    resize(const range_type& range)
    {
        range_ = range;
        size_t str = 1;
        for(size_t i = N-1; i > 0; --i) {
            stride_[i] = str;
            str *= range_[i];
        }
        stride_[0] = str;
        data_.resize(range_[0]*str);
    }

    //! return sliced tensor
    SlicedTensor<T> 
    slice(const range_type& lbound, const range_type& ubound) const;

    //! set random elements
    template<class Generator>
    void 
    randomize(Generator gen = DefaultRandomGen())
    {
        std::generate(data_.begin(), data_.end(), gen);
    }

    //! completely swap state with another Tensor
    void 
    swap(Tensor& x)
    {
        range_.swap(x.range_);
        stride_.swap(x.stride_);
        data_.swap(x.data_);
    }

    //! deallocate storage, empty() will return true
    void 
    clear()
    {
        range_.clear();
        stride_.clear();
        data_.clear();
    }

    private:

    //! enables boost serialization
    friend class boost::serialization::access;

    template<class Archive>
    void 
    serialize(Archive& ar, const unsigned int version)
    {
        ar & range_ & stride_ & data_;
    }

    private:

    // data members go here

    range_type
        range_; //!< range (shape)

    range_type
        stride_; //!< stride

    data_type
        data_; //!< data stored as 1D array

    };

} //namespace btas

#endif
