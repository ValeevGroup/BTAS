#ifndef __BTAS_TENSOR_H
#define __BTAS_TENSOR_H

namespace btas
{

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

    //
    // Constructors
    //

    //! default constructor
    TArray();

    //! constructor with array range, for ndim() == 1
    explicit 
    TArray(int n01) 

    //! constructor with array range, for ndim() == 2
    TArray(int n01, int n02) 

    //etc... up to 10 or so indices

    //
    // Accessors and iterators
    //

    //! number of indices (tensor rank)
    int 
    ndim() const;

    //! list of index sizes/ranges
    range_type 
    range() const;

    //! returns true if default-constructed or clear() was called
    bool 
    empty();


    //
    // Overloaded operators
    // (prefer external methods?)
    //

    //! addition of tensors
    Tensor 
    operator+(Tensor &);

    //! subtraction of tensors
    Tensor 
    operator-(Tensor &);

    //! multiply by scalar
    void 
    operator*=(T &);


    //
    // Iteration and element access
    //

    //! begin const iteration over tensor elements
    const_iterator begin() const 

    //! end const iteration over tensor elements
    const_iterator end() const

    //! begin iteration over tensor elements
    iterator begin() 

    //! end iteration over tensor elements
    iterator end() ;


    //! returns element i01 (ndim() == 1) without range check
    T 
    operator[](int i01) const;

    //! returns element i01,i02 (ndim() == 2) without range check
    T 
    operator[](int i01, int i02) const;

    //etc... up to 10 or so indices

    //! returns element i01 (ndim() == 1) without range check
    T& 
    operator[](int i01);

    //! returns element i01,i02 (ndim() == 2) without range check
    T&
    operator[](int i01, int i02);

    //etc... up to 10 or so indices
    
    //! returns element i01 (ndim() == 1) with range check
    T 
    at(int i01) const;

    //! returns element i01,i02 (ndim() == 2) with range check
    T 
    at(int i01, int i02) const;

    //etc... up to 10 or so indices

    //! returns element i01 (ndim() == 1) with range check
    T& 
    at(int i01);

    //! returns element i01,i02 (ndim() == 2) with range check
    T&
    at(int i01, int i02);

    //etc... up to 10 or so indices

    //! return number of elements in storage
    int
    size() const;

    //! returns const pointer to start of elements
    const T* 
    data() const;

    //! returns pointer to start of elements
    T* 
    data();   

    //! resize array range, for N = 1
    void 
    resize(int n01) ;

    //etc... up to 10 or so indices

    //! slice array to return sub-array object
    TSubArray<T> 
    subarray(const IVector<N>& lbound, const IVector<N>& ubound) const 

    //! set random elements
    template<class Generator>
    void 
    randomize(Generator gen = DefaultRandomGen());

    //! completely swap state with another Tensor
    void 
    swap(Tensor &);

    //! deallocate storage, empty() will return true
    void 
    clear() 

    //! enables boost serialization
    friend class boost::serialization::access;

    template<class Archive>
    void 
    serialize(Archive& ar, const unsigned int version);

    private:

    //data members go here

    };

} //namespace btas

#endif
