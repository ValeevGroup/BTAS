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
    Tensor();

    //! constructor with index range, for ndim() == 1
    explicit 
    Tensor(int n01);

    //! constructor with index ranges, for ndim() == 2
    Tensor(int n01, int n02);

    //! constructor with index ranges, for ndim() == 3
    Tensor(int n01, int n02, int n03);

    //! constructor with index ranges, for ndim() == 4
    Tensor(int n01, int n02, int n03, int n04);

    //! constructor with index ranges, for ndim() == 5
    Tensor(int n01, int n02, int n03, int n04,
           int n05);

    //! constructor with index ranges, for ndim() == 6
    Tensor(int n01, int n02, int n03, int n04,
           int n05, int n06);

    //! constructor with index ranges, for ndim() == 7
    Tensor(int n01, int n02, int n03, int n04,
           int n05, int n06, int n07);

    //! constructor with index ranges, for ndim() == 8
    Tensor(int n01, int n02, int n03, int n04,
           int n05, int n06, int n07, int n08);

    //! constructor with index ranges, for ndim() == 9
    Tensor(int n01, int n02, int n03, int n04,
           int n05, int n06, int n07, int n08,
           int n09);

    //! constructor with index ranges, for ndim() == 10
    Tensor(int n01, int n02, int n03, int n04,
           int n05, int n06, int n07, int n08,
           int n09, int n10);

    //! constructor with index ranges, for ndim() == 11
    Tensor(int n01, int n02, int n03, int n04,
           int n05, int n06, int n07, int n08,
           int n09, int n10, int n11);

    //! constructor with index ranges, for ndim() == 12
    Tensor(int n01, int n02, int n03, int n04,
           int n05, int n06, int n07, int n08,
           int n09, int n10, int n11, int n12);

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


    //! return element i01 without range check (ndim()==1)
    T 
    operator()(int i01) const;

    //! return element i01,i02,... without range check (ndim()==2)
    T 
    operator()(int i01, int i02) const;

    //! return element i01,i02,... without range check (ndim()==3)
    T 
    operator()(int i01, int i02, int i03) const;

    //! return element i01,i02,... without range check (ndim()==4)
    T 
    operator()(int i01, int i02, int i03, int i04) const;

    //! return element i01,i02,... without range check (ndim()==5)
    T 
    operator()(int i01, int i02, int i03, int i04,
               int i05) const;

    //! return element i01,i02,... without range check (ndim()==6)
    T 
    operator()(int i01, int i02, int i03, int i04,
               int i05, int i06) const;

    //! return element i01,i02,... without range check (ndim()==7)
    T 
    operator()(int i01, int i02, int i03, int i04,
               int i05, int i06, int i07) const;

    //! return element i01,i02,... without range check (ndim()==8)
    T 
    operator()(int i01, int i02, int i03, int i04,
               int i05, int i06, int i07, int i08) const;

    //! return element i01,i02,... without range check (ndim()==9)
    T 
    operator()(int i01, int i02, int i03, int i04,
               int i05, int i06, int i07, int i08,
               int i09) const;

    //! return element i01,i02,... without range check (ndim()==10)
    T 
    operator()(int i01, int i02, int i03, int i04,
               int i05, int i06, int i07, int i08,
               int i09, int i10) const;

    //! return element i01,i02,... without range check (ndim()==11)
    T 
    operator()(int i01, int i02, int i03, int i04,
               int i05, int i06, int i07, int i08,
               int i09, int i10, int i11) const;

    //! return element i01,i02,... without range check (ndim()==12)
    T 
    operator()(int i01, int i02, int i03, int i04,
               int i05, int i06, int i07, int i08,
               int i09, int i10, int i11, int i12) const;

    //! access element i01 without range check (ndim()==1)
    T& 
    operator()(int i01);

    //! access element i01,i02,... without range check (ndim()==2)
    T& 
    operator()(int i01, int i02);

    //! access element i01,i02,... without range check (ndim()==3)
    T& 
    operator()(int i01, int i02, int i03);

    //! access element i01,i02,... without range check (ndim()==4)
    T& 
    operator()(int i01, int i02, int i03, int i04);

    //! access element i01,i02,... without range check (ndim()==5)
    T& 
    operator()(int i01, int i02, int i03, int i04,
               int i05);

    //! access element i01,i02,... without range check (ndim()==6)
    T& 
    operator()(int i01, int i02, int i03, int i04,
               int i05, int i06);

    //! access element i01,i02,... without range check (ndim()==7)
    T& 
    operator()(int i01, int i02, int i03, int i04,
               int i05, int i06, int i07);

    //! access element i01,i02,... without range check (ndim()==8)
    T& 
    operator()(int i01, int i02, int i03, int i04,
               int i05, int i06, int i07, int i08);

    //! access element i01,i02,... without range check (ndim()==9)
    T& 
    operator()(int i01, int i02, int i03, int i04,
               int i05, int i06, int i07, int i08,
               int i09);

    //! access element i01,i02,... without range check (ndim()==10)
    T& 
    operator()(int i01, int i02, int i03, int i04,
               int i05, int i06, int i07, int i08,
               int i09, int i10);

    //! access element i01,i02,... without range check (ndim()==11)
    T& 
    operator()(int i01, int i02, int i03, int i04,
               int i05, int i06, int i07, int i08,
               int i09, int i10, int i11);

    //! access element i01,i02,... without range check (ndim()==12)
    T& 
    operator()(int i01, int i02, int i03, int i04,
               int i05, int i06, int i07, int i08,
               int i09, int i10, int i11, int i12);

    //! return element i01 with range check (ndim()==1)
    T 
    at(int i01) const;

    //! return element i01,i02,... with range check (ndim()==2)
    T 
    at(int i01, int i02) const;

    //! return element i01,i02,... with range check (ndim()==3)
    T 
    at(int i01, int i02, int i03) const;

    //! return element i01,i02,... with range check (ndim()==4)
    T 
    at(int i01, int i02, int i03, int i04) const;

    //! return element i01,i02,... with range check (ndim()==5)
    T 
    at(int i01, int i02, int i03, int i04,
               int i05) const;

    //! return element i01,i02,... with range check (ndim()==6)
    T 
    at(int i01, int i02, int i03, int i04,
               int i05, int i06) const;

    //! return element i01,i02,... with range check (ndim()==7)
    T 
    at(int i01, int i02, int i03, int i04,
               int i05, int i06, int i07) const;

    //! return element i01,i02,... with range check (ndim()==8)
    T 
    at(int i01, int i02, int i03, int i04,
               int i05, int i06, int i07, int i08) const;

    //! return element i01,i02,... with range check (ndim()==9)
    T 
    at(int i01, int i02, int i03, int i04,
               int i05, int i06, int i07, int i08,
               int i09) const;

    //! return element i01,i02,... with range check (ndim()==10)
    T 
    at(int i01, int i02, int i03, int i04,
               int i05, int i06, int i07, int i08,
               int i09, int i10) const;

    //! return element i01,i02,... with range check (ndim()==11)
    T 
    at(int i01, int i02, int i03, int i04,
               int i05, int i06, int i07, int i08,
               int i09, int i10, int i11) const;

    //! return element i01,i02,... with range check (ndim()==12)
    T 
    at(int i01, int i02, int i03, int i04,
               int i05, int i06, int i07, int i08,
               int i09, int i10, int i11, int i12) const;

    //! access element i01 with range check (ndim()==1)
    T& 
    at(int i01);

    //! access element i01,i02,... with range check (ndim()==2)
    T& 
    at(int i01, int i02);

    //! access element i01,i02,... with range check (ndim()==3)
    T& 
    at(int i01, int i02, int i03);

    //! access element i01,i02,... with range check (ndim()==4)
    T& 
    at(int i01, int i02, int i03, int i04);

    //! access element i01,i02,... with range check (ndim()==5)
    T& 
    at(int i01, int i02, int i03, int i04,
       int i05);

    //! access element i01,i02,... with range check (ndim()==6)
    T& 
    at(int i01, int i02, int i03, int i04,
       int i05, int i06);

    //! access element i01,i02,... with range check (ndim()==7)
    T& 
    at(int i01, int i02, int i03, int i04,
       int i05, int i06, int i07);

    //! access element i01,i02,... with range check (ndim()==8)
    T& 
    at(int i01, int i02, int i03, int i04,
       int i05, int i06, int i07, int i08);

    //! access element i01,i02,... with range check (ndim()==9)
    T& 
    at(int i01, int i02, int i03, int i04,
       int i05, int i06, int i07, int i08,
       int i09);

    //! access element i01,i02,... with range check (ndim()==10)
    T& 
    at(int i01, int i02, int i03, int i04,
       int i05, int i06, int i07, int i08,
       int i09, int i10);

    //! access element i01,i02,... with range check (ndim()==11)
    T& 
    at(int i01, int i02, int i03, int i04,
       int i05, int i06, int i07, int i08,
       int i09, int i10, int i11);

    //! access element i01,i02,... with range check (ndim()==12)
    T& 
    at(int i01, int i02, int i03, int i04,
       int i05, int i06, int i07, int i08,
       int i09, int i10, int i11, int i12);
    
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
