#ifndef BTAS_KRP
#define BTAS_KRP
#include <btas/btas.h>
namespace btas{
	template<class Tensor>
	void KhatriRaoProduct(const Tensor &A, const Tensor &B, Tensor &AB_product) { 
	 	// The khatri-rao product is an outer product of column vectors in
	  // two matrices, then ordered to make a super column in a new matrix
	  // The dimension of this product is  B(NXM) (.) C(KXM) = D(N*K X M)
	  if(A.rank() > 2 || B.rank() > 2)
	  	BTAS_EXCEPTION_MESSAGE(__FILE__, __LINE__, "A.rank() > 2 || B.rank() > 2, Matrices required");
	  AB_product.resize(
	      Range{Range1{A.extent(0) * B.extent(0)}, Range1{A.extent(1)}});
	  for(auto i = 0; i < A.extent(0); ++i)
	  	for(auto j = 0; j < B.extent(0); ++j)
	  		for(auto k = 0; k < A.extent(1); ++k)
	  			AB_product(i*B.extent(0)+j, k) = A(i,k) * B(j,k);
	}
} // namespace btas
#endif //BTAS_KRP