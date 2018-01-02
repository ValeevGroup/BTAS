#ifndef BTAS_SWAP
#define BTAS_SWAP
#ifdef _HAS_INTEL_MKL
#include <mkl_trans.h>
#include <btas/btas.h>
#include <vector>
					//***IMPORTANT***//
//do not use swap to first then use swap to back
//swap to first preserves order while swap to back does not
//If you use swap to first, to undo the transpositions
//make LD_in_front = true and same goes for swap to back
//do not mix swap to first and swap to back
namespace btas{
	template <typename Tensor>
	void swap_to_first(Tensor& A, int mode, bool LD_in_front = false, bool for_ALS_update = true){
		//For switching when the dimension of interests is in the back use b/B
		//otherwise use f/F
		// A{T1,T2,T3,T4,T5} (3,b) -> A{T3,T1,T2,T4,T5}
		// A{T3, T1, T2, T4, T5} (3, f) -> A{T1, T2, T3, T4, T5}
		if(mode == 0)
			return;
		std::vector<int> aug_dims;
		auto size = A.range().area();
		for(int i = 0; i < A.rank(); i++){
			aug_dims.push_back(A.extent(i));
		}
		if(for_ALS_update){
			auto temp = aug_dims[0];
			aug_dims[0] = aug_dims[mode];
			aug_dims[mode] = temp;
		}
		else{
			auto temp = (LD_in_front) ? aug_dims[0]: aug_dims[mode];
			auto erase = (LD_in_front) ? aug_dims.begin(): aug_dims.begin() + mode;
			auto begin = (LD_in_front) ? aug_dims.begin() + mode : aug_dims.begin();
			auto end = begin + 1;
			aug_dims.erase(erase);
			aug_dims.insert(begin, temp);
		}
		size_t rows = 1;
		size_t cols = 1;
		auto step = 1;
		if(mode == A.rank() - 1){
			rows = (LD_in_front) ? A.extent(0) : size / A.extent(mode);
			cols = (LD_in_front) ? size / A.extent(0) : A.extent(mode);

			double * data_ptr = A.data();
			mkl_dimatcopy('R', 'T', rows, cols, 1.0, data_ptr, cols, rows);
		}
		else{
			for(int i = 0; i <= mode; i++)
				rows *= A.extent(i);
			cols = size / rows;
			double * data_ptr = A.data();

			mkl_dimatcopy('R', 'T', rows, cols, 1.0, data_ptr, cols, rows);

			step = rows;
			size_t in_rows = (LD_in_front) ? A.extent(0): rows/A.extent(mode);
			size_t in_cols = (LD_in_front) ? rows/A.extent(0) : A.extent(mode);
			
			for(int i = 0; i < cols; i++){
				data_ptr = A.data() + i * step;
				mkl_dimatcopy('R', 'T', in_rows, in_cols, 1.0, data_ptr, in_cols, in_rows);
			}
			data_ptr = A.data();
			mkl_dimatcopy('R', 'T', cols, rows, 1.0, data_ptr, rows, cols);
		}
		A.resize(aug_dims);
	}
	template <typename Tensor>
	void swap_to_back(Tensor& A, int mode, bool is_in_back = false){
  	if(mode > A.rank())
  		BTAS_EXCEPTION_MESSAGE(__FILE__, __LINE__, "mode > A.rank(), mode out of range");
  	if(mode == A.rank()-1)
  		return;
  	size_t rows = 1;
  	size_t cols = 1;
  	auto ndim = A.rank();
  	auto midpoint = (is_in_back) ? ndim - 1 - mode: mode+1;
  	std::vector<size_t> aug_dims;
  	for(int i = midpoint; i< ndim; i++){
  		aug_dims.push_back(A.extent(i));
  		cols *= A.extent(i);
  	}
  	for(int i = 0; i < midpoint; i++){
			aug_dims.push_back(A.extent(i));
			rows *= A.extent(i);
		}
		double * data_ptr = A.data();
		mkl_dimatcopy('R', 'T', rows, cols, 1.0, data_ptr, cols, rows);
		A.resize(aug_dims);
		return;
  }
}//namespace btas
#endif //_HAS_INTEL_MKL
#endif //BTAS_SWAP