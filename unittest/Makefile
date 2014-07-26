BTAS_SOURCE = ..
BOOSTPREFIX = /opt/local

CXX = clang++ -std=c++11 
CXXFLAGS= -g
CPPFLAGS += -I$(BTAS_SOURCE)
LDFLAGS += -lboost_serialization

#Define source files ----------

SOURCES = test.cc
SOURCES+= range_test.cc
SOURCES+= tensor_test.cc
SOURCES+= tensorview_test.cc
SOURCES+= tensor_func_test.cc
SOURCES+= contract_test.cc
SOURCES+= tensor_blas_test.cc

OBJECTS=$(SOURCES:%.cc=%.o)

run: test
	./test

test: $(OBJECTS)
	$(CXX) $^ -o $@ $(LDFLAGS) 

clean:
	rm -fr *.o test

DEP_HEADERS = $(BTAS_SOURCE)/btas/range.h
range_test.o: $(DEP_HEADERS)

DEP_HEADERS += $(BTAS_SOURCE)/btas/tensor.h
tensor_test.o: $(DEP_HEADERS)

DEP_HEADERS += $(BTAS_SOURCE)/btas/tensor_func.h
tensor_func_test.o: $(DEP_HEADERS)

DEP_HEADERS += $(BTAS_SOURCE)/btas/tensor_func.h
tensorview_test.o: $(DEP_HEADERS)

DEP_HEADERS += $(BTAS_SOURCE)/btas/generic/contract.h
DEP_HEADERS += $(BTAS_SOURCE)/btas/generic/dot_impl.h
DEP_HEADERS += $(BTAS_SOURCE)/btas/generic/scal_impl.h
DEP_HEADERS += $(BTAS_SOURCE)/btas/generic/axpy_impl.h
DEP_HEADERS += $(BTAS_SOURCE)/btas/generic/gemv_impl.h
DEP_HEADERS += $(BTAS_SOURCE)/btas/generic/gemm_impl.h
DEP_HEADERS += $(BTAS_SOURCE)/btas/generic/ger_impl.h

contract_test.o: $(DEP_HEADERS)
dot_test.o: $(DEP_HEADERS)
