CCCOM=g++ -std=c++11
BTAS_SOURCE=..
INCLUDEFLAGS=-I$(BTAS_SOURCE)

#Define source files ----------

SOURCES = test.cc
SOURCES+= range_test.cc
SOURCES+= tensor_test.cc
SOURCES+= tensor_func_test.cc
SOURCES+= contract_test.cc
SOURCES+= dot_test.cc


#Define Flags ----------
CCFLAGS= -I. $(INCLUDEFLAGS) -g

OBJECTS=$(patsubst %.cc,%.o, $(SOURCES))

%.o: %.cc
	$(CCCOM) -c $(CCFLAGS) -o $@ $<

#Targets -----------------

run: test
	./test

test: $(OBJECTS)
	$(CCCOM) $(CCFLAGS) $(OBJECTS) -o test $(LIBFLAGS)

clean:
	rm -fr *.o test

DEP_HEADERS = $(BTAS_SOURCE)/btas/range.h
range_test.o: $(DEP_HEADERS)

DEP_HEADERS += $(BTAS_SOURCE)/btas/tensor.h
tensor_test.o: $(DEP_HEADERS)

DEP_HEADERS += $(BTAS_SOURCE)/btas/tensor_func.h
tensor_func_test.o: $(DEP_HEADERS)

DEP_HEADERS += $(BTAS_SOURCE)/btas/generic/contract.h
DEP_HEADERS += $(BTAS_SOURCE)/btas/generic/dot_impl.h

contract_test.o: $(DEP_HEADERS)
dot_test.o: $(DEP_HEADERS)
