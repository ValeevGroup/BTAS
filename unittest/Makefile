CCCOM=g++ -std=c++11
OPTIMIZATIONS= -O2
DEBUGFLAGS=-DDEBUG -g -O0 -Wall
INCLUDEFLAGS=-I..

#Define source files ----------

SOURCES = test.cc
SOURCES+= range_test.cc
SOURCES+= tensor_test.cc
SOURCES+= tensor_func_test.cc


#Define Flags ----------
CCFLAGS= -I. $(INCLUDEFLAGS) $(OPTIMIZATIONS)
CCGFLAGS= -I. $(INCLUDEFLAGS) $(DEBUGFLAGS)
LIBFLAGS= 
LIBGFLAGS= 

OBJECTS=$(patsubst %.cc,%.o, $(SOURCES))
GOBJECTS=$(patsubst %.cc,.debug_objs/%.o, $(SOURCES))

#Rules ------------------

%.o: %.cc
	$(CCCOM) -c $(CCFLAGS) -o $@ $<

.debug_objs/%.o: %.cc
	$(CCCOM) -c $(CCGFLAGS) -o $@ $<

#Targets -----------------

run: test
	./test

test: $(OBJECTS)
	$(CCCOM) $(CCFLAGS) $(OBJECTS) -o test $(LIBFLAGS)

test-g: mkdebugdir $(GOBJECTS)
	$(CCCOM) $(CCGFLAGS) $(GOBJECTS) -o test-g $(LIBGFLAGS)

mkdebugdir:
	mkdir -p .debug_objs

clean:
	rm -fr *.o .debug_objs test test-g
