CXX =  g++ # g++ clang++ #  #mpicxx #
CXXFLAGS = -std=c++11 -lstdc++ -g -fopenmp -O3 #  -lc++abi -stdlib=libc++

USE_CLANG = 0

CPPFLAGS = -I./include \
 -Wno-deprecated \
 -isystem benchmark/include \
  -I$(XGBOOST_ROOT)/include -I$(XGBOOST_ROOT)/rabit/include

 XGBOOST_ROOT= /home/zampieri/Documents/CERN/xgboost
 INCLUDE_DIR= -I$(XGBOOST_ROOT)/include -I$(XGBOOST_ROOT)/dmlc-core/include -I$(XGBOOST_ROOT)/rabit/include
 LIB_DIR= -L$(XGBOOST_ROOT)/lib


ROOT_FLAGS = `root-config --cflags --glibs`

LIBFLAGS = -L -Lbuild/src -lpthread -O3 -lbenchmark # /home/zampieri/Documents/CERN/root/tmva/tmva/src/BDT/libbenchmark.a#

DEPS = jitted_bdt.h forest.h

OBJS = build/benchmark.o build/bdt.o build/unique_bdt.o #build/test.o
EXE = mybenchmark.exe

.PHONY : all clean distclean

ifeq ($(USECLANG),1)
				 CXXFLAGS += -v -stdlib=libc++
         ROOT_FLAGS =
				 LIBFLAGS = -L -Lbuild/src -lpthread -O3
				 XGBOOST_ROOT =
				 INCLUDE_DIR =
				 LIB_DIR =
else
         # ROOT_FLAGS = `root-config --cflags --glibs`
endif


all: $(EXE)

$(EXE): $(OBJS) $(DEPS)
	#$(CXX) $(CXXFLAGS) -o $@  $^  $(LIBFLAGS) -lbenchmark
	$(CXX) $(CXXFLAGS) -o $@  $^  $(LIBFLAGS) $(ROOT_FLAGS) $(XGBOOST_ROOT)/lib/libxgboost.so

$(OBJS) : build/%.o: %.cxx
	#$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(ROOT_FLAGS) -c $< -o $@





clean :
	$(RM) build/*.o *.o

distclean : clean
	$(RM) $(EXE)
