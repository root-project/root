CXX = g++ #mpicxx # clang #
CXXFLAGS =  -std=c++11 -g -fopenmp -O3 # -stdlib=libc++

CPPFLAGS = -I./include \
 -Wno-deprecated \
 -gtest/include
  -I$(XGBOOST_ROOT)/include -I$(XGBOOST_ROOT)/rabit/include

 XGBOOST_ROOT=/home/zampieri/Documents/CERN/xgboost
 INCLUDE_DIR=-I$(XGBOOST_ROOT)/include -I$(XGBOOST_ROOT)/dmlc-core/include -I$(XGBOOST_ROOT)/rabit/include
 LIB_DIR=-L$(XGBOOST_ROOT)/lib



ROOT_FLAGS = `root-config --cflags --glibs`

LIBFLAGS = -L -Lbuild/src -lbenchmark -lpthread -O3

OBJS = build/tests.o build/bdt.o build/unique_bdt.o #build/test.o
EXE = tests.exe

.PHONY : all clean distclean

all: $(EXE)

$(EXE): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@  $^  $(LIBFLAGS) $(ROOT_FLAGS) $(XGBOOST_ROOT)/lib/libxgboost.so

$(OBJS) : build/%.o: %.cxx
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(ROOT_FLAGS) -c $< -o $@





clean :
	$(RM) build/*.o *.o

distclean : clean
	$(RM) $(EXE)
