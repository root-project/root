CXX = g++ #mpicxx
CXXFLAGS = -std=c++11 -Wall -g -fopenmp


XGBOOST_ROOT=/home/zampieri/Documents/CERN/xgboost
INCLUDE_DIR=-I$(XGBOOST_ROOT)/include -I$(XGBOOST_ROOT)/dmlc-core/include -I$(XGBOOST_ROOT)/rabit/include
LIB_DIR=-L$(XGBOOST_ROOT)/lib

ROOT_FLAGS = `root-config --cflags --glibs`

CPPFLAGS = -I./include \
 -Wno-deprecated \
 -I$(XGBOOST_ROOT)/include -I$(XGBOOST_ROOT)/rabit/include

LIBFLAGS = -L -L.

OBJS = build/main.o build/bdt.o build/unique_bdt.o
EXE = main.exe



.PHONY : all clean distclean

all: $(EXE)

$(EXE): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIR) $(LIBFLAGS) -o $@  $^  $(ROOT_FLAGS) $(XGBOOST_ROOT)/lib/libxgboost.so

#$(TEST_EXE) : $(OBJS)
#	$(CXX) $(CXXFLAGS) $(ROOT_FLAGS) -o $@ $^

$(OBJS) : build/%.o: %.cxx
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(ROOT_FLAGS) -c $< -o $@
	#$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(ROOT_FLAGS) -c $< -o $@







clean :
	$(RM) build/*.o *.o

distclean : clean
	$(RM) $(EXE)
