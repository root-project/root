CXX = g++ #mpicxx # clang #
CXXFLAGS =  -std=c++11 -fopenmp -g# -stdlib=libc++

ROOT_FLAGS = `root-config --cflags --glibs`

LD_FLAGS = -L -L.. -lpthread

CPPFLAGS = -I../include \
 -Wno-deprecated \
 -I$(XGBOOST_ROOT)/include -I$(XGBOOST_ROOT)/rabit/include

 XGBOOST_ROOT= /home/zampieri/Documents/CERN/xgboost
 INCLUDE_DIR= -I$(XGBOOST_ROOT)/include -I$(XGBOOST_ROOT)/dmlc-core/include -I$(XGBOOST_ROOT)/rabit/include
 LIB_DIR= -L$(XGBOOST_ROOT)/lib

OBJS = build/arrayBdtPreds.o
EXE = arrayBdtPreds.exe

.PHONY : all clean distclean

all: $(EXE)

# linking
arrayBdtPreds.exe: build/arrayBdtPreds.o
	$(CXX) $(CXXFLAGS) -o $@  $^  $(LD_FLAGS) $(ROOT_FLAGS)

build/arrayBdtPreds.o: arrayBdtPreds.cxx
		$(CXX) $(CXXFLAGS) $(ROOT_FLAGS) -c $< -o $@ $(CPPFLAGS)


clean :
	$(RM) build/*.o *.o

distclean : clean
	$(RM) $(EXE)
