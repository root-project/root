CXX = g++ #mpicxx # clang #
CXXFLAGS =  -std=c++11 -fopenmp -O3 # -stdlib=libc++

# some variables
GOOGLE_TEST_LIB=/home/zampieri/Documents/CERN/googletest
XGBOOST_ROOT=/home/zampieri/Documents/CERN/xgboost

GOOGLE_TEST_LIB ?= gtest
GOOGLE_TEST_INCLUDE = /usr/local/include

INCLUDE_DIR=-I$(XGBOOST_ROOT)/include \
			-I$(XGBOOST_ROOT)/dmlc-core/include \
			-I$(XGBOOST_ROOT)/rabit/include
LIB_DIR=-L$(XGBOOST_ROOT)/lib



ROOT_FLAGS = `root-config --cflags --glibs`

LD_FLAGS = -L -L.. -lpthread -Lgtest/lib -lgtest

CPPFLAGS = -I../include \
 -I./include \
 -Wno-deprecated \
 -Igtest/include \
 -I$(GOOGLE_TEST_LIB) \
 #-I$(GOOGLE_TEST_INCLUDE)

OBJS = build/testAll.o
EXE = testAll.exe

.PHONY : all clean distclean

all: $(EXE)

testAll.exe: $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@  $^  $(LD_FLAGS) $(ROOT_FLAGS) $(XGBOOST_ROOT)/lib/libxgboost.so

build/testAll.o: testAll.cxx
		$(CXX) $(CXXFLAGS) $(ROOT_FLAGS) -c $< -o $@ $(CPPFLAGS)

clean :
	$(RM) build/*.o *.o

distclean : clean
	$(RM) $(EXE)
