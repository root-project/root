CXX = g++ #mpicxx
CXXFLAGS = -std=c++11 -Wall -g

CPPFLAGS = -I./include \
 -Wno-deprecated \

ROOT_FLAGS = `root-config --cflags --glibs`
#ROOT_FLAGS = $(shell root-config --cflags)
ROOT_BIS = -I$(ROOTSYS)/include
ROOTCXXFLAGS=`root-config --cflags`
LDFLAGSROOT=`root-config --glibs`

LIBFLAGS = -L

OBJS = build/main.o build/bdt.o build/unique_bdt.o #build/test.o
EXE = main.exe

.PHONY : all clean distclean

all: $(EXE)

$(EXE): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@  $^  $(ROOT_FLAGS)

#$(TEST_EXE) : $(OBJS)
#	$(CXX) $(CXXFLAGS) $(ROOT_FLAGS) -o $@ $^

$(OBJS) : build/%.o: %.cxx
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(ROOT_FLAGS) -c $< -o $@
	#$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(ROOT_FLAGS) -c $< -o $@



clean :
	$(RM) build/*.o *.o

distclean : clean
	$(RM) $(EXE)
