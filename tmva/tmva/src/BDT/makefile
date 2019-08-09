CXX=g++ #g++ #clang++-libc++ #g++ #mpicxx
CXXFLAGS = -std=c++11 -Wall -g -ggdb

USE_CLANG=0

CPPFLAGS = -I./include \
 -Wno-deprecated \
 -I.

LIBFLAGS = -L -L.

OBJS = build/main.o build/bdt.o build/unique_bdt.o
EXE = main.exe

ROOT_FLAGS = `root-config --cflags --glibs`
ifeq ($(USECLANG),1)
        ROOT_FLAGS =
else
        ROOT_FLAGS = `root-config --cflags --glibs`
endif


.PHONY : all clean distclean

all: $(EXE)

$(EXE): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LIBFLAGS) -o $@  $^  $(ROOT_FLAGS)

#$(TEST_EXE) : $(OBJS)
#	$(CXX) $(CXXFLAGS) $(ROOT_FLAGS) -o $@ $^

$(OBJS) : build/%.o: %.cxx
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(ROOT_FLAGS) -c $< -o $@
	#$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@







clean :
	$(RM) build/*.o *.o

distclean : clean
	$(RM) $(EXE)
