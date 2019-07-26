CXX = g++ #mpicxx
CXXFLAGS = -std=c++11 -g

CPPFLAGS = -I./include \
 -Wno-deprecated \
 -isystem benchmark/include -O3

ROOT_FLAGS = `root-config --cflags --glibs`

LIBFLAGS = -L -Lbuild/src -lbenchmark -lpthread -O3

OBJS = build/benchmark.o build/bdt.o build/unique_bdt.o #build/test.o
EXE = mybenchmark.exe

.PHONY : all clean distclean

all: $(EXE)

$(EXE): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@  $^  $(LIBFLAGS) $(ROOT_FLAGS)

$(OBJS) : build/%.o: %.cxx
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(ROOT_FLAGS) -c $< -o $@





clean :
	$(RM) build/*.o *.o

distclean : clean
	$(RM) $(EXE)
