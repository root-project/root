CXX = g++ #mpicxx
CXXFLAGS = -std=c++11

CPPFLAGS = -I./include \
 -Wno-deprecated \
 -isystem benchmark/include

ROOT_FLAGS = `root-config --cflags --glibs`

LIBFLAGS = -L -Lbuild/src -lbenchmark -lpthread -O2

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
