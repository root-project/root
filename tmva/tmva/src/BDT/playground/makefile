CXX = g++ #mpicxx # clang #
CXXFLAGS =  -std=c++11 -fopenmp -O3 # -stdlib=libc++

ROOT_FLAGS = `root-config --cflags --glibs`

LD_FLAGS = -L -L.. -lpthread

CPPFLAGS = -I../include \
 -Wno-deprecated

OBJS = build/arrayBdtPreds.o build/uniqueBdtPreds.o \
			build/uniqueBdtBatch.o build/unique_bdt.o \
			 build/JitForestBdtPreds.o build/JitForestBdtAll.o\
			 build/JitForestBdtAll_Static.o
EXE = arrayBdtPreds.exe uniqueBdtPreds.exe \
      uniqueBdtBatch.exe JitForestBdtPreds.exe \
			JitForestBdtAll.exe JitForestBdtAll_Static.exe

.PHONY : all clean distclean

all: $(EXE)

# linking
JitForestBdtAll_Static.exe: build/JitForestBdtAll_Static.o build/unique_bdt.o
	$(CXX) $(CXXFLAGS) -o $@  $^  $(LD_FLAGS) $(ROOT_FLAGS)

JitForestBdtAll.exe: build/JitForestBdtAll.o build/unique_bdt.o
	$(CXX) $(CXXFLAGS) -o $@  $^  $(LD_FLAGS) $(ROOT_FLAGS)

JitForestBdtPreds.exe: build/JitForestBdtPreds.o build/unique_bdt.o
	$(CXX) $(CXXFLAGS) -o $@  $^  $(LD_FLAGS) $(ROOT_FLAGS)

arrayBdtPreds.exe: build/arrayBdtPreds.o build/unique_bdt.o
	$(CXX) $(CXXFLAGS) -o $@  $^  $(LD_FLAGS) $(ROOT_FLAGS)

uniqueBdtPreds.exe: build/uniqueBdtPreds.o build/unique_bdt.o
	$(CXX) $(CXXFLAGS) -o $@  $^  $(LD_FLAGS) $(ROOT_FLAGS)

uniqueBdtBatch.exe: build/uniqueBdtBatch.o build/unique_bdt.o
	$(CXX) $(CXXFLAGS) -o $@  $^  $(LD_FLAGS) $(ROOT_FLAGS)

# compilation
build/JitForestBdtAll_Static.o: JitForestBdtAll_Static.cxx
		$(CXX) $(CXXFLAGS) $(ROOT_FLAGS) -c $< -o $@ $(CPPFLAGS)

build/JitForestBdtAll.o: JitForestBdtAll.cxx
		$(CXX) $(CXXFLAGS) $(ROOT_FLAGS) -c $< -o $@ $(CPPFLAGS)

build/JitForestBdtPreds.o: JitForestBdtPreds.cxx
		$(CXX) $(CXXFLAGS) $(ROOT_FLAGS) -c $< -o $@ $(CPPFLAGS)

build/uniqueBdtBatch.o: uniqueBdtBatch.cxx
		$(CXX) $(CXXFLAGS) $(ROOT_FLAGS) -c $< -o $@ $(CPPFLAGS)

build/uniqueBdtPreds.o: uniqueBdtPreds.cxx
		$(CXX) $(CXXFLAGS) $(ROOT_FLAGS) -c $< -o $@ $(CPPFLAGS)

build/arrayBdtPreds.o: arrayBdtPreds.cxx
		$(CXX) $(CXXFLAGS) $(ROOT_FLAGS) -c $< -o $@ $(CPPFLAGS)


build/unique_bdt.o:  ../src/unique_bdt.cxx
		$(CXX) $(CXXFLAGS) $(ROOT_FLAGS) -c $< -o $@ $(CPPFLAGS)

clean :
	$(RM) build/*.o *.o

distclean : clean
	$(RM) $(EXE)
