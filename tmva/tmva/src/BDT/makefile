CXX = g++ #mpicxx
CXXFLAGS = -std=c++11 -Wall -g -ggdb


ROOT_FLAGS = `root-config --cflags --glibs`

CPPFLAGS = -I./include \
 -Wno-deprecated \
 -I.

LIBFLAGS = -L -L.

OBJS = build/main.o build/bdt.o build/unique_bdt.o
EXE = main.exe



.PHONY : all clean distclean

all: $(EXE)

$(EXE): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LIBFLAGS) -o $@  $^  $(ROOT_FLAGS)

#$(TEST_EXE) : $(OBJS)
#	$(CXX) $(CXXFLAGS) $(ROOT_FLAGS) -o $@ $^

$(OBJS) : build/%.o: %.cxx
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(ROOT_FLAGS) -c $< -o $@
	#$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(ROOT_FLAGS) -c $< -o $@







clean :
	$(RM) build/*.o *.o

distclean : clean
	$(RM) $(EXE)
