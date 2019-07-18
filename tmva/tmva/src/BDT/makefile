CXX = g++ #mpicxx
CXXFLAGS = -std=c++11 -Wall -g

CPPFLAGS = -I./include \
 -Wno-deprecated \

#ROOT_FLAGS = root-config --cflags --glibs
ROOT_FLAGS = $(shell root-config --cflags)

LIBFLAGS = -L

OBJS = build/main.o build/bdt.o build/unique_bdt.o #build/test.o
EXE = main.exe
TEST_EXE = test.exe

.PHONY : all clean distclean

all: $(EXE) #$(TEST_EXE)

$(EXE): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

#$(TEST_EXE) : $(OBJS)
#	$(CXX) $(CXXFLAGS) $(ROOT_FLAGS) -o $@ $^

$(OBJS) : build/%.o: %.cxx
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@
	#$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(ROOT_FLAGS) -c $< -o $@



clean :
	$(RM) build/*.o *.o

distclean : clean
	$(RM) $(EXE) $(TEST_EXE)
