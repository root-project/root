CXX = g++ #mpicxx
CXXFLAGS = -std=c++14 -Wall -g

CPPFLAGS = -I./include \
 -Wno-deprecated \

LIBFLAGS = -L

OBJS = build/main.o build/bdt.o
EXE = main.exe

.PHONY : all clean distclean

all: $(EXE)

$(EXE): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(OBJS) : build/%.o: %.cxx
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

clean :
	$(RM) build/*.o *.o

distclean : clean
	$(RM) $(EXE)
