CXX = mpicxx
CXXFLAGS = -std=c++11 -Wall -g

CPPFLAGS = -I./include -I/home/zampieri/Documents/CERN/PicklingTools170Release/C++ -pthread -D_REENTRANT \
-I/home/zampieri/Documents/CERN/PicklingTools170Release/opencontainers_1_8_5/include \
-I/home/zampieri/Documents/CERN/PicklingTools170Release/C++ \
-fno-strict-aliasing -DLINUX_ -DOC_NEW_STYLE_INCLUDES -Wno-deprecated \
-I/home/zampieri/Documents/CERN/PicklingTools170Release/C++/opencontainers_1_8_5/include


LIBFLAGS = -L

OBJS = my_pickle.o
EXE = my_pickle

.PHONY : all clean distclean

all: $(EXE)

$(EXE): my_pickle.o
	$(CXX) $(CXXFLAGS) -o $@ $^

$(OBJS) : %.o: %.cxx
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@


clean :
	$(RM) build/*.o *.o

distclean : clean
	$(RM) $(EXE)
