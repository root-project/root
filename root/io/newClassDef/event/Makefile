local_all: main
.PHONY: main

include $(ROOTSYS)/test/Makefile

arch := $(shell grep arch `which root-config` | head -1 | cut -d'=' -f2-)

ifeq ("X","Y")
CXXFLAGS := $(shell root-config --cflags)
LIBS := $(shell root-config --libs)

ifeq ($(findstring kcc,$(arch)),kcc)

CC = KCC --no_exceptions
LD = KCC --no_exceptions

else

ifeq ($(findstring sgicc,$(arch)),sgicc)
CC = CC -n32 -I/usr/include/CC.sgi 
LD = CC -n32 -I/usr/include/CC.sgi 

else

ifeq ($(findstring solaris,$(arch)),solaris)

CC = CC 
LD = CC 

else

CC = g++
LD = g++

endif
endif
endif
endif

ifeq ($(ARCH),win32)
CXXFLAGS += /TP /GX
LIBSFORLINK = $(LIBS)
LDFLAGS := /link $(LDFLAGS)
SYSLIBS += msvcprt.lib 
else
LIBSFORLINK = 
endif

test2:
	echo $(arch)

all:	driver$(ExeSuf) libIotest.$(DllSuf)x
	./driver$(ExeSuf)

test:   driver$(ExeSuf)
	rm driver.output
	./driver$(ExeSuf) 2>&1 | grep -v -- '->[^0][^ ]' | sed -e 's/Dumping object at:[0-9a-f]*,/Dumping object,/' > driver.output 
	diff good.output driver.output
	@echo "All test succeeded in `pwd`"

local_clean:
	rm -rf *.o *.so ti_files ii_files 
	rm -rf mydict.* driver$(ExeSuf) *~

clean: local_clean


#driver: main.cc mydict.cc RootCaloHit.h  RootData.h  RootPCellID.h $(ROOTSYS)/include/Rtypes.h


driver$(ExeSuf): main.cc mydict.cc
	$(CXX)  $^ $(CXXFLAGS) $(LDFLAGS) $(LIBS) $(OutPutOpt)$@

stuff:
ifeq ($(ARCH),aix)
		/usr/ibmcxx/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
else
ifeq ($(ARCH),aix5)
		/usr/vacpp/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
else
ifeq ($(ARCH),alphacxx)
# due to a bug in cxx/ld under osf3.xx, one cannot use cxx to generate
# a shared library. One must use ld instead.
		ld -L/usr/lib/cmplrs/cxx -rpath /usr/lib/cmplrs/cxx \
		   -expect_unresolved "*" -g0 -O1 -shared \
		   /usr/lib/cmplrs/cc/crt0.o /usr/lib/cmplrs/cxx/_main.o \
		   -o $@ $^ -lcxxstd -lcxx -lexc -lots -lc
else
ifeq ($(ARCH),macosx)
# We need to make both the .dylib and the .so
		$(LD) $(SOFLAGS) $^ -lstdc++ $(OutPutOpt) $@
		$(LD) -bundle -undefined suppress -Wl,-x $(LDFLAGS) $^ \
		   $(OutPutOpt) $@
else
ifeq ($(PLATFORM),win32)
		$(LD) $(SOFLAGS) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
else
		$(LD) $(SOFLAGS) $(LDFLAGS) $^ $(OutPutOpt) $@
endif
endif
endif
endif
endif

#	$(CXX) -g -o driver $(CXXFLAGS) main.cc mydict.cc $(LIBS) 

infodriver: tmain.cc mydict.cc RootCaloHit.h  RootData.h  RootPCellID.h $(ROOTSYS)/include/Rtypes.h
	$(CXX) -g -o infodriver $(CXXFLAGS) tmain.cc mydict.cc $(LIBS) 

mydict.cc:  LinkDef.h RootData.h RootCaloHit.h  RootPCellID.h $(ROOTSYS)/bin/rootcint
	rootcint -f mydict.cc -c RootData.h RootCaloHit.h LinkDef.h

libIotest.so: main.cc mydict.cc RootCaloHit.h  RootData.h  RootPCellID.h $(ROOTSYS)/include/Rtypes.h
	$(CXX) -g -shared -o libIotest.so -DSHARED $(CXXFLAGS) main.cc mydict.cc $(LIBS) 

.cc.$(ObjSuf):
	$(CXX) $(CXXFLAGS) -c $<
