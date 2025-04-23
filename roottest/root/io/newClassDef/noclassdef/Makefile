SHELL=/bin/sh
TEST_TARGETS += noclassdef mydict-diff
CLEAN_TARGETS += *.o *.so ti_files ii_files mydict.* driver$(ExeSuf) *~ \
	*.obj .def *.root *.mode driver.output $(ALL_LIBRARIES)

ifeq ($(strip $(ROOTTEST_HOME)),)
   export ROOTTEST_HOME := $(shell git rev-parse --show-toplevel)/
   ifeq ($(strip $(ROOTTEST_HOME)),)
      export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
   endif
   ifeq ($(strip $(ROOTTEST_HOME)),)
      $(error The head of roottest was not found.  Set ROOTTEST_HOME)
   endif
endif
include $(ROOTTEST_HOME)/scripts/Rules.mk

local_all: test driver$(ExeSuf) libIotest.$(DllSuf)
.PHONY: main

#include $(ROOTSYS)/test/Makefile.arch

#arch := $(shell grep arch `which root-config` | head -1 | cut -d'=' -f2-)
arch := $(shell root-config --arch)
CXXFLAGS :=  $(patsubst -O,-g,$(CXXFLAGS))

ifeq ($(findstring linuxicc,$(arch)),linuxicc)
CXXFLAGS += -wd191
endif

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

ifeq ($(PLATFORM),win32)
CXXFLAGS += -TP
LIBSFORLINK = $(LIBS)
CLDFLAGS := -link $(LDFLAGS) -debug
SYSLIBS += msvcprt.lib 
else
LIBSFORLINK = 
CLDFLAGS := $(LDFLAGS)
endif

ifeq ($(FAIL),)
#Make sure the failing code is excluded
exec1 := $(shell if test -e fail.mode ; then rm -f fail.mode *.o mydict.*; fi )
exec2 := $(shell if test ! -e pass.mode ; then touch pass.mode; fi 1>&2 )
MODE_FILE = pass.mode
CFLAGS += -DSKIP_FAILURE
else
#Make sure the failing code is included
exec1 := $(shell if test -e pass.mode; then rm -f pass.mode *.o mydict.*; fi )
exec2 := $(shell if test ! -e fail.mode ; then touch fail.mode; fi 1>&2 )
MODE_FILE = fail.mode
CFLAGS += -DINC_FAILURE
endif

test2:
	echo $(arch)

shared: libIotest.$(DllSuf)

noclassdef:   driver$(ExeSuf)
	$(CMDECHO)rm -f driver.output
	$(CMDECHO)./driver$(ExeSuf) 2>&1 | sed -e 's/Dumping object at: 0x[0-9a-f]*,/Dumping object,/' -e 's/TStreamerInfo:://' | sed -e 's/type .*RootPCnodict/type class RootPCnodict/' > driver.output 
	$(CMDECHO)diff -bw good.linux.$(ROOTBITS).output driver.output
#	@echo "All test succeeded in `pwd`"

otest:   driver$(ExeSuf)
	$(CMDECHO)rm -f odriver.output
	$(CMDECHO)./driver$(ExeSuf) -readold 2>&1 | grep -v -- '->[^0][^ ]' | sed -e 's/Dumping object at: 0x[0-9a-f]*,/Dumping object,/' > odriver.output 
	$(CMDECHO)diff -bw good.linux.output odriver.output
	$(CMDECHO)echo "Testing reading old succeeded in `pwd`"



#driver: main.cc mydict.cc RootCaloHit.h  RootData.h  RootPCellID.h $(ROOT_LOC)/include/Rtypes.h


driver$(ExeSuf): main.cc mydict.cc 
	$(CMDECHO)$(CXX)  $^ $(CFLAGS) $(CXXFLAGS) $(CLDFLAGS) $(ROOTLIBS) $(OutPutOpt)$@

stuff:
ifeq ($(ARCH),aix)
		$(CMDECHO)/usr/ibmcxx/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
else
ifeq ($(ARCH),aix5)
		$(CMDECHO)/usr/vacpp/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
else
ifeq ($(ARCH),alphacxx)
# due to a bug in cxx/ld under osf3.xx, one cannot use cxx to generate
# a shared library. One must use ld instead.
		$(CMDECHO)ld -L/usr/lib/cmplrs/cxx -rpath /usr/lib/cmplrs/cxx \
		   -expect_unresolved "*" -g0 -O1 -shared \
		   /usr/lib/cmplrs/cc/crt0.o /usr/lib/cmplrs/cxx/_main.o \
		   -o $@ $^ -lcxxstd -lcxx -lexc -lots -lc
else
ifeq ($(ARCH),macosx)
# We need to make both the .dylib and the .so
		$(CMDECHO)$(LD) $(SOFLAGS) $^ -lstdc++ $(OutPutOpt) $@
		$(CMDECHO)$(LD) -bundle -undefined suppress -Wl,-x $(LDFLAGS) $^ \
		   $(OutPutOpt) $@
else
ifeq ($(PLATFORM),win32)
		$(CMDECHO)$(LD) $(SOFLAGS) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
else
		$(CMDECHO)$(LD) $(SOFLAGS) $(LDFLAGS) $^ $(OutPutOpt) $@
endif
endif
endif
endif
endif

#	$(CXX) -g -o driver $(CXXFLAGS) main.cc mydict.cc $(LIBS) 

infodriver: tmain.cc mydict.cc RootCaloHit.h  RootData.h  RootPCellID.h $(ROOT_LOC)/include/Rtypes.h
	$(CMDECHO)$(CXX) -g -o infodriver $(CFLAGS) $(CXXFLAGS) tmain.cc mydict.cc $(LIBS) 

mydict.cc:  LinkDef.h RootData.h RootCaloHit.h  RootPCellID.h $(ROOTCINT)
	$(CMDECHO)rootcint -f mydict.cc $(CFLAGS) RootData.h RootCaloHit.h LinkDef.h > mydict.log 2>&1

mydict-diff: mydict.cc
	$(CMDECHO)diff mydict-gen.ref mydict.log

ifeq ($(PLATFORM),win32)
libIotest.dll: main.$(ObjSuf) mydict.$(ObjSuf)
	$(BuildFromObjs)

#	$(CMDECHO)bindexplib $* $^ > libIotest.def
#	$(CMDECHO)lib -nologo -MACHINE:IX86 $^ -def:libIotest.def \
#                  $(OutPutOpt)libIotest.lib
#	$(CMDECHO)$(LD) $(SOFLAGS) $(LDFLAGS) $^ libIotest.exp $(LIBS) \
#                   $(OutPutOpt)$@
else
libIotest.so: main.cc mydict.cc RootCaloHit.h  RootData.h  RootPCellID.h $(ROOT_LOC)/include/Rtypes.h
	$(CMDECHO)$(CXX) -g -shared -o libIotest.so -DSHARED $(CFLAGS) $(CXXFLAGS) main.cc mydict.cc $(LIBS) 
endif

.cc.$(ObjSuf):
	$(CMDECHO)$(CXX) $(CFLAGS) $(CXXFLAGS) -c $<
