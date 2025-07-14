CLEAN_TARGETS += main *Dict* Event.root *~  *.log $(ALL_LIBRARIES)
TEST_TARGETS += ptrArr ArraySave

ifeq ($(strip $(ROOTTEST_HOME)),)
   export ROOTTEST_HOME := $(shell git rev-parse --show-toplevel)/roottest/
   ifeq ($(strip $(ROOTTEST_HOME)),)
      export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
   endif
   ifeq ($(strip $(ROOTTEST_HOME)),)
      $(error The head of roottest was not found.  Set ROOTTEST_HOME)
   endif
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


ifneq ($(FAIL),)
ArraySave: RealArraySave
endif

# we expect the test to fail!
ArraySave: 
ifeq ($(FAIL),)
	@echo WARNING `pwd` has failing test, run make RealArraySave to see it.
endif

RealArraySave: RealArraySave.log
	$(CMDECHO) diff -u  RealArraySave.log  RealArraySave.ref

RealArraySave.log: foo_C.$(DllSuf) bar_C.$(DllSuf) main_C.$(DllSuf) $(ROOT_LOC)/lib/libTreePlayer.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l RealArraySave.C > RealArraySave.log

#main: main.C fooDict.C barDict.C foo.C bar.C
#	g++ -g -o $@ $^ -I. `root-config --cflags --libs` -lTreePlayer

#libFoo.so: main.C fooDict.C barDict.C foo.C bar.C
#	g++ -g -shared -o $@ $^ -I. `root-config --cflags --libs`

%Dict.C: %.h %Linkdef.h
	$(CMDECHO) rootcint -f $@ $^


ptrArr: ptrArr.log
	$(TestDiff)
