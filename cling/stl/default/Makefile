# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += *dict* *.so $(ALL_LIBRARIES) *.log

#Set the list of target to make while testing
TEST_TARGETS += defaultArgTest VectorSort

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

defaultArgTest: dict.cc mylist_C.so
#	$(CMDECHO)echo Running 'mytest' in `pwd`
	$(CMDECHO) echo mytest > /dev/null

dict.cc: mylist.C $(ROOTCINT)
	$(CMDECHO) rootcint -f dict.cc mylist.C LinkDef.h

mylist_C.so:
	$(CMDECHO) $(CALLROOTEXE) -q -l -b build.C > build.log 2>&1

VectorSort.log.tmp: VectorSort.C
	$(CMDECHO) LINES=0 $(CALLROOTEXE) -q -l -b VectorSort.C > $@ 2>&1

VectorSort.log: VectorSort.log.tmp
ifeq ($(ClingWorkAroundValuePrinterNotFullyQualified),)
	$(CMDECHO) $(RemoveLeadingDirs) -e 's,^(vector,(class vector,' -e 's,iterator) @0[x[:xdigit:]]*$$,iterator),' -e 's/::__1//g' $^ > $@ 2>&1
else
	$(CMDECHO) $(RemoveLeadingDirs) -e 's,^(vector,(class vector,' -e 's,::iterator)[x[:xdigit:]]*$$,::iterator),' -e 's/::__1//g' -e 's, @0x[a-fA-F0-9]*,,' $^ > $@ 2>&1
endif

VectorSort: VectorSort.log
	$(TestDiffCintSpecific)

