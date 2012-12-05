# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += *dict* *.so $(ALL_LIBRARIES) *.log

#Set the list of target to make while testing
TEST_TARGETS += defaultArgTest VectorSort

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(PWD) : '\(.*/roottest/\)')
endif
include $(ROOTTEST_HOME)/scripts/Rules.mk

defaultArgTest: dict.cc mylist_C.so
#	$(CMDECHO)echo Running 'mytest' in `pwd`
	$(CMDECHO) echo mytest > /dev/null

dict.cc: mylist.C $(ROOTCINT)
	$(CMDECHO) rootcint -f dict.cc -c mylist.C LinkDef.h

mylist_C.so:
	$(CMDECHO) $(CALLROOTEXE) -q -l -b build.C > build.log 2>&1

VectorSort.log: VectorSort.C
ifeq ($(ClingWorkAroundValuePrinterNotFullyQualified),)
	$(CMDECHO) LINES=0 $(CALLROOTEXE) -q -l -b $(@:.log=.C) \
	  | $(RemoveLeadingDirs) -e 's,^(vector,(class vector,' -e 's,::iterator)[x[:xdigit:]]*$$,::iterator),' \
	  > $@ 2>&1
else
	$(CMDECHO) LINES=0 $(CALLROOTEXE) -q -l -b $(@:.log=.C) \
	  | $(RemoveLeadingDirs) -e 's,^(vector,(class vector,' -e 's,::iterator)[x[:xdigit:]]*$$,::iterator),' -e 's/const_iterator/class vector<unsigned int::const_iterator/' -e 's, @0x[a-fA-F0-9]*,,' \
	  > $@ 2>&1
endif

VectorSort: VectorSort.log
	$(TestDiffCintSpecific)
