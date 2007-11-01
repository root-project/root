# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += *dict* *.so $(ALL_LIBRARIES) *.log

#Set the list of target to make while testing
TEST_TARGETS += defaultArgTest

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
