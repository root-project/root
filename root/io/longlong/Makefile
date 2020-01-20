# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += longlongdict.*

#Set the list of target to make while testing
TEST_TARGETS += longlongtest

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

longlongdict.C: Class.h $(ROOTCINT)
	$(CMDECHO)rootcint -f longlongdict.C Class.h linkdef.h

longlongtest: longlongdict.C
	$(CMDECHO) test ! `(cat longlongdict.C | grep -v G__letLonglong | grep "long long"  > /dev/null) && echo $?`

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


