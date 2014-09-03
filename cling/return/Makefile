# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES)

#Set the list of target to make while testing
TEST_TARGETS += constRetrunTest barna exitcode

# adjust the location of Rules.mk as needed.
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

constRetrunTest:
ifeq ($(FAIL),)
	$(WarnFailTest)
else
	$(CMDECHO) $(CALLROOTEXE) -q -b -l run.C > run.log
endif

barna.log: barna.C
	$(CMDECHO) $(CALLROOTEXE) -q -b -l barna.C > barna.log 2>&1

barna: barna.log
	$(TestDiffW)

# Test the exit code of running a script with return value
exitcode.log: exitcode.C
	$(CMDECHO) $(CALLROOTEXE) -l -b -q $^ > $@ 2>& 1; EXITCODE="$$?"; if test "$$EXITCODE" -ne "12"; then echo "$@: unexpected exit code $$EXITCODE" >& 2; fi; test "$$EXITCODE" -eq "12"

# Test the printout of running a script with return value
exitcode: exitcode.log
	$(TestDiffW)
