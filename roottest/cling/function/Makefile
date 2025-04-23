# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log

#Set the list of target to make while testing
TEST_TARGETS += defaultParams longArgTest overloadResolution staticfunc 
#privateConstructor


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

ifeq ($(FAIL),)
longArgTest:
	$(WarnFailTest)
else
longArgTest: longArgTest.log
	$(TestDiff)
endif

overloadResolution.log: TheClass_h.$(DllSuf)

overloadResolution: overloadResolution.log
	$(TestDiff)

defaultParams.log: Params_h.$(DllSuf)

# This will skip the test (if there is a known failure)
ifeq ($(FAIL),)
defaultParams:
	$(WarnFailTest)
else
defaultParams: defaultParams.log
	$(TestDiff)
endif

ifneq ($(ClingWorkAroundMissingAutoLoading),)
staticfunc.log: CALLROOTEXE += -e 'gSystem->Load("libPhysics");'
endif

staticfunc.log: staticfunc.C runstaticfunc.C staticfunc.ref
	$(CMDECHO) $(CALLROOTEXE) -b -q -l runstaticfunc.C > staticfunc.raw.log 2>&1 ; export result=$$? ; cat staticfunc.raw.log | grep -v input_line | $(RemoveDirs) -e 's?[(:][0-9][0-9][):]?:--:?'  > staticfunc.log ; \
	    test $$result -eq 0 || handleError.sh --result=$$result --log=staticfunc.log --test=$@

staticfunc: staticfunc.log
	$(TestDiffCintSpecific)

privateConstructor: privateConstructor.log
	$(TestDiff)

execrefDefaultParams.log: refFunction_C.$(DllSuf) refClasses_cxx.$(DllSuf)
ifneq ($(ClingWorkAroundMissingDynamicScope)$(ClingWorkAroundMissingUnloading),)
	$(CMDECHO)cp execrefDefaultParams.ref execrefDefaultParams.log
endif

