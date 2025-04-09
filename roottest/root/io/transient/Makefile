# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log  missingdict_rflx.cpp AtomTransient.root

#Set the list of target to make while testing
TEST_TARGETS += transientTest missingdict

# adjust the location of Rules.mk as needed.
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
include $(ROOTTEST_HOME)/scripts/Reflex.mk


.PRECIOUS: missingdict_rflx_cpp.$(DllSuf) missingdict_rflx.cpp

transientTest.log: TBigDSWriteEvent_cxx.$(DllSuf)  TBigDSWriteParticle_cxx.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -l -b run.C 2>&1 > transientTest.log

ifeq ($(FAIL),)
transientTest: 
# Having a data member whose ClassDef version is zero leads to a questionable warning message
	$(WarnFailTest)
else
transientTest: transientTest.log
	$(CMDECHO) diff -b  transientTest.ref  transientTest.log
endif

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

execNew.clog: execOld.clog

ifeq ($(findstring gccxml,$(notdir $(HAVEGCCXML))),gccxml)
missingdict.log: missingdict_rflx_cpp.$(DllSuf)

missingdict: missingdict.log
	$(TestDiff)
else
missingdict:
endif

execTransientNoctor.log: libtransientNoctor_dictrflx.$(DllSuf)

