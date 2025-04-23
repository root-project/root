CLEAN_TARGETS += *.log dummy* *pcm *o *rflx.* $(ALL_LIBRARIES) inst.root instlib* inst2lib*
TEST_TARGETS += execLoadLibs runInvalidDeclRecovery.success
#runInvalidDeclRecoveryCI.success

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

instlib.cxx:
	$(CMDECHO) rootcling -f instlib.cxx -s instlib.so -rml instlib.so -rmf instlib.rootmap instGood.cxx linkdef.h

# Custom rule required here, as we must make *sure* that there is no rootmap
# file generated/left over.
inst2lib.cxx:
	$(CMDECHO) rootcling -f inst2lib.cxx -s inst2lib.so instGood.cxx secondLinkdef.h

instlib.$(DllSuf): instlib.$(ObjSuf)
	$(BuildFromObj)

inst2lib.$(DllSuf): inst2lib.$(ObjSuf)
	$(BuildFromObj)

execWriteFile.log: instlib.$(DllSuf) inst2lib.$(DllSuf)

execTest.log: execWriteFile.success

runInvalidDeclRecovery.success: runInvalidDeclRecovery.C $(UTILS_PREREQ) $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) $(CALLROOTEXE) -q -l -b $< > runInvalidDeclRecovery.log 2>runInvalidDeclRecovery.ign.elog || handleError.sh --cmd='Execution of $<' --result=$$? --log=runInvalidDeclRecovery.ign.elog --test=runInvalidDeclRecovery.C && touch $@

#runInvalidDeclRecoveryCI.success: runInvalidDeclRecoveryCI.C $(UTILS_PREREQ) $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
#	$(CMDECHO) $(CALLROOTEXE) -q -l -b $< > runInvalidDeclRecoveryCI.log 2>runInvalidDeclRecoveryCI.ign.elog || handleError.sh --cmd='Execution of $<' --result=$$? --log=runInvalidDeclRecoveryCI.ign.elog --test=runInvalidDeclRecoveryCI.C && touch $@

