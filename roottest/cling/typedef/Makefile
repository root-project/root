#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log ptrptr *_timestamp

#Set the list of target to make while testing
TEST_TARGETS += ptrptr_test typedefAndScope_test

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


ptrptr_test : ptrptr

ptrptr : ptrptr.log
	$(TestDiffCintSpecificW) && touch $@

ptrptr.log : runptrptr.C ptrptr.C vclass_h.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l runptrptr.C | grep ptrptr.C | $(RemoveLeadingDirs) > ptrptr.log 


typedefAndScope_test : typedefAndScope_timestamp

typedefAndScope_timestamp : typedefAndScope.log
	$(CMDECHO) diff -b typedefAndScope.ref typedefAndScope.log && touch $@

typedefAndScope.log : testclass.h testclass_h.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l -e 'gSystem->Load("testclass_h");' > typedefAndScope.log 2>&1

