#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log ptrptr *_timestamp

#Set the list of target to make while testing
TEST_TARGETS += ptrptr_test typedefAndScope_test

# adjust the location of Rules.mk as needed.
ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


ptrptr_test : ptrptr

ptrptr : ptrptr.log
	$(TestDiffCintSpecific) && touch $@

ptrptr.log : runptrptr.C ptrptr.C vclass_h.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l runptrptr.C | grep ptrptr.C | $(RemoveLeadingDirs) > ptrptr.log 


typedefAndScope_test : typedefAndScope_timestamp

typedefAndScope_timestamp : typedefAndScope.log
	$(CMDECHO) diff -b typedefAndScope.log typedefAndScope.ref && touch $@

typedefAndScope.log : testclass.h testclass_h.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l testclass.h+ > typedefAndScope.log 2> typedefAndScope.build.log

