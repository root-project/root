# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log 

#Set the list of target to make while testing
TEST_TARGETS += CollectionSplit

# adjust the location of Rules.mk as needed.
ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


CollectionSplit.log: tobj_cxx.$(DllSuf)
	$(CMDECHO) root.exe -q -b -l tobj.cxx+ | grep Br > CollectionSplit.log

CollectionSplit: CollectionSplit.log
	$(CMDECHO) diff -b CollectionSplit.ref CollectionSplit.log 

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


