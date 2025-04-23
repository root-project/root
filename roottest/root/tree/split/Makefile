# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log abstract.root customClassTree.root

#Set the list of target to make while testing
TEST_TARGETS += CollectionSplit

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


CollectionSplit.log: tobj_cxx.$(DllSuf) $(ROOTCORELIBS)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l tobj.cxx+ | grep Br > CollectionSplit.log

CollectionSplit: CollectionSplit.log
	$(CMDECHO) diff -bu CollectionSplit.ref CollectionSplit.log 

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

abstract.root: abstract_cxx.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE)  -b -q -l 'runabstract.cxx(0)' > abstract_root.log

abstract.log: abstract.root
	$(CMDECHO) $(CALLROOTEXE)  -b -q -l 'runabstract.cxx(1)' > abstract.log

abstract: abstract.log
	$(TestDiff)

execReadCustomStream.clog: execWriteCustomStream.success

