# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log

#Set the list of target to make while testing
TEST_TARGETS += nestedClones3 nestedClones nestedClones2

# adjust the location of Rules.mk as needed.
ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


nestedClones2.log: digiRun.C
	$(CMDECHO) $(CALLROOTEXE) -q -l -b digiRun.C >nestedClones2.log 2>&1

nestedClones2: nestedClones2.log
	$(CMDECHO) diff -b nestedClones2.log nestedClones2.ref

mytest:
	$(CMDECHO) echo mytest > /dev/null

nestedClones:
ifeq ($(FAIL),)
# one of the reason of the failure is the fact that the first	
# clones array is empty!
	$(WarnFailTest)
else
	$(CMDECHO) $(CALLROOTEXE) -b -q -l runNestedClones.C
endif

nestedClones3: nestedClones3.log
	$(TestDiff)
