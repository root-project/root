# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += filltree.output arrayDrawTest.output myTree.root $(ALL_LIBRARIES)

#Set the list of target to make while testing
TEST_TARGETS += arrayDrawTest

# adjust the location of Rules.mk as needed.
include ../../Rules.mk

ifneq ($(FAIL),)
   #force re-execution in the fail case
	forceExec := $(shell rm arrayDrawTest.output)
endif
   

myTree.root: runFillTree.C fillTree.cxx Data.h Data.cxx 
	$(CMDECHO) root.exe -q -b -l runFillTree.C > filltree.output

arrayDrawTest.output: runArrayTest.C myTree.root
	$(CMDECHO) root.exe -q -b -l runArrayTest.C > arrayDrawTest.output 2>&1 

arrayDrawTest: arrayDrawTest.output
ifeq ($(FAIL),)
	$(WarnFailTest)
endif
	$(CMDECHO) echo arrayDrawTest > /dev/null
	$(CMDECHO) diff -b arrayDrawTest.good arrayDrawTest.output

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


