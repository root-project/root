# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += filltree.output arrayDrawTest.output myTree.root $(ALL_LIBRARIES) *.log

#Set the list of target to make while testing
TEST_TARGETS += arrayDrawTest arrayEventListTest boolOptTest

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

arrayEventListTest.log: eventlist.C
	$(CMDECHO) root.exe -b -l -q eventlist.C > arrayEventListTest.log 2>&1 

arrayEventListTest: arrayEventListTest.log
	$(CMDECHO) diff -b arrayEventListTest.ref arrayEventListTest.log

boolOptTest.log: boolOpt.C
	$(CMDECHO) root.exe -b -l -q boolOpt.C > boolOptTest.log 2>&1 

boolOptTest: boolOptTest.log
	$(CMDECHO) diff -b boolOptTest.ref boolOptTest.log

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


