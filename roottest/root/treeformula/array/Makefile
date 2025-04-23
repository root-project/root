# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += filltree.output arrayDrawTest.output myTree.root $(ALL_LIBRARIES) *.log  NtpRecord.root *.clog twodim*.root

#Set the list of target to make while testing
TEST_TARGETS += vectorInVector indexLoad vectorOfvector ScanString multidim twodimvar arrayDrawTest arrayEventListTest boolOptTest indexMultiDim nestedind

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


ifneq ($(FAIL),)
   #force re-execution in the fail case
	forceExec := $(shell rm -f arrayDrawTest.output)
endif

myTree.root: runFillTree.C fillTree.cxx Data.h Data.cxx 
	$(CMDECHO) $(CALLROOTEXE) -q -b -l runFillTree.C > filltree.output 2>&1

arrayDrawTest.output: runArrayTest.C myTree.root
	$(CMDECHO) $(CALLROOTEXE) -q -b -l runArrayTest.C > arrayDrawTest.output 2>&1 

arrayDrawTest: arrayDrawTest.output
ifeq ($(FAIL),)
	$(WarnFailTest)
endif
	$(CMDECHO) echo arrayDrawTest > /dev/null
	$(CMDECHO) diff -bw arrayDrawTest.good arrayDrawTest.output

arrayEventListTest.log: eventlist.C
	$(CMDECHO) $(CALLROOTEXE) -b -l -q eventlist.C > arrayEventListTest.log 2>&1 

arrayEventListTest: arrayEventListTest.log
	$(CMDECHO) diff -b arrayEventListTest.ref arrayEventListTest.log

boolOptTest.log: boolOpt.C
	$(CMDECHO) $(CALLROOTEXE) -b -l -q boolOpt.C > boolOptTest.log 2>&1 

boolOptTest: boolOptTest.log
	$(CMDECHO) diff -bw boolOptTest.ref boolOptTest.log

indexMultiDim.log: indexMultiDim.C orange.root
	$(CMDECHO) $(CALLROOTEXE) -b -l -q indexMultiDim.C > indexMultiDim.log 2>&1 || \
		diff -b indexMultiDim.ref indexMultiDim.log

indexMultiDim: indexMultiDim.log
	$(TestDiffW)

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

twodimvar.log: twodimvar_C.$(DllSuf)

twodimvar: twodimvar.log
	$(TestDiff)

multidim.log: NtpLib_C.$(DllSuf)

multidim: multidim.log
	$(TestDiff)

ScanString.log: ScanString_C.$(DllSuf)

ScanString: ScanString.log
	$(TestDiff)

vectorOfvector: vectorOfvector.clog
ifeq ($(ClingWorkAroundErracticValuePrinter),)	
	$(TestDiff)
else
	$(TestDiffCintSpecific)
endif

indexLoad: indexLoad.clog
	$(TestDiff)

ifeq ($(ARCH)$(FAIL),win32)
vectorInVector:
	$(WarnFailTest)
else
vectorInVector: vectorInVector.log
	$(TestDiff)
endif

nestedind: nestedind.log
	$(TestDiff)
