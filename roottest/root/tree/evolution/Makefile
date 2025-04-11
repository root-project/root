# This is a template for all makefiles.

#Set the list of files to be deleted by clean (Targets can also be specified).:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log clonesfile.root cloning2.root cloning.root listfile.root \
  missingArray.root missingCollection.root SimpleOne.root SimpleTwo.root vectorfile.root

# Set the list of target to make while testing.  By default, mytest is the
# only target added.  If the name of the target is changed in the rules then
# the name should be changed accordingly in this list.

TEST_TARGETS += cloning mytest SimpleTest containerEvolution missingArray

# Search for Rules.mk in roottest/scripts
# Algorithm:  Find the current working directory and remove everything after
#  '*roottest/'.  Append the path for Rules.mk from within roottest, which 
#  should be 'scripts/Rules.mk'.  The roottest path is saved in the
#  ROOTTEST_HOME variable for use by the SUBDIRECTORIES variable and is 
#  exported to eliminate the overhead of findding the path again during 
#  recursive calls of gmake.
# Since all makefiles should be under roottest or one of its
#  subdirectories and all recursions of gmake are called by 
#    'cd [DIR]; gmake ...'
#  this algorithm should not fail in finding /roottest/ in the
#  current working directory.  
# Issues:  This algorithm will fail if a makefile is called from outside the 
#  roottest folder, as in executing 'gmake -f ~/roottest/Makefile' from 
#  the home directory.

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

# The name of this target should be changed according to the test being run.  Any name changes
#  should also be reflected in the TEST_TARGETS variable above.

mytest:
	$(CMDECHO) echo mytest > /dev/null


# Outputs a message if the FAIL variable is null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


# Remember that sometest.log can be made automatically from
#  from runsometest.C
# This target attempts to create a log file (which should be the output of running the test), and
#  compares its output with a prebuilt reference file.  The reference file should be copied from
#  the log files produced during what is considered a successful run of the test.  The testWithDiff
#  files should be renamed according to the test being run.
# By default the TestDiff variable is set to compare the .log and .ref files
# To run a diff and ignore whitespacing use $(TestDiffW)

testWithDiff: testWithDiff.log testWithDiff.ref
	$(TestDiff)

MyClassClones_cxx.$(DllSuf): MyClass.h MyClassClones.cxx

MyClassVector_cxx.$(DllSuf): MyClass.h MyClassVector.cxx

MyClassList_cxx.$(DllSuf) : MyClass.h MyClassVector.cxx

ifneq ($(ClingWorkAroundMissingDynamicScope),)
clonesfile.root: CALLROOTEXE += -e 'gROOT->ProcessLine(".L MyClassClones.cxx+");'
endif

clonesfile.root: $(ROOTCORELIBS) MyClassClones_cxx.$(DllSuf) WriteClones.C
	$(CMDECHO) $(CALLROOTEXE) -q -b -l WriteClones.C > WriteClones.log 2>&1 || handleError.sh --result=$$? --log=WriteClones.log

ifneq ($(ClingWorkAroundMissingDynamicScope),)
vectorfile.root: CALLROOTEXE += -e 'gROOT->ProcessLine(".L MyClassVector.cxx+");'
endif
 
vectorfile.root: $(ROOTCORELIBS) MyClassVector_cxx.$(DllSuf) WriteVector.C
	$(CMDECHO) $(CALLROOTEXE) -q -b -l WriteVector.C > WriteVector.log 2>&1  || handleError.sh --result=$$? --log=WriteVector.log

ifneq ($(ClingWorkAroundMissingDynamicScope),)
listfile.root: CALLROOTEXE += -e 'gROOT->ProcessLine(".L MyClassList.cxx+");'
endif

listfile.root: $(ROOTCORELIBS) MyClassList_cxx.$(DllSuf) WriteList.C
	$(CMDECHO) $(CALLROOTEXE) -q -b -l WriteList.C > WriteList.log 2>&1  || handleError.sh --result=$$? --log=WriteList.log

ReadClones.log: $(ROOTCORELIBS) MyClassClones_cxx.$(DllSuf) ReadClones.C clonesfile.root  vectorfile.root listfile.root
	$(CMDECHO) $(CALLROOTEXE) -q -b -l ReadClones.C > ReadClones.log 2>&1 || handleError.sh --result=$$? --log=ReadClones.log

ReadVector.log:  $(ROOTCORELIBS) MyClassVector_cxx.$(DllSuf) ReadVector.C clonesfile.root vectorfile.root listfile.root
	$(CMDECHO) $(CALLROOTEXE) -q -b -l ReadVector.C > ReadVector.log 2>&1 || handleError.sh --result=$$? --log=ReadVector.log

ReadList.log:  $(ROOTCORELIBS) MyClassList_cxx.$(DllSuf) ReadList.C clonesfile.root vectorfile.root listfile.root
	$(CMDECHO) $(CALLROOTEXE) -q -b -l ReadList.C > ReadList.log 2>&1 || handleError.sh --result=$$? --log=ReadList.log

ReadNolib.log:  $(ROOTCORELIBS) ReadNolib.C clonesfile.root vectorfile.root listfile.root
	$(CMDECHO) $(CALLROOTEXE) -q -b -l ReadNolib.C > ReadNolib.raw.log 2>&1 ; result=$$? ; \
		cat ReadNolib.raw.log | grep -v 'Warning in <TClass::Init>: no dictionary for class' > ReadNolib.log ; \
		test $$result -eq 0 || handleError.sh --result=$$? --log=ReadNoLib.log

containerEvolution.log: $(ROOTCORELIBS) MyClassClones_cxx.$(DllSuf) MyClassVector_cxx.$(DllSuf) MyClassList_cxx.$(DllSuf)

containerEvolution: containerEvolution.log ReadClones ReadVector ReadList ReadNolib
	$(TestDiff)

.PHONY: ReadClones ReadVector ReadList ReadNolib
ReadClones: ReadClones.log
	$(TestDiffW)
ReadVector: ReadVector.log
	$(TestDiffW)
ReadList: ReadList.log
	$(TestDiffW)
ReadNolib: ReadNolib.log
	$(TestDiffW)

SimpleOne_C.$(DllSuf): Simple.h
SimpleTwo_C.$(DllSuf): Simple.h

ifneq ($(ClingWorkAroundMissingDynamicScope),)
SimpleOne.root: CALLROOTEXE += -e 'gROOT->ProcessLine(".L SimpleOne.C+");'

SimpleTwo.root: CALLROOTEXE += -e 'gROOT->ProcessLine(".L SimpleTwo.C+");'

endif

SimpleOne.root: SimpleOne_C.$(DllSuf) SimpleWrite.C $(ROOTCORELIBS)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l 'SimpleWrite.C(1)' > SimpleWriteOne.log

SimpleTwo.root: SimpleTwo_C.$(DllSuf) SimpleWrite.C $(ROOTCORELIBS)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l 'SimpleWrite.C(2)' > SimpleWriteTwo.log

SimpleRead.log: SimpleOne.root SimpleTwo.root SimpleRead.C $(ROOTCORELIBS)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l SimpleRead.C > SimpleRead.log 2>&1

SimpleTest: SimpleRead.log 
	$(CMDECHO) diff -u -b -w SimpleRead.ref SimpleRead.log

ifneq ($(ClingWorkAroundMissingDynamicScope),)
# cloning.root: CALLROOTEXE += -e 'gROOT->ProcessLine(".L cloningOne.C+");'

cloning.log: CALLROOTEXE += -e 'gROOT->ProcessLine(".L cloningTwo.C+");' 
endif

cloning.root: cloningOne_C.$(DllSuf) cloningTwo_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l 'runcloning.C(1)' > cloning_root.log

cloning.log: cloningTwo_C.$(DllSuf) cloning.root

cloning: cloning.log
ifeq ($(FAIL),)
	$(WarnFailTest)
else
	$(TestDiff)
endif

missingArray.root: missingArray1_cc.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l 'missingArray1.cc+' > missingArray.root_root.log

missingArray.clog: missingArray.root missingArray2_cc.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l 'missingArray2.cc+' > missingArray.clog

missingArray: missingArray.clog
	$(TestDiffW)

missingCollection.root: writeMissingCollection_cxx.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l writeMissingCollection.cxx+ > writeMissingCollection.clog 2>&1 || handleError.sh --result=$$? --log=writeMissingCollection.clog

execMissingCollection.clog: missingCollection.root

execMissingCollection.success: EXTRA_DIFFOPTS += -w
