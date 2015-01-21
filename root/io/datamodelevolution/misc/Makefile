# This is a template for all makefiles.

#Set the list of files to be deleted by clean (Targets can also be specified).:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog treeTest.root arr_cint* arr_rflx*

# Set the list of target to make while testing.  By default, mytest is the
# only target added.  If the name of the target is changed in the rules then
# the name should be changed accordingly in this list.

TEST_TARGETS += postobj arrayRules

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
   export ROOTTEST_HOME := $(shell git rev-parse --show-toplevel)/
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

ifneq ($(ClingWorkAroundMissingDynamicScope),)
treeTest.root: CALLROOTEXE += -e 'gROOT->ProcessLine(".L SomeClass.cxx+");'
endif

treeTest.root: writeTree.C MyClass_cxx.$(DllSuf) SomeClass_cxx.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l writeTree.C > treeTree_root.log 2>&1

ifneq ($(ClingWorkAroundMissingDynamicScope),)
postobj.log: CALLROOTEXE += -e 'gROOT->ProcessLine(".L MyClass.cxx+");'
endif

postobj.log: readTree.C treeTest.root MyClass_cxx.$(DllSuf) SomeClass_cxx.$(DllSuf)

postobj: postobj.log
	$(TestDiff)

execReadOrderClass.clog: execWriteOrderClass.success

arr_cint.cpp: arrayRules.h arrayRulesLinkDef.h $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) rootcint -f arr_cint.cpp -c arrayRules.h arrayRulesLinkDef.h > arr_cint.log 2>&1 || handleError.sh --cmd='rootcint -f arr_cint.cxx -c arrayRules.h ArrayRulesLinkDef.h' --result=$$? --log=arr_cint.log --test=$@  

arr_rflx.cpp: arrayRules.h arrayRulesSelection.xml $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) genreflex arrayRules.h -o arr_rflx.cpp -s arrayRulesSelection.xml > arr_rflx.log 2>&1 || handleError.sh --cmd='genreflex arrayRules.h -o arr_rflx.cxx -s arrayRulesSelection.xml' --result=$$? --log=arr_rflx.log --test=$@ 

arrayRules: arr_cint.$(ObjSuf) arr_rflx.$(ObjSuf)
