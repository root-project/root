# This is a template for all makefiles.

#Set the list of files to be deleted by clean (Targets can also be specified).:
CLEAN_TARGETS += $(ALL_LIBRARIES) branch.root treeobj.root abc*.root abc_rflx.cpp copy.root temp.root blah.root 

# Set the list of target to make while testing.  By default, mytest is the
# only target added.  If the name of the target is changed in the rules then
# the name should be changed accordingly in this list.

TEST_TARGETS += noname simple i_simple separate abc updates

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
include $(ROOTTEST_HOME)/scripts/Reflex.mk


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



noname: noname.log
	$(TestDiff)

simple: simple.clog
	$(TestDiff)

i_simple: simple.log
	$(TestDiff)

separate: separate.log
	$(TestDiff)

abc_rflx.cpp: abc.h
	$(CMDECHO) genreflex abc.h > abc_reflex_cpp.log 2>&1

libabc_rflx_C.$(DllSuf): abc_rflx.cpp


ifneq ($(ClingWorkAroundMissingDynamicScope),)
abc_cint.root: CALLROOTEXE += -e 'gROOT->ProcessLine(".L abc.h+");' 
endif

abc_cint.root: abc_h.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -l -q abcwrite.C\(\"cint\"\) > abc_cint.log 2>&1

abc_reflex.root: libabc_rflx_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -l -q abcwrite.C\(\"reflex\"\) > abc_reflex.log 2>&1

ifneq ($(ClingWorkAroundMissingDynamicScope),)
abc.cintcint.log: CALLROOTEXE += -e 'gROOT->ProcessLine(".L abc.h+");' 
endif

abc.cintcint.log: abc_cint.root abc_h.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -l -q abcread.C\(\"cint\",\"cint\"\) 2>&1 | grep -v Processing > abc.cintcint.log 

abc.cintreflex.log: abc_cint.root libabc_rflx_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -l -q abcread.C\(\"cint\",\"reflex\"\) 2>&1 | grep -v Processing > abc.cintreflex.log 

abc.reflexcint.log: abc_reflex.root abc_h.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -l -q abcread.C\(\"reflex\",\"cint\"\) 2>&1 | grep -v Processing > abc.reflexcint.log 

abc.reflexreflex.log: abc_reflex.root libabc_rflx_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -l -q abcread.C\(\"reflex\",\"reflex\"\) 2>&1 | grep -v Processing > abc.reflexreflex.log 


abc.cintcint.success: abc.cintcint.log
	$(CMDECHO) diff -b -u abc.ref abc.cintcint.log && touch abc.cintcint.success

abc.reflexcint.success: abc.reflexcint.log
	$(CMDECHO) diff -b -u abc.ref abc.reflexcint.log && touch abc.reflexcint.success

abc.cintreflex.success: abc.cintreflex.log
	$(CMDECHO) diff -b -u abc.ref abc.cintreflex.log && touch abc.cintreflex.success

abc.reflexreflex.success: abc.reflexreflex.log
	$(CMDECHO) diff -b -u abc.ref abc.reflexreflex.log && touch abc.reflexreflex.success

ifeq ($(findstring gccxml,$(notdir $(HAVEGCCXML))),gccxml)
abc: abc.cintcint.success abc.cintreflex.success abc.reflexcint.success abc.reflexreflex.success
else
abc: abc.cintcint.success
endif

updates: updates.log
	$(TestDiff)

ifneq ($(ClingWorkAroundMissingAutoLoading),)
updates.log: CALLROOTEXE += -e 'gSystem->Load("libTreePlayer");'
endif

