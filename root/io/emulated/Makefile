# This is a template for all makefiles.

#Set the list of files to be deleted by clean (Targets can also be specified).:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog inherit*.root

# Set the list of target to make while testing.  By default, mytest is the
# only target added.  If the name of the target is changed in the rules then
# the name should be changed accordingly in this list.

TEST_TARGETS += read rewrite readAbstract

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
	export ROOTTEST_HOME:=$(shell expr $(PWD) : '\(.*/roottest/\)')
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

ifneq ($(ClingWorkAroundMissingSmartInclude),)
CALLROOTEXE_INH_EXTRA = -e 'gSystem->Load("classes_h");'
endif

ifneq ($(ClingWorkAroundMissingSmartInclude),)
CALLROOTEXE_READ_EXTRA = -e 'gSystem->Load("marker_h");'
endif

ifneq ($(ClingWorkAroundMissingAutoLoading),)
rewrite.log read.log inherit.root: CALLROOTEXE += -e 'gSystem->Load("libTreePlayer");'
endif

abstract.root: writeAbstract_cxx.$(DllSuf) abstract.h
	$(CMDECHO) $(CALLROOTEXE) -q -b -l writeAbstract.cxx+ > writeAbstract.log 2>&1 

inherit.root: write.C marker_h.$(DllSuf) classes_h.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) $(CALLROOTEXE_INH_EXTRA) -q -b -l write.C > write.log 2>&1 

read.log: read.C marker_h.$(DllSuf) inherit.root
	$(CMDECHO) $(CALLROOTEXE) $(CALLROOTEXE_READ_EXTRA) -q -b -l read.C > read.log 2>&1 

rewrite.log: rewrite.C marker_h.$(DllSuf) inherit.root
	$(CMDECHO) $(CALLROOTEXE) $(CALLROOTEXE_READ_EXTRA) -q -b -l rewrite.C > rewrite.log 2>&1 

readAbstract.log: readAbstract_cxx.$(DllSuf) abstract.root
	$(CMDECHO) $(CALLROOTEXE) -q -b -l readAbstract.cxx+ > readAbstract.log 2>&1 

read: read.log
	$(TestDiff)

rewrite: rewrite.log
	$(TestDiff)

readAbstract: readAbstract.log
	$(TestDiff)

