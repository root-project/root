# This is a template for all makefiles.

SMATRIXTESTS := testInversion testOperations testKalman testSMatrix 

#Set the list of files to be deleted by clean (Targets can also be specified).:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog $(addsuffix .cxx,$(SMATRIXTESTS)) 

# Set the list of target to make while testing.  By default, mytest is the
# only target added.  If the name of the target is changed in the rules then
# the name should be changed accordingly in this list.

TEST_TARGETS += $(SMATRIXTESTS)

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
	export ROOTTEST_HOME:=$(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


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

TESTDIR:=$(ROOT_SRCDIRDEP)/math/smatrix/test

%.cxx: $(TESTDIR)/%.cxx
	@cp $< $@

%.h: $(TESTDIR)/%.h
	@cp $< $@

$(addsuffix _cxx.$(DllSuf),$(SMATRIXTESTS)): matrix_util.h TestTimer.h matrix_op.h

testWithDiff: testWithDiff.log testWithDiff.ref
	$(TestDiff)

#skip tests on Windows 
testSMatrix.elog: testSMatrix_cxx.$(DllSuf)


testSMatrix: testSMatrix.elog
	$(TestDiff)

testKalman.elog: testKalman_cxx.$(DllSuf)

testKalman: testKalman.elog
	$(TestDiff)

testOperations.elog: testOperations_cxx.$(DllSuf)

testOperations: testOperations.elog
	$(TestDiff)

testInversion.elog: testInversion_cxx.$(DllSuf)

testInversion: testInversion.elog
	$(TestDiff)
