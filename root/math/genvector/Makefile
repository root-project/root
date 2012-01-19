# This is a template for all makefiles.


#Set the list of files to be deleted by clean (Targets can also be specified).:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog 

# Set the list of target to make while testing.  By default, mytest is the
# only target added.  If the name of the target is changed in the rules then
# the name should be changed accordingly in this list.

GENVECTORTESTS := rotationApplication testGenVector coordinates4D coordinates3D
TEST_TARGETS += $(GENVECTORTESTS)

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

##skip tests on Windows 
##ifneq ($(PLATFORM),win32)
##endif

%.cxx: $(ROOT_SRCDIRDEP)/math/genvector/test/%.cxx
	cp $< $@

%.h: $(ROOT_SRCDIRDEP)/math/genvector/test/%.h
	cp $< $@

$(addsuffix _cxx.$(DllSuf),$(GENVECTORTESTS)): CoordinateTraits.h RotationTraits.h

#this suppress check in log file
coordinates3D.elog: coordinates3D_cxx.$(DllSuf)

coordinates3D: coordinates3D.elog
	$(TestDiff)

coordinates4D.elog: coordinates4D_cxx.$(DllSuf)

coordinates4D: coordinates4D.elog
	$(TestDiff)

rotationApplication.elog: rotationApplication_cxx.$(DllSuf)

rotationApplication: rotationApplication.elog
	$(TestDiff)

testGenVector.log: testGenVector_cxx.$(DllSuf)

testGenVector: testGenVector.log
	$(TestDiff)


