# This is a template for all makefiles.

#Set the list of files to be deleted by clean (Targets can also be specified).:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log

# Set the list of target to make while testing.  By default, mytest is the
# only target added.  If the name of the target is changed in the rules then
# the name should be changed accordingly in this list.

TEST_TARGETS += mytest

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
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


# Uncomment the following line to have gmake called recursively on subdirectories which contain a file named "Makefile".
#SUBDIRS = $(shell $(ROOTTEST_HOME)scripts/subdirectories .)

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

clonesfile.root: $(ROOTCORELIBS) MyClassClones_cxx.$(DllSuf) WriteClones.C
	$(CMDECHO) root.exe -q -b -l WriteClones.C > WriteClones.log
	
vectorfile.root: $(ROOTCORELIBS) MyClassVector_cxx.$(DllSuf) WriteVector.C
	$(CMDECHO) root.exe -q -b -l WriteVector.C > WriteVector.log
	
listfile.root: $(ROOTCORELIBS) MyClassList_cxx.$(DllSuf) WriteList.C
	$(CMDECHO) root.exe -q -b -l WriteList.C > WriteList.log
	
ReadClones.log: $(ROOTCORELIBS) MyClassClones_cxx.$(DllSuf) ReadClones.C clonesfile.root
	$(CMDECHO) root.exe -q -b -l ReadClones.C > ReadClones.log
	
ReadVector.log:  $(ROOTCORELIBS) MyClassVector_cxx.$(DllSuf) ReadVector.C clonesfile.root vectorfile.root listfile.root
	$(CMDECHO) root.exe -q -b -l ReadVector.C > ReadVector.log

ReadList.log:  $(ROOTCORELIBS) MyClassList_cxx.$(DllSuf) ReadList.C clonesfile.root vectorfile.root listfile.root
	$(CMDECHO) root.exe -q -b -l ReadList.C > ReadList.log

ReadNolib.log:  $(ROOTCORELIBS) ReadNolib.C clonesfile.root
	$(CMDECHO) root.exe -q -b -l ReadNolib.C > ReadNolib.log 2>&1

containerEvolution.log: $(ROOTCORELIBS) MyClassClones_cxx.$(DllSuf) MyClassVector_cxx.$(DllSuf) MyClassList_cxx.$(DllSuf)

containerEvolution: containerEvolution.log ReadClones.log ReadVector.log ReadList.log ReadNolib.log
	$(CMDECHO) diff -b -w ReadClones.ref ReadClones.log
	$(CMDECHO) diff -b -w ReadVector.ref ReadVector.log
	$(CMDECHO) diff -b -w ReadList.ref ReadList.log
	$(CMDECHO) diff -b -w ReadNolib.ref ReadNolib.log
	$(TestDiff)
	
SimpleOne.root: SimpleOne_C.$(DllSuf) SimpleWrite.C $(ROOTCORELIBS)
	$(CMDECHO) root.exe -q -b -l 'SimpleWrite.C(1)' > SimpleWriteOne.log
	
SimpleTwo.root: SimpleTwo_C.$(DllSuf) SimpleWrite.C $(ROOTCORELIBS)
	$(CMDECHO) root.exe -q -b -l 'SimpleWrite.C(2)' > SimpleWriteTwo.log
	
SimpleRead.log: SimpleOne.root SimpleTwo.root SimpleRead.C $(ROOTCORELIBS)
	$(CMDECHO) root.exe -q -b -l SimpleRead.C > SimpleRead.log 2>&1

SimpleTest: SimpleRead.log 
	$(CMDECHO) diff -b -w SimpleRead.ref SimpleRead.log

