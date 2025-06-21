# This is a template for all makefiles.

#Set the list of files to be deleted by clean (Targets can also be specified).:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log

# Set the list of target to make while testing.  By default, mytest is the only target added.  If
#  the name of the target is changed in the rules then the name should be changed accordingly in
#  this list.

TEST_TARGETS += embedstl pair  stlString secondindex

ifeq ($(OS),Windows_NT)
	TEST_TARGETS += mapvector
	CLEAN_TARGETS += mapvector.root
else
	UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        TEST_TARGETS += mapvector
		CLEAN_TARGETS += mapvector.root
    endif
    ifeq ($(UNAME_S),Darwin)
		MACOSVERSION := $(shell sw_vers --productVersion)
        ifneq ($(MACOSVERSION),26.0)
            TEST_TARGETS += mapvector
			CLEAN_TARGETS += mapvector.root
        endif
    endif
endif


#stlBrowse

# Searched for Rules.mk in roottest/scripts
# Algorithm:  Find the current working directory and remove everything after
#  '*roottest/'.  Append the path for Rules.mk from within roottest, which 
#  should be 'scripts/Rules.mk'.  The roottest path is saved in the ROOTTEST_HOME
#  variable for use by the SUBDIRECTORIES variable and is exported to eliminate the 
#  overhead of findding the path again during recursive calls of gmake.
# Since all makefiles should be under roottest or one of its
#  subdirectories and all recursions of gmake are called by 'cd [DIR]; gmake ...'
#  this algorithm should not fail in finding /roottest/ in the
#  current working directory.  
# Issues:  This algorithm will fail if a makefile is called from outside the 
#  roottest folder, as in executing 'gmake -f ~/roottest/Makefile' from the home
#  directory.

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
# By default the TestDiff variable is set to unix 'diff' with some options passed to it in Rules.mk.
#  To run a diff and ignore whitespacing the -w options should be specified before the files are
#  listed.

testWithDiff: testWithDiff.log testWithDiff.ref
	$(TestDiff)

stlString: stlString.log
	$(TestDiff)

stlBrowse: stlBrowse.log
	$(TestDiff)

pair: pair.log
	$(TestDiff)

embedstl: embedstl.log
	$(TestDiff)

mapvector.root: mapvector_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l mapvector.C+ > mapvector_root.build.log 2>&1 || handleError.sh --cmd='Execution of $<' --result=$$? --log=mapvector_root.build.log --test=$*

mapvector.log: mapvector.root

mapvector: mapvector.log
	$(TestDiff)

secondindex: secondindex.log
	$(TestDiff)
