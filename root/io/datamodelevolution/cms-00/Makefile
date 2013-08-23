# This is a template for all makefiles.

#Set the list of files to be deleted by clean (Targets can also be specified).:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog *estv1.root *_cint.cpp *_rflx.cpp *_cint.h *_rflx.h ../logs/cms-00/*.log

# Set the list of target to make while testing.  By default, mytest is the
# only target added.  If the name of the target is changed in the rules then
# the name should be changed accordingly in this list.

TEST_TARGETS += test1 read_test1 test2 reflex_test1 reflex_test2 

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

.PHONY: test1 test2 test3 test4 reflex_test1

test1.root: test1.clog
reflex_test1.root: reflex_test1.clog

test1_cxx.$(DllSuf): DataModelV1.h

read_test1_cxx.$(DllSuf): DataModelV1.h

test1.clog: test1_cxx.$(DllSuf) libDataModelV1_dictcint.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -q -l test1.cxx+ > test1.clog 2>&1 || handleError.sh --result=$$? --log=$@ --test=$<  

ifneq ($(ClingWorkAroundNoPrivateClassIO),)
test1:
	@true
else
test1: test1.clog
	$(TestDiff)
endif

read_test1.clog: read_test1_cxx.$(DllSuf) libDataModelV1_dictcint.$(DllSuf) test1
	$(CMDECHO) $(CALLROOTEXE) -b -q -l read_test1.cxx+ > read_test1.clog 2>&1 || handleError.sh --result=$$? --log=$@ --test=$<  

test1.comp.log: read_test1.clog
	$(CMDECHO) (cd ../logs/cms-00; ../../common/compare.sh -s 0) > test1.comp.log

ifneq ($(ClingWorkAroundNoPrivateClassIO),)
read_test1:
	@true
else
read_test1: test1.comp.log
	$(TestDiff)
endif

reflex_test1.clog: test1_cxx.$(DllSuf) libDataModelV1_dictrflx.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -q -l test1.cxx+\(\"r\"\) > reflex_test1.clog 2>&1 || handleError.sh --result=$$? --log=$@ --test=$<  

test2_cxx.$(DllSuf): DataModelV2.h

test2.clog: test2_cxx.$(DllSuf) test1 libDataModelV2_dictcint.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -q -l test2.cxx+ > test2.clog 2>&1 || handleError.sh --result=$$? --log=$@ --test=$<  

test2.comp.log: test2.clog
	$(CMDECHO) (cd ../logs/cms-00; ../../common/compare.sh -s 1) > test2.comp.log 

ifneq ($(ClingWorkAroundNoPrivateClassIO),)
test2:
	@true
else
test2: test2.comp.log
	$(TestDiff)
endif

reflex_test2.clog: test2_cxx.$(DllSuf) libDataModelV2_dictrflx.$(DllSuf) reflex_test1
	$(CMDECHO) $(CALLROOTEXE) -b -q -l test2.cxx+\(\"r\"\) > reflex_test2.clog 2>&1 || handleError.sh --result=$$? --log=$@ --test=$<  

reflex_test2.comp.log: reflex_test2.clog
	$(CMDECHO) (cd ../logs/cms-00; ../../common/compare.sh -s 1 -d reflex_ ) > reflex_test2.comp.log

test3.clog: test3_cxx.$(DllSuf) test2  libDataModelV3_dictcint.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -q -l test3.cxx+ > test3.clog 2>&1 || handleError.sh --result=$$? --log=$@ --test=$<  

test3.comp.log: test3.clog
	$(CMDECHO) (cd ../logs/cms-00; ../../common/compare.sh -s 2) > test3.comp.log

test3: test3.comp.log
	$(TestDiff)

reflex_test3.clog: test3_cxx.$(DllSuf) reflex_test2 libDataModelV3_dictrflx.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -q -l test3.cxx+\(\"r\"\) > reflex_test3.clog 2>&1

reflex_test3.comp.log: reflex_test3.clog
	$(CMDECHO) (cd ../logs/cms-00; ../../common/compare.sh -s 2 -d reflex_ ) > reflex_test3.comp.log

ifeq ($(findstring gccxml,$(notdir $(HAVEGCCXML))),gccxml)

reflex_test1: reflex_test1.clog
	$(TestDiff)

reflex_test2: reflex_test2.comp.log
	$(TestDiff)

reflex_test3: reflex_test3.clog
	$(TestDiff)

else

reflex_test1:
ifeq ($(HAVEGCCXML),"nocintex")
	$(CMDECHO) echo "Your ROOT build does not include cintex: skipping test $@"
else
	$(CMDECHO) echo "GCCXML not in \$$PATH: skipping test $@"
endif

reflex_test2: reflex_test1

reflex_test3: reflex_test2

endif

execReadRename1.clog: execWriteRename.success

execReadRename2.clog: execWriteRename.success

