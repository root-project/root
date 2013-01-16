# This is a template for all makefiles.

#Set the list of files to be deleted by clean (Targets can also be specified).:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog NuEvent_DST.mark

# Set the list of target to make while testing.  By default, mytest is the
# only target added.  If the name of the target is changed in the rules then
# the name should be changed accordingly in this list.

TEST_TARGETS += slowreading mytest

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

NuEvent_DST.mark: Read.C NuEvent_60_cxx.$(DllSuf) NuEvent_61_cxx.$(DllSuf) NuEvent_62_cxx.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -l -q 'Read.C("NuEvent_60_cxx", "rootfiles/DST_60.root", 1)' > NuEvent_DST.log 2>&1 || handleError.sh --log=NuEvent_DST.log --result=$$?

CALLROOTEXE_ESC :=$(subst ','\'',$(CALLROOTEXE))

readDST.00.log: Read.C NuEvent_60_cxx.$(DllSuf) NuEvent_DST.mark
	$(CMDECHO) export CMDECHO="$(CMDECHO)"; ./trytwice.sh '$(CALLROOTEXE_ESC)' "NuEvent_60_cxx" "rootfiles/DST_60.root" readDST.00.log

readDST.01.log: Read.C NuEvent_60_cxx.$(DllSuf) NuEvent_DST.mark
	$(CMDECHO) export CMDECHO="$(CMDECHO)"; ./trytwice.sh '$(CALLROOTEXE_ESC)' "NuEvent_60_cxx" "rootfiles/DST_61.root" readDST.01.log

readDST.02.log: Read.C NuEvent_60_cxx.$(DllSuf) NuEvent_DST.mark
	$(CMDECHO) export CMDECHO="$(CMDECHO)"; ./trytwice.sh '$(CALLROOTEXE_ESC)' "NuEvent_60_cxx" "rootfiles/DST_62.root" readDST.02.log

readDST.10.log: Read.C NuEvent_61_cxx.$(DllSuf) NuEvent_DST.mark
	$(CMDECHO) export CMDECHO="$(CMDECHO)"; ./trytwice.sh '$(CALLROOTEXE_ESC)' "NuEvent_61_cxx" "rootfiles/DST_60.root" readDST.10.log

readDST.11.log: Read.C NuEvent_61_cxx.$(DllSuf) NuEvent_DST.mark
	$(CMDECHO) export CMDECHO="$(CMDECHO)"; ./trytwice.sh '$(CALLROOTEXE_ESC)' "NuEvent_61_cxx" "rootfiles/DST_61.root" readDST.11.log

readDST.12.log: Read.C NuEvent_61_cxx.$(DllSuf) NuEvent_DST.mark
	$(CMDECHO) export CMDECHO="$(CMDECHO)"; ./trytwice.sh '$(CALLROOTEXE_ESC)' "NuEvent_61_cxx" "rootfiles/DST_62.root" readDST.12.log

readDST.20.log: Read.C NuEvent_62_cxx.$(DllSuf) NuEvent_DST.mark
	$(CMDECHO) export CMDECHO="$(CMDECHO)"; ./trytwice.sh '$(CALLROOTEXE_ESC)' "NuEvent_62_cxx" "rootfiles/DST_60.root" readDST.20.log

readDST.21.log: Read.C NuEvent_62_cxx.$(DllSuf) NuEvent_DST.mark
	$(CMDECHO) export CMDECHO="$(CMDECHO)"; ./trytwice.sh '$(CALLROOTEXE_ESC)' "NuEvent_62_cxx" "rootfiles/DST_61.root" readDST.21.log

readDST.22.log: Read.C NuEvent_62_cxx.$(DllSuf) NuEvent_DST.mark
	$(CMDECHO) export CMDECHO="$(CMDECHO)"; ./trytwice.sh '$(CALLROOTEXE_ESC)' "NuEvent_62_cxx" "rootfiles/DST_62.root" readDST.22.log


slowreading.log: readDST.00.log readDST.01.log readDST.02.log \
                 readDST.10.log readDST.11.log readDST.12.log \
                 readDST.20.log readDST.21.log readDST.22.log
	$(CMDECHO) cat $^ > slowreading.log

slowreading: slowreading.log
	$(TestDiff)
