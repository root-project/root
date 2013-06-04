# %ROOTTEST-duration%:35
# This is a template for all makefiles.

#Set the list of files to be deleted by clean (Targets can also be specified).:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log dt_reference.root Event.new.*.root \
  loadernotapp$(ExeSuf) loadernotapp.dSYM loadernotapp.success

# Set the list of target to make while testing.  By default, mytest is the
# only target added.  If the name of the target is changed in the rules then
# the name should be changed accordingly in this list.

TEST_TARGETS += missingInfo nospace namespace autoload mytest drawing testloadernotapp

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
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
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



autoload: autoload.log
	$(TestDiff)

namespace.log: fornamespace_C.$(DllSuf)

namespace: namespace.log
	$(TestDiff)

nospace.log: WrapSimple_h.$(DllSuf)

nospace: nospace.log
	$(TestDiff)

drawing.raw.log: DrawTest.sh  dt_DrawTest.C $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV) MainEvent.cxx Event.cxx Event.h \
                 $(wildcard dt_wrap*.C) dt_MakeFiles.sh dt_RunDrawTest.sh
	$(CMDECHO) ./DrawTest.sh > drawing.raw.log 2>&1

drawing.log: drawing.raw.log
	$(CMDECHO) grep -v 'ROOTMARKS' drawing.raw.log | grep '^..' | \
	grep -v -e '^g\?make' -e ' creating shared library' | \
	grep -v -e 'Event.*root made' -e 'dt.*made' -e ExecutionContext > drawing.log

drawing: drawing.log
	$(TestDiff)

missingInfo.log: missingInfo_C.$(DllSuf)

missingInfo: missingInfo.log
	$(TestDiff)

ifeq ($(PLATFORM),win32)
CXXFLAGS += -TP
LIBSFORLINK = $(LIBS)
CLDFLAGS := -link $(LDFLAGS)
SYSLIBS += msvcprt.lib 
else
LIBSFORLINK = 
endif

loadernotapp$(ExeSuf): loadernotapp.cxx
	$(CMDECHO)$(CXX)  $^ $(CFLAGS) $(CXXFLAGS) $(CLDFLAGS) $(ROOTLIBS) $(OutPutOpt)$@

loadernotapp.log: loadernotapp$(ExeSuf)
	$(CMDECHO) ./loadernotapp$(ExeSuf) > loadernotapp.log

testloadernotapp: loadernotapp.success

loadernotapp.success: loadernotapp.log
	$(SuccessTestDiff) && touch $@

execResolveTypedef.success: EXTRA_DIFFOPTS += -w
