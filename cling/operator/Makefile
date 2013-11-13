# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log

#Set the list of target to make while testing
TEST_TARGETS += EqualTest ParanTest ConversionOp dicttest

# adjust the location of Rules.mk as needed.
ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


mytest:
	$(CMDECHO) echo mytest > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

# Remember that sometest.log can be made automatically from
# from runsometest.C

EqualTest.log : runEqualTest.C equal_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -l -b  runEqualTest.C | \
	$(RemoveLeadingDirs) -e 's?FILE:.*[/\]?FILE:?' -e 's/.dll/.so/g' \
	| grep -v -e tagnum -e 'int c' -e '~privateOp2' > $@ 2>&1

EqualTest: EqualTest.log
	$(TestDiffCintSpecific)

t01.log: t01.C
	$(CMDECHO) $(CALLROOTEXE) -q -b -l t01.C > t01.log

ParanTest: t01.log
	$(CMDECHO) diff -b -w t01.log t01.ref

ConversionOp.log : runConversionOp.C ConversionOp.C ConversionOp_h.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -l -b  runConversionOp.C 2>&1 | \
	        $(RemoveLeadingDirs) -e 's?FILE:.*[/\]?FILE:?' -e 's/.dll/.so/g' \
	        | grep -v -e tagnum -e 'int c' -e 'public: void ~' > $@ 2>&1

ConversionOp: ConversionOp.log
ifeq ($(ClingWorkAroundPrintfIssues)$(ClingWorkAroundMissingUnloading),)
	$(TestDiffCintSpecific)
	$(WarnFailTest) # see ConversionOp.C for the skipped, failing tests
else
	# Don't verify this test until both the printf issue and the unloading are fixed/implemented.
endif

dicttest: dicttest_h.$(DllSuf)
