# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log

#Set the list of target to make while testing
TEST_TARGETS += transientTest

# adjust the location of Rules.mk as needed.
include ../../Rules.mk

transientTest.log: TBigDSWriteEvent_cxx.$(DllSuf)  TBigDSWriteParticle_cxx.$(DllSuf)
	$(CMDECHO) root.exe -q -l -b run.C 2>&1 > transientTest.log

transientTest: transientTest.log
	$(CMDECHO) diff -b  transientTest.ref  transientTest.log

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


