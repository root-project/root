# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log

#Set the list of target to make while testing
TEST_TARGETS += Loopbreak

# adjust the location of Rules.mk as needed.
include ../../root/Rules.mk

mytest:
	$(CMDECHO) echo mytest > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

Loopbreak: Loopbreak.log
ifeq ($(FAIL),)
	$(WarnFailTest)
else
	$(TestDiff)
endif

# Remember that sometest.log can be made automatically from
# from runsometest.C

testWithDiff: testWithDiff.log
	$(TestDiff)



