# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES)

#Set the list of target to make while testing
TEST_TARGETS += customStreamerTest

# adjust the location of Rules.mk as needed.
include ../../Rules.mk

customStreamerTest.log:
	$(CMDECHO) root.exe -q -b -l run.C 2>&1 | grep -v 'creating shared library ' > $@

customStreamerTest: customStreamerTest.log
	$(CMDECHO) diff -b customStreamerTest.log customStreamerTest.ref

mytest:
	$(CMDECHO) echo mytest > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


