# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += 

#Set the list of target to make while testing
TEST_TARGETS += addTest

# adjust the location of Rules.mk as needed.
include ../../Rules.mk

addTest:
	$(CMDECHO) root.exe -b -q -l run.C > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


