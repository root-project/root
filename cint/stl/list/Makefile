# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += 

#Set the list of target to make while testing
TEST_TARGETS += 

# adjust the location of Rules.mk as needed.
include ../../../root/Rules.mk

vectorTest:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif
	$(CMDECHO) echo mytest > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


