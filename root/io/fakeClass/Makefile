# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += testempty.root testbase.root 

#Set the list of target to make while testing
TEST_TARGETS += forwardDeclAndFake

# adjust the location of Rules.mk as needed.
include ../../Rules.mk

forwardDeclAndFake: t01.out
		  $(CMDECHO)root.exe -q -b -l t01.C >|t01.out 2>&1 

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


