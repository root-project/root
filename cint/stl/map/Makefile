# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += dict* *.o

#Set the list of target to make while testing
TEST_TARGETS += mapTest

# adjust the location of Rules.mk as needed.
include ../../../root/Rules.mk

mapTest: dict.o testing.o
	$(CMDECHO) echo mytest > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

dict.C: testing.C
	$(CMDECHO) rootcint -f dict.C -c testing.C LinkDef.h


