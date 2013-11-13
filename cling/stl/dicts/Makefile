# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += *.o MyDict* $(ALL_LIBRARIES)

#Set the list of target to make while testing
TEST_TARGETS += stldictTest

# adjust the location of Rules.mk as needed.
ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

stldictTest: MyDict1.o MyDict2.o MyDict3.o
	$(CMDECHO) echo mytest > /dev/null

MyDict1.cxx: MyClass1.h $(ROOTCINT)
	$(CMDECHO) rootcint -f MyDict1.cxx -c MyClass1.h

MyDict2.cxx: MyClass2.h $(ROOTCINT)
	$(CMDECHO) rootcint -f MyDict2.cxx -c MyClass2.h

MyDict3.cxx: MyClass3.h $(ROOTCINT)
	$(CMDECHO) rootcint -f MyDict3.cxx -c MyClass3.h




