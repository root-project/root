# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += *.o

#Set the list of target to make while testing
TEST_TARGETS += referencesTest

# adjust the location of Rules.mk as needed.
ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


ifeq ($(FAIL),)
referencesTest: 
	$(WarnFailTest)
else
referencesTest: libmaster.$(DllSuf)
	$(CMDECHO) echo mytest > /dev/null
endif

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

MasterDict.C: MyClass.hh MasterLinkDef.hh $(ROOTSYS)/bin/rootcint
	$(CMDECHO) rootcint -f $@ -c MyClass.hh MasterLinkDef.hh

libmaster.$(DllSuf): MasterDict.o MyClass.o
	$(LD) $(SOFLAGS) $(LDFLAGS) $^ -o $@
