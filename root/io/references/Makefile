# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += *.o lotsRef.root $(ALL_LIBRARIES)

#Set the list of target to make while testing
TEST_TARGETS += lotsRef referencesTest

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

lotsRef.root: lotsRef_C.$(DllSuf) $(ROOTCORELIBS)
	$(CMDECHO) (root -q -b -l 'lotsRef.C+(300)' && root -q -b -l 'lotsRef.C+(2)' && root -q -b -l 'lotsRef.C+(1)' ) > logRef_root.log 2>&1 && \
	diff -bw logRef_root.ref logRef_root.log

lotsRef.log: lotsRef.root lotsRef_C.$(DllSuf) A_C.$(DllSuf)

lotsRef: lotsRef.log
	$(TestDiff) && diff -bw logRef_root.ref logRef_root.log