# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += *.o lotsRef.root $(ALL_LIBRARIES) *.log *.build.log refarray.root

#Set the list of target to make while testing
TEST_TARGETS += lotsRef referencesTest refarray

# adjust the location of Rules.mk as needed.
ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

ifneq ($(MAKECMDGOALS),clean)
-include $(wildcard *.d)
endif

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

MasterDict.C: MyClass.hh MasterLinkDef.hh $(ROOTCINT)
	$(CMDECHO) rootcint -f $@ -c MyClass.hh MasterLinkDef.hh

libmaster.$(DllSuf): MasterDict.o MyClass.o
	$(LD) $(SOFLAGS) $(LDFLAGS) $^ -o $@

lotsRef_C.$(DllSuf): A_C.$(DllSuf) lotsRef.C create.C
	$(CMDECHO) $(CALLROOTEXE) -q -b -l create.C > lotsRef_C.build.log 2>&1 

lotsRef.root: lotsRef_C.$(DllSuf) $(ROOTCORELIBS) 
	$(CMDECHO) ($(CALLROOTEXE) -q -b -l 'lotsRef.C+(300)' && $(CALLROOTEXE) -q -b -l 'lotsRef.C+(2)' && $(CALLROOTEXE) -q -b -l 'lotsRef.C+(1)' ) > logRef_root.log 2>&1 && \
	diff -bw logRef_root.ref logRef_root.log

lotsRef.log: lotsRef.root lotsRef_C.$(DllSuf) A_C.$(DllSuf)

lotsRef: lotsRef.log
	$(TestDiff) && diff -bw logRef_root.log logRef_root.ref

refarray.root: refarray_C.$(DllSuf) $(ROOTCORELIBS) 
	$(CMDECHO) $(CALLROOTEXE) -q -b -l 'refarray.C+(1)' > refarray_root.log 2>&1

refarray_full.log: refarray.root refarray_C.$(DllSuf) $(ROOTCORELIBS)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l 'refarray.C+' > refarray_full.log 2>&1

refarray.log: refarray_full.log
	$(CMDECHO) cat refarray_full.log | sed -e 's/at [x0-9a-fA-F]* is/at 0xNA is/' -e 's/ProcessID.*)/na)/' > refarray.log

refarray: refarray.log
	$(TestDiff)
