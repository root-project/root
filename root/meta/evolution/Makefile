# This is a template for all makefiles.

#Set the list of files to be deleted by clean (Targets can also be specified).:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog data*.root

# Set the list of target to make while testing.  By default, mytest is the
# only target added.  If the name of the target is changed in the rules then
# the name should be changed accordingly in this list.

TEST_TARGETS += foreign warning

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


# Outputs a message if the FAIL variable is null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

testWithDiff: testWithDiff.log testWithDiff.ref
	$(TestDiff)

data1_C.$(DllSuf) : data.h data1.C

data2_C.$(DllSuf) : data.h data2.C

data1.root: writefile.C data1_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -q -l 'writefile.C(1)' 2>&1 > data1.log

data2.root: writefile.C data2_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -q -l 'writefile.C(2)' 2>&1 > data2.log

data3.root: writefile.C data3_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -q -l 'writefile.C(3)' 2>&1 > data3.log

data4.root: writefile.C data4_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -q -l 'writefile.C(4)' 2>&1 > data4.log

data5.root: writefile.C data5_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -q -l 'writefile.C(5)' 2>&1 > data4.log

data6.root: writefile.C data6_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -q -l 'writefile.C(6)' 2>&1 > data4.log

foreign.log: data1.root data2.root data3.root data4.root data2_C.$(DllSuf)

warning.log: data1.root data2.root data3.root data4.root data5.root \
	         data6.root data2_C.$(DllSuf) checkWarn.C $(ROOTCORELIBS)
	$(CMDECHO) ($(CALLROOTEXE) -q -b -l 'checkWarn.C(0)' \
	    && $(CALLROOTEXE) -q -b -l 'checkWarn.C(1)' \
	    && $(CALLROOTEXE) -q -b -l 'checkWarn.C(2)' \
	    && $(CALLROOTEXE) -q -b -l 'checkWarn.C(3)' \
	    && $(CALLROOTEXE) -q -b -l 'checkWarn.C(4)' \
	    && $(CALLROOTEXE) -q -b -l 'checkWarn.C(10)' \
	    && $(CALLROOTEXE) -q -b -l 'checkWarn.C(11)' \
	    && $(CALLROOTEXE) -q -b -l 'checkWarn.C(12)' \
	    && $(CALLROOTEXE) -q -b -l 'checkWarn.C(13)' \
	    && $(CALLROOTEXE) -q -b -l 'checkWarn.C(14)' \
	    && $(CALLROOTEXE) -q -b -l 'checkWarn.C(15)' \
	    && $(CALLROOTEXE) -q -b -l 'checkWarn.C(16)' \
	    && $(CALLROOTEXE) -q -b -l 'checkWarn.C(17)' \
	    && $(CALLROOTEXE) -q -b -l 'checkWarn.C(18)' \
	    ) 2>&1 | sed -e 's@offset=  [0-9]*@offset= n/a@' > warning.log

ifeq ($(ClingWorkAroundJITandInline),)
foreign: foreign.log
	$(TestDiff)
else
foreign:
endif

warning: warning.log
	$(TestDiff)

