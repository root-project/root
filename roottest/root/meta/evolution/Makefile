# This is a template for all makefiles.

BASE_FILES = baseClass_merge1.root baseClass_merge2.root baseClass_53410_v1.root baseClass_53410_v2.root
MIXED_FILES = mixedBase_merge1.root mixedBase_merge2.root mixedBase_53410_v1.root mixedBase_53410_v2.root
BASE_SUCCESS = $(subst .root,.success,$(BASE_FILES) $(MIXED_FILES))

#Set the list of files to be deleted by clean (Targets can also be specified).:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog data*.root mixedBase_v?.root baseClass_v?.root baseClass_merge?.root mixedBase_merge?.root baseClass_merged?.root

# Set the list of target to make while testing.  By default, mytest is the
# only target added.  If the name of the target is changed in the rules then
# the name should be changed accordingly in this list.

TEST_TARGETS += foreign warning $(BASE_SUCCESS)

ifeq ($(strip $(ROOTTEST_HOME)),)
   export ROOTTEST_HOME := $(shell git rev-parse --show-toplevel)/roottest/
   ifeq ($(strip $(ROOTTEST_HOME)),)
      export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
   endif
   ifeq ($(strip $(ROOTTEST_HOME)),)
      $(error The head of roottest was not found.  Set ROOTTEST_HOME)
   endif
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

baseClass_merge1.root: execBaseClass_v2.success
	$(CMDECHO) hadd -f $@ baseClass_v1.root baseClass_v2.root > $@.build.log 2>&1 || handleError.sh --cmd="hadd -f $@ baseClass_v1.root baseClass_v2.root" --result=$$? --log=$@.build.log --test=$@

baseClass_merge2.root: execBaseClass_v2.success
	$(CMDECHO) hadd -f $@ baseClass_v2.root baseClass_v1.root > $@.build.log 2>&1 || handleError.sh --cmd="hadd -f $@ baseClass_v2.root baseClass_v1.root" --result=$$? --log=$@.build.log --test=$@

mixedBase_merge1.root: execMixedBaseClass_v2.success
	$(CMDECHO) hadd -f $@ mixedBase_v1.root mixedBase_v2.root > $@.build.log 2>&1 || handleError.sh --cmd="hadd -f $@ mixedBase_v1.root mixedBase_v2.root" --result=$$? --log=$@.build.log --test=$@

mixedBase_merge2.root: execMixedBaseClass_v2.success
	$(CMDECHO) hadd -f $@ mixedBase_v2.root mixedBase_v1.root > $@.build.log 2>&1 || handleError.sh --cmd="hadd -f $@ mixedBase_v2.root mixedBase_v1.root" --result=$$? --log=$@.build.log --test=$@

execBaseClass_v2.clog: execBaseClass_v1.success

execMixedBaseClass_v2.clog: execMixedBaseClass_v1.success

$(subst .root,.log,$(BASE_FILES)): %.log: %.root execBaseClass_v2_cxx.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -q -l execBaseClass_v2.cxx+\(\"$<\"\) > $@ 2>&1 || handleError.sh --cmd="Execution of execBaseClass_v2.cxx+(\"$<\") > $@" --result=$$? --log=$@.build.log --test=$@

$(subst .root,.log,$(MIXED_FILES)): %.log: %.root execMixedBaseClass_v2_cxx.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -q -l execMixedBaseClass_v2.cxx+\(\"$<\"\) > $@ 2>&1 || handleError.sh --cmd="Execution of execMixedBaseClass_v2.cxx+(\"$<\") > $@" --result=$$? --log=$@.build.log --test=$@

$(BASE_SUCCESS): %.success: %.log
	$(SuccessTestDiff) && touch $@





