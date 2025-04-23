CLEAN_TARGETS += *.so Class*.root typedef*.root $(ALL_LIBRARIES) *.clog *.log *.tmp skiptestfile.xml arr.root FixedArray*.root boolUpdate.root  data.root  skiptestfile.root $(FILES) $(FILESFAIL) *.success $(STL_FILES) atlas32 inheritstl.root
TEST_TARGETS += foreign310 boolUpdate skip ClassConv ConvArray typedef Missing FixedArrayLengthChange ClonesArrayEvo NumericalTypeEvolution execioReadRuleEquivalentCode

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


ifeq ($(FAIL),)
ConvArray: readold.log
	$(WarnFailTest)
else
ConvArray: readold.log readnew.log
endif

ClassConv: ClassConv.log
	$(TestDiff)

ifneq ($(ClingWorkAroundMissingDynamicScope),)
data.root: CALLROOTEXE += -e 'gROOT->ProcessLine(".L MyClassOld.cxx+");'
endif

data.root: MyClassOld_cxx.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l write.C > write.log 2>&1 || handleError.sh --cmd='Execution of $<' --result=$$? --log=write.log --test=$*

ifneq ($(ClingWorkAroundMissingDynamicScope),)
readold.log: CALLROOTEXE += -e 'gROOT->ProcessLine(".L MyClassOld.cxx+");'
endif

readold.log: MyClassOld_cxx.$(DllSuf) data.root
	$(CMDECHO) $(CALLROOTEXE) -q -b -l readold.C > readold.log 2>&1 || handleError.sh --cmd='Execution of $<' --result=$$? --log=readold.log --test=$*

readnew.log: MyClass_C.$(DllSuf) data.root
	$(CMDECHO) $(CALLROOTEXE) -q -b -l readnew.C > readnew.log 2>&1 || handleError.sh --cmd='Execution of $<' --result=$$? --log=readnewlog --test=$*

ClassConv.root: ClassConvOld_cxx.$(DllSuf)  ClassConvWriteOld.cxx
	$(CMDECHO) $(CALLROOTEXE) -q -b -l ClassConvWriteOld.cxx > ClassConvWriteOld.log 2>&1 || handleError.sh --cmd='Execution of $<' --result=$$? --log=ClassConvWriteOld.log --test=$*

ClassConvReadOld.log: ClassConvOld_cxx.$(DllSuf) ClassConv.root ClassConvReadOld.cxx
	$(CMDECHO) $(CALLROOTEXE) -q -b -l ClassConvReadOld.cxx > ClassConvReadOld.log 2>&1 || handleError.sh --cmd='Execution of $<' --result=$$? --log=ClassConvReadOld.log --test=$*

ClassConvNew.root: ClassConvNew_cxx.$(DllSuf) ClassConvWriteNew.cxx
	$(CMDECHO) $(CALLROOTEXE) -q -b -l ClassConvWriteNew.cxx > ClassConvWriteNew.log 2>&1 || handleError.sh --cmd='Execution of $<' --result=$$? --log=ClassConvWriteNew.log --test=$*

ClassConv.log: ClassConv.root ClassConvNew.root ClassConvNew_cxx.$(DllSuf)  ClassConvOther_cxx.$(DllSuf) ClassConvReadOld.log
	$(CMDECHO) $(CALLROOTEXE) -q -b -l ClassConvReadNew.cxx > ClassConv.log 2>&1 || handleError.sh --cmd='Execution of $<' --result=$$? --log=ClassConv.log --test=$*
	$(CMDECHO) $(CALLROOTEXE) -q -b -l ClassConvReadOther.cxx >> ClassConv.log 2>&1 || handleError.sh --cmd='Execution of $<' --result=$$? --log=ClassConv.log --test=$*

typedefWrite.log : $(ROOTCORELIBS) typedefWrite_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l typedefWrite.C+ > typedefWrite.log 2>&1  || handleError.sh --cmd='Execution of $<' --result=$$? --log=typedefWrite.log --test=$*

typedefWrite : typedefWrite.log
	$(TestDiff)

typedef.log: typedefWrite.log

typedef: typedef.log
	$(TestDiff)

skipWrite.log: skipWrite_C.$(DllSuf) $(ROOTCORELIBS)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l skipWrite.C+ > skipWrite.log 2>&1  || handleError.sh --cmd='Execution of $<' --result=$$? --log=skipWrite.log --test=$*

skip.log: skipWrite.log skipRead_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l runSkip.C 2>&1 | sed -e's/method=.*//' > skip.log

skip: skip.log
	$(TestDiff)

boolUpdate.root: boolUpdate.keeproot
	$(CMDECHO) cp boolUpdate.keeproot boolUpdate.root

boolUpdate.log: boolUpdate_C.$(DllSuf) boolUpdate.root

boolUpdate: boolUpdate.log
	$(TestDiff)

foreign310.log: cms_cxx.$(DllSuf)

foreign310: foreign310.log
	$(TestDiff)

Missing: Missing.log
	$(TestDiff)

fixarr1_C.$(DllSuf) : fixarr.h
fixarr2_C.$(DllSuf) : fixarr.h
fixarr3_C.$(DllSuf) : fixarr.h

fixarr.log: fixarr1_C.$(DllSuf) fixarr2_C.$(DllSuf) fixarr3_C.$(DllSuf)
	$(CMDECHO) ( $(CALLROOTEXE) -q -b -l fixarr1.C+ && \
		$(CALLROOTEXE) -q -b -l fixarr2.C+ && \
		$(CALLROOTEXE) -q -b -l fixarr3.C+ ) \
		 > fixarr.log 2>&1  || handleError.sh --cmd='Execution of $<' --result=$$? --log=fixarr.log --test=$*

fixarr: fixarr.log
	$(TestDiff)

.PHONY: WriteFixedArrayOld WriteFixedArrayNew
.PHONY: ReadFixedArrayOldWithNew ReadFixedArrayNewWithOld

FixedArrayLengthChange: WriteFixedArrayOld WriteFixedArrayNew
	$(CMDECHO) $(MAKE) ReadFixedArrayOldWithNew
	$(CMDECHO) $(MAKE) ReadFixedArrayNewWithOld

WriteFixedArrayOld : WriteFixedArrayOld.clog
	$(TestDiff)

WriteFixedArrayNew : WriteFixedArrayNew.clog
	$(TestDiff)

ReadFixedArrayOldWithNew : ReadFixedArrayOldWithNew.clog
	$(TestDiff)

ReadFixedArrayNewWithOld : ReadFixedArrayNewWithOld.clog
	$(TestDiff)

ClonesArrayEvo.log: Event_3_cxx.$(DllSuf)

ClonesArrayEvo: ClonesArrayEvo.log
	$(TestDiff)

.PRECIOUS: lib%.$(DllSuf) %.root

$(SUCCESSES) : check%.success: check.%.log
ifeq ($(FAST),)
	$(CMDECHO) diff -b check.ref $< && touch $@
else
	$(CMDECHO) diff -b check_fast.ref $< && touch $@
endif

atlas32: assertMakeAtlas32.elog
	$(CMDECHO) diff -w assertMakeAtlas32.elog  assertMakeAtlas32.ref

assertLongLong64.elog: atlas32

include $(ROOTTEST_HOME)/scripts/Reflex.mk

execioReadRuleEquivalentCode: execioReadRuleEquivalentCode.log

execioReadRuleEquivalentCode.log: libioReadRuleEquivalentCode_dictrflx.$(DllSuf)

execStlInheritNew.clog: execStlInheritOld.success

