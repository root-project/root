CLEAN_TARGETS += *.so Class*.root typedef*.root $(ALL_LIBRARIES) *.clog *.log *.tmp *.xml arr.root FixedArray*.root boolUpdate.root  data.root  skiptestfile.root $(FILES) $(FILESFAIL) *.rootmap *.success $(STL_FILES) atlas32
TEST_TARGETS += foreign310 boolUpdate skip ClassConv ConvArray typedef Missing FixedArrayLengthChange ClonesArrayEvo NumericalTypeEvolution

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
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
	$(CMDECHO) $(CALLROOTEXE) -q -b -l write.C > write.log 

ifneq ($(ClingWorkAroundMissingDynamicScope),)
readold.log: CALLROOTEXE += -e 'gROOT->ProcessLine(".L MyClassOld.cxx+");'
endif

readold.log: MyClassOld_cxx.$(DllSuf) data.root
	$(CMDECHO) $(CALLROOTEXE) -q -b -l readold.C > readold.log 2>&1

readnew.log: MyClass_C.$(DllSuf) data.root
	$(CMDECHO) $(CALLROOTEXE) -q -b -l readnew.C > readnew.log 2>&1

ClassConv.root: ClassConvOld_cxx.$(DllSuf)  ClassConvWriteOld.cxx
	$(CMDECHO) $(CALLROOTEXE) -q -b -l ClassConvWriteOld.cxx > ClassConvWriteOld.log

ClassConvReadOld.log: ClassConvOld_cxx.$(DllSuf) ClassConv.root ClassConvReadOld.cxx 
	$(CMDECHO) $(CALLROOTEXE) -q -b -l ClassConvReadOld.cxx > ClassConvReadOld.log

ClassConvNew.root: ClassConvNew_cxx.$(DllSuf) ClassConvWriteNew.cxx
	$(CMDECHO) $(CALLROOTEXE) -q -b -l ClassConvWriteNew.cxx > ClassConvWriteNew.log

ClassConv.log: ClassConv.root ClassConvNew.root ClassConvNew_cxx.$(DllSuf)  ClassConvOther_cxx.$(DllSuf) ClassConvReadOld.log
	$(CMDECHO) $(CALLROOTEXE) -q -b -l ClassConvReadNew.cxx > ClassConv.log 2>&1 
	$(CMDECHO) $(CALLROOTEXE) -q -b -l ClassConvReadOther.cxx >> ClassConv.log 2>&1 

typedef.root : $(ROOTCORELIBS) typedefWrite_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l typedefWrite.C+ > typedefWrite.log

typedef.log: typedef.root

typedef: typedef.log
	$(TestDiff)

skipWrite.log: skipWrite_C.$(DllSuf) $(ROOTCORELIBS)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l skipWrite.C+ > skipWrite.log

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
		 > fixarr.log 2>&1

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

FILES := int.root float16.root double32.root regular.root char.root  \
       short.root long.root longlong.root uchar.root ushort.root \
       uint.root ulong.root ulonglong.root float.root double.root \
       double32enough.root double32mantis.root \
       float16enough.root float16mantis.root
FILESFAIL := double32tooshort.root float16tooshort.root
ifneq ($(FAST),)
FILES := int.root double32.root longlong.root float.root
FILESFAIL := 
endif

LIBS := $(patsubst %.root,lib%.$(DllSuf),$(FILES) $(FILESFAIL))

LOGS = $(patsubst %.root,check.%.log,$(FILES) $(FILESFAIL))

SUCCESSES = $(patsubst %.root,check%.success,$(FILES) $(FILESFAIL))

.PHONY: files

.PRECIOUS: lib%.$(DllSuf) %.root

files: $(FILES) $(FILESFAIL)

logs: $(LOGS)

NumericalTypeEvolution: $(SUCCESSES)

$(LIBS): lib%.$(DllSuf): rootalias.C float16.cxx float16.h compile.C
	$(CMDECHO) $(CALLROOTEXE) -b -l -q 'compile.C(0, "$*")' > $*.log 2>&1

%.root: lib%.$(DllSuf) rootalias.C run.C
	$(CMDECHO) $(CALLROOTEXE) -b -l -q 'run.C("$*")' > lib$*.log 2>&1 

$(LOGS) : check.%.log: %.root $(FILES) $(FILESFAIL) read.C
	$(CMDECHO) $(CALLROOTEXE) -b -l -q 'read.C("$(FILES)", "$(FILESFAIL)", 0, "$*")' 2>&1 | grep -v Processing > $@

$(SUCCESSES) : check%.success: check.%.log
ifeq ($(FAST),)
	$(CMDECHO) diff -b check.ref $< && touch $@
else
	$(CMDECHO) diff -b check_fast.ref $< && touch $@
endif

STL_FILES := map.root vector.root multimap.root list.root

STL_LIBS := $(patsubst %.root,lib%.$(DllSuf),$(STL_FILES))

STL_LOGS = $(patsubst %.root,check.%.log,$(STL_FILES))

STL_SUCCESSES = $(patsubst %.root,check%.success,$(STL_FILES))

.PHONY: stl_files

stl_files: $(STL_FILES)

stl_logs: $(STL_LOGS)

StlTypeEvolution: $(patsubst %.root,check%.success,$(STL_FILES))

$(STL_LIBS): lib%.$(DllSuf): rootalias.C maptovector.cxx maptovector.h compile.C
	$(CMDECHO) $(CALLROOTEXE) -b -l -q 'compile.C(1, "$*")' > $*.log 2>&1

$(STL_LOGS) : check.%.log: %.root $(STL_FILES) read.C
	$(CMDECHO) $(CALLROOTEXE) -b -l -q 'read.C("$(STL_FILES)", "", 1, "$*")' 2>&1 | grep -v Processing > $@

$(STL_SUCCESSES) : check%.success: check.%.log
	$(CMDECHO) diff -b check.stl.ref $< && touch $@

atlas32: assertMakeAtlas32.elog
	$(CMDECHO) diff -w assertMakeAtlas32.elog  assertMakeAtlas32.ref

assertLongLong64.elog: atlas32

