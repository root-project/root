CLEAN_TARGETS += *.so *.root $(ALL_LIBRARIES) *.log *.xml
TEST_TARGETS += skip ClassConv ConvArray typedef

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
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

data.root: MyClassOld_cxx.$(DllSuf)
	$(CMDECHO) root.exe -q -b -l write.C > write.log 

readold.log: MyClassOld_cxx.$(DllSuf) data.root
	$(CMDECHO) root.exe -q -b -l readold.C > readold.log 2>&1

readnew.log: MyClass_C.$(DllSuf) data.root
	$(CMDECHO) root.exe -q -b -l readnew.C > readnew.log 2>&1

ClassConv.root: ClassConvOld_cxx.$(DllSuf)  ClassConvWriteOld.cxx
	$(CMDECHO) root.exe -q -b -l ClassConvWriteOld.cxx > ClassConvWriteOld.log

ClassConvReadOld.log: ClassConvOld_cxx.$(DllSuf) ClassConv.root ClassConvReadOld.cxx 
	$(CMDECHO) root.exe -q -b -l ClassConvReadOld.cxx > ClassConvReadOld.log

ClassConvNew.root: ClassConvNew_cxx.$(DllSuf) ClassConvWriteNew.cxx
	$(CMDECHO) root.exe -q -b -l ClassConvWriteNew.cxx > ClassConvWriteNew.log

ClassConv.log: ClassConv.root ClassConvNew.root ClassConvNew_cxx.$(DllSuf)  ClassConvOther_cxx.$(DllSuf) ClassConvReadOld.log
	$(CMDECHO) root.exe -q -b -l ClassConvReadNew.cxx > ClassConv.log 2>&1 
	$(CMDECHO) root.exe -q -b -l ClassConvReadOther.cxx >> ClassConv.log 2>&1 

typedef.root : $(ROOTCORELIBS) typedefWrite_C.$(DllSuf)
	$(CMDECHO) root.exe -q -b -l typedefWrite.C+ > typedefWrite.log

typedef.log: typedef.root

typedef: typedef.log
	$(TestDiff)

skipWrite.log: skipWrite_C.$(DllSuf) $(ROOTCORELIBS)
	$(CMDECHO) root.exe -q -b -l skipWrite.C+ > skipWrite.log

skip.log: skipWrite.log skipRead_C.$(DllSuf)
	$(CMDECHO) root.exe -q -b -l runSkip.C 2>&1 | sed -e's/method=.*//' > skip.log 

skip: skip.log
	$(TestDiff)
