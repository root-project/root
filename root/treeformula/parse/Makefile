# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS +=  *.log generatedSel* $(ALL_LIBRARIES) cmake* test.root *.clog  speedtest.root myTest.root

#Set the list of target to make while testing
TEST_TARGETS += QEvent functca cscan gaus aliases templateName BoolOrder \
				testStringFormula testBoolOpt expotest NonFormulaMean \
				templateBase strings base form nodot

# adjust the location of Rules.mk as needed.
ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


testStringFormula.log: testString.C $(ROOTCORELIBS) $(ROOTV)
	$(CMDECHO) $(CALLROOTEXE) -q -l -b testString.C > testStringFormula.log 2>&1

testStringFormula: testStringFormula.log
	$(TestDiff)

testBoolOpt.log: testBoolOpt.C $(ROOTCORELIBS) $(ROOTV)
	$(CMDECHO) $(CALLROOTEXE) -b -l -q testBoolOpt.C > testBoolOpt.log 2>&1

testBoolOpt : testBoolOpt.log
	$(CMDECHO) diff -b testBoolOpt.ref testBoolOpt.log

expotest: expotest.log
	$(TestDiff)

NonFormulaMean:
ifeq ($(FAIL),)
	$(WarnFailTest)
else
	$(CMDECHO) !($(CALLROOTEXE) -q -b -l NonFormulaMean.C)
endif

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

#BoolOrder.log: runBoolOrder.C BoolOrder.ref $(ROOT_LOC)/lib/libTreePlayer.$(LibSuf)
#	$(CMDECHO) $(CALLROOTEXE) -q -b -l runBoolOrder.C 2>&1 | grep -v -e LorentzVector -e TVector3 > $@

BoolOrder: BoolOrder.log
	$(TestDiff)

templateName.log: templateName_cpp.$(DllSuf) $(ROOTCORELIBS) $(ROOT_LOC)/lib/libTreePlayer.$(LibSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l runtemplateName.C 2>&1 | grep -v 'ROOT version' | grep -v -e 'Creating library' -e \\.cxx -e 'creating shared' -e 'Warning in <TClassTable::Add>: class generatedSel already in TClassTable' > templateName.log

templateName: templateName.log
	$(TestDiff)

templateBase: templateBase.log
	$(TestDiff)

aliases: aliases.log
	$(TestDiff)

gaus: gaus.log
	$(TestDiff)

cscan: cscan.clog
	$(TestDiff)

functca.log: EventTcaMember_cc.$(DllSuf)

functca: functca.log
	$(TestDiff)

QEvent.log: QEvent_cc.$(DllSuf)

QEvent: QEvent.log
	$(TestDiff)

strings: strings.log

runbase_C.$(DllSuf): ClassDefinitions.C

speedtest.root: runbase_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l runbase.C+\(true\) 2>&1 > speedtest.log

base.clog: speedtest.root

base: base.clog 
	$(TestDiff)

form: form.log
	$(TestDiff)

nodot: nodot.clog
	$(TestDiff)
