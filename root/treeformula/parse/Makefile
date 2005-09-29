# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS +=  *.log generatedSel* $(ALL_LIBRARIES) cmake*

#Set the list of target to make while testing
TEST_TARGETS += cscan gaus aliases templateName BoolOrder testStringFormula testBoolOpt expotest NonFormulaMean \
	templateBase

# adjust the location of Rules.mk as needed.
ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


testStringFormula.log: testString.C
	$(CMDECHO) root.exe -q -l -b testString.C > testStringFormula.log

testStringFormula: testStringFormula.log
	$(CMDECHO) echo testStringFormula > /dev/null

testBoolOpt.log: testBoolOpt.C
	$(CMDECHO) root.exe -b -l -q testBoolOpt.C > testBoolOpt.log 2>&1

testBoolOpt : testBoolOpt.log
	$(CMDECHO) diff -b testBoolOpt.ref testBoolOpt.log

expotest: expotest.log
	$(TestDiff)

NonFormulaMean:
ifeq ($(FAIL),)
	$(WarnFailTest)
else
	$(CMDECHO) !(root.exe -q -b -l NonFormulaMean.C)
endif

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

#BoolOrder.log: runBoolOrder.C BoolOrder.ref $(ROOTSYS)/lib/libTreePlayer.$(DllSuf)
#	$(CMDECHO) root.exe -q -b -l runBoolOrder.C 2>&1 | grep -v -e LorentzVector -e TVector3 > $@

BoolOrder: BoolOrder.log
	$(TestDiff)

templateName.log: templateName_cpp.$(DllSuf) $(ROOTCORELIBS) $(ROOT_LOC)/lib/libTreePlayer.$(DllSuf)
	$(CMDECHO) root.exe -q -b -l runtemplateName.C 2>&1 | grep -v -e 'Creating library' -e \\.cxx -e 'creating shared' -e 'Warning in <TClassTable::Add>: class generatedSel already in TClassTable' > templateName.log

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
