TEST_TARGETS += scopeTest scopeTest2
CLEAN_TARGETS += $(ALL_LIBRARIES) output *dict*

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk



badDict.C: mycl.C LinkDef.h
	$(CMDECHO) rootcint -f badDict.C -c -DBAD mycl.C LinkDef.h

goodDict.C: mycl.C LinkDef.h
	$(CMDECHO) rootcint -f goodDict.C -c mycl.C LinkDef.h

myclbad.$(DllSuf): badDict.$(ObjSuf)
	$(BuildFromObj)

myclgood.$(DllSuf): goodDict.$(ObjSuf)
	$(BuildFromObj)

scopeTest: run.C myclbad.$(DllSuf)
	$(CMDECHO) root.exe -l -q -b run.C  2>&1 | grep -v Processing | $(RemoveLeadingDirs) | sed -e s/myclbad.so/mycl.so/ > output 
	$(CMDECHO) diff -wb good.output output

scopeTest2: run.C myclgood.$(DllSuf)
	$(CMDECHO) root.exe -l -q -b rungood.C   2>&1 | grep -v Processing |$(RemoveLeadingDirs) |sed -e s/myclgood.so/mycl.so\ / > output  
	$(CMDECHO) diff -wb good.output output


