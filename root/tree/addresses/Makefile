# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS +=  *.log $(ALL_LIBRARIES) *.d

#Set the list of target to make while testing
TEST_TARGETS += headerTest clonesATest EmbeddedTest inheritTest relationsTest

# adjust the location of Rules.mk as needed.
ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


inherit_C.$(DllSuf): inherit.C buildInherit.C

inheritTest.log: runInherit.C inherit_C.$(DllSuf)
	$(CMDECHO) root.exe -q -b -l runInherit.C > inheritTest.log 2>&1

inheritTest: inheritTest.log
	$(CMDECHO) diff -b inheritTest.log inheritTest.ref

header_h.$(DllSuf): header.h buildHeader.C

headerTest.log: runHeader.C header_h.$(DllSuf)
	$(CMDECHO) root.exe -q -b -l runHeader.C > headerTest.log 2>&1

headerTest: headerTest.log
	$(CMDECHO) diff -b headerTest.log headerTest.ref

clonesA.log: clonesA_rw.C clonesA_Event.h clonesA_Event.cxx
	$(CMDECHO) root.exe -q -b -l clonesA_rw.C > clonesA.log 2>&1

clonesATest: clonesA.log
	$(CMDECHO) test ! `(grep exception clonesA.log > /dev/null) && echo $?`

Embedded_load_C.$(DllSuf) : buildEmbedded.C Embedded_load.C Embedded_objects.cpp Embedded_objects.h

Embedded.root: Embedded_load_C.$(DllSuf) embwrite.C

relations_load_C.$(DllSuf) : relations.h

relations_write_C.$(DllSuf) : relations.h

relations_read_C.$(DllSuf) : relations.h

EmbeddedTest.log: Embedded_load_C.$(DllSuf) Embedded.root embread.C
	$(CMDECHO) root.exe -q -b -l embread.C > EmbeddedTest.log

EmbeddedTest: EmbeddedTest.log
	$(CMDECHO) diff -b EmbeddedTest.log EmbeddedTest.ref

relations_load_C.$(DllSuf) : relations.h

relations_write_C.$(DllSuf) : relations.h

relations_read_C.$(DllSuf) : relations.h

relationsTest.log: relations_load_C.$(DllSuf)  relations_write_C.$(DllSuf)  relations_read_C.$(DllSuf) 
	$(CMDECHO)  root.exe -q -b -l relationsRun.C | sed 's|File  Size =.*\*||g' > relationsTest.log

relationsTest: relationsTest.log 
	$(CMDECHO) diff -b relationsTest.log relationsTest.ref

mytest:
	$(CMDECHO) echo mytest > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


