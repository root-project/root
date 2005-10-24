# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS +=  *.log $(ALL_LIBRARIES) *.d *.root

#Set the list of target to make while testing
TEST_TARGETS += ursula inheritAndOwn BadBaseAddresses treeBranch iobug merging headerTest clonesATest EmbeddedTest inheritTest relationsTest

# adjust the location of Rules.mk as needed.
ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

inherit_C.$(DllSuf): inherit.C buildInherit.C

inheritTest.log: runInherit.C inherit_C.$(DllSuf)
	$(CMDECHO) root.exe -q -b -l runInherit.C > inheritTest.log 2>&1

inheritTest: inheritTest.log
	$(CMDECHO) diff -bw inheritTest.log inheritTest.ref

header_h.$(DllSuf): header.h buildHeader.C

headerTest.log: runHeader.C header_h.$(DllSuf)
	$(CMDECHO) root.exe -q -b -l runHeader.C > headerTest.log 2>&1

headerTest: headerTest.log
	$(CMDECHO) diff -bw headerTest.log headerTest.ref

clonesA.log: clonesA_rw.C clonesA_Event.h clonesA_Event.cxx
	$(CMDECHO) root.exe -q -b -l clonesA_rw.C > clonesA.log 2>&1

clonesATest: clonesA.log
	$(CMDECHO) test ! `(grep exception clonesA.log > /dev/null) && echo $?`

Embedded_load_C.$(DllSuf) : buildEmbedded.C Embedded_load.C Embedded_objects.cpp Embedded_objects.h

Embedded.root: Embedded_load_C.$(DllSuf) embwrite.C
	$(CMDECHO) root.exe -q -b -l embwrite.C > embwrite.log

relations_load_C.$(DllSuf) : relations.h

relations_write_C.$(DllSuf) : relations.h

relations_read_C.$(DllSuf) : relations.h

EmbeddedTest.log: Embedded_load_C.$(DllSuf) Embedded.root embread.C
	$(CMDECHO) root.exe -q -b -l embread.C | sed -e "s/e+006/e+06/g" > EmbeddedTest.log

EmbeddedTest: EmbeddedTest.log
	$(CMDECHO) diff -bw EmbeddedTest.ref EmbeddedTest.log

relations_load_C.$(DllSuf) : relations.h

relations_write_C.$(DllSuf) : relations_write.C relations.h relations_load_C.$(DllSuf)
	$(BuildWithLib)

relations_read_C.$(DllSuf) : relations.h

relationsTest.log: relations_load_C.$(DllSuf)  relations_write_C.$(DllSuf)  relations_read_C.$(DllSuf) $(ROOTCORELIBS)
	$(CMDECHO)  root.exe -q -b -l relationsRun.C | sed 's|File  Size =.*\*||g' > relationsTest.log

relationsTest: relationsTest.log 
	$(CMDECHO) diff -bw relationsTest.ref relationsTest.log

mytest:
	$(CMDECHO) echo mytest > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


merging.log: merging.C

merging: merging.log
	$(TestDiff)

iobug.log : iobug_C.$(DllSuf)

iobug: iobug.log
	$(TestDiff)

treeBranch.log: userClass_C.$(DllSuf) treeBranch_C.$(DllSuf)

treeBranch: treeBranch.log
	$(TestDiff)

inheritAndOwn.log: test_Persistency0_C.$(DllSuf)  test_Persistency3_C.$(DllSuf)

inheritAndOwn: inheritAndOwn.log
	$(TestDiff)

BadBaseAddresses.log: test_Persistency1_C.$(DllSuf)

BadBaseAddresses: BadBaseAddresses.log
	$(TestDiff)

ursula: ursula.log
	$(TestDiff)
