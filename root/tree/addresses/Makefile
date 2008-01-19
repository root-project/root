# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += *.clog *.log $(ALL_LIBRARIES) *.d 
CLEAN_TARGETS += Embedded.root RootRelations.root clonesA_Event.root 
CLEAN_TARGETS += data.root dummy*.root problem.root tree.root merge.root
CLEAN_TARGETS += configtest.root

#Set the list of target to make while testing 
TEST_TARGETS += ursula inheritAndOwn BadBaseAddresses treeBranch iobug \
	merging headerTest clonesATest EmbeddedTest inheritTest \
	relationsTest suetest Jantest deepnesT baseString prova \
	augertest longlong

# adjust the location of Rules.mk as needed.
ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

inherit_C.$(DllSuf): inherit.C buildInherit.C

inheritTest.log: runInherit.C inherit_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l runInherit.C > inheritTest.log 2>&1

inheritTest: inheritTest.log
	$(CMDECHO) diff --strip-trailing-cr -ubw inheritTest.log inheritTest.ref

header_h.$(DllSuf): header.h buildHeader.C

headerTest.log: runHeader.C header_h.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l runHeader.C > headerTest.log 2>&1

headerTest: headerTest.log
	$(CMDECHO) diff --strip-trailing-cr -ubw headerTest.log headerTest.ref

clonesA_Event_cxx.$(DllSuf): clonesA_Event.h clonesA_Event.cxx

clonesA.log: clonesA_rw.C clonesA_Event_cxx.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l clonesA_rw.C > clonesA.log 2>&1

clonesATest: clonesA.log
	$(CMDECHO) test ! `(grep exception clonesA.log > /dev/null) && echo $?`

Embedded_load_C.$(DllSuf) : buildEmbedded.C Embedded_load.C Embedded_objects.cpp Embedded_objects.h

Embedded.root: Embedded_load_C.$(DllSuf) embwrite.C
	$(CMDECHO) $(CALLROOTEXE) -q -b -l embwrite.C > embwrite.log

relations_load_C.$(DllSuf) : relations.h relationsLinkdef.h

relations_write_C.$(DllSuf) : relations.h

relations_read_C.$(DllSuf) : relations.h

EmbeddedTest.log: Embedded_load_C.$(DllSuf) Embedded.root embread.C
	$(CMDECHO) $(CALLROOTEXE) -q -b -l embread.C | sed -e "s/e+006/e+06/g" > EmbeddedTest.log

EmbeddedTest: EmbeddedTest.log
	$(CMDECHO) diff --strip-trailing-cr -ubw EmbeddedTest.ref EmbeddedTest.log

relations_load_C.$(DllSuf) : relations.h

relations_write_C.$(DllSuf) : relations_write.C relations.h relations_load_C.$(DllSuf)
	$(BuildWithLib)

relations_read_C.$(DllSuf) : relations.h

relationsTest.log: relations_load_C.$(DllSuf)  relations_write_C.$(DllSuf)  relations_read_C.$(DllSuf) $(ROOTCORELIBS)
	$(CMDECHO)  $(CALLROOTEXE) -q -b -l relationsRun.C | sed 's|File  Size =.*\*||g' > relationsTest.log

relationsTest: relationsTest.log 
	$(CMDECHO) diff --strip-trailing-cr -ubw relationsTest.ref relationsTest.log

mytest:
	$(CMDECHO) echo mytest > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


merging.log: merging.C

merging: merging.log
	$(Testdiff --strip-trailing-cr)

iobug.log : iobug_C.$(DllSuf)

iobug: iobug.log
	$(Testdiff --strip-trailing-cr)

treeBranch.log: userClass_C.$(DllSuf) treeBranch_C.$(DllSuf)

treeBranch: treeBranch.log
	$(Testdiff --strip-trailing-cr)

inheritAndOwn.log: test_Persistency0_C.$(DllSuf)  test_Persistency3_C.$(DllSuf)

inheritAndOwn: inheritAndOwn.log
	$(Testdiff --strip-trailing-cr)

BadBaseAddresses.log: test_Persistency1_C.$(DllSuf)

BadBaseAddresses: BadBaseAddresses.log
	$(Testdiff --strip-trailing-cr)

ursula: ursula.log
	$(Testdiff --strip-trailing-cr)

ConfigRecord_cxx.$(DllSuf): ConfigRecord.cxx ConfigRecord.h sueloader_C.$(DllSuf)
	$(BuildWithLib)	

configtest.root: ConfigRecord_cxx.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l suetestwrt.C > configtest.log 2>&1
	$(CMDECHO) diff --strip-trailing-cr -b configtest.ref configtest.log

suetest.log: configtest.root
	$(CMDECHO) $(CALLROOTEXE) -q -b -l suetestrd.C > suetest.log 2>&1

suetest: suetest.log
	$(Testdiff --strip-trailing-cr)

Jantest.log: JansEvent_C.$(DllSuf)

Jantest: Jantest.log
	$(Testdiff --strip-trailing-cr)

deepnest: deepnest.clog
	$(Testdiff --strip-trailing-cr)

baseString: baseString.clog
	$(Testdiff --strip-trailing-cr)

prova: prova.log
	$(Testdiff --strip-trailing-cr)

augertest: augertest.log
	$(Testdiff --strip-trailing-cr)

longlong: longlong.clog
	$(Testdiff --strip-trailing-cr)
