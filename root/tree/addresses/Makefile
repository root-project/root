# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += 

#Set the list of target to make while testing
TEST_TARGETS += mytest headerTest clonesATest

# adjust the location of Rules.mk as needed.
include ../../Rules.mk

header_h.$(DllSuf): header.h buildHeader.C
	$(CMDECHO) root.exe -q -b -l buildHeader.C

headerTest.log: runHeader.C header_h.$(DllSuf)
	$(CMDECHO) root.exe -q -b -l runHeader.C > headerTest.log 2>&1

headerTest: headerTest.log
	$(CMDECHO) diff headerTest.log headerTest.ref

clonesA.log: clonesA_rw.C clonesA_Event.h clonesA_Event.cxx
	$(CMDECHO) root.exe -q -b -l clonesA_rw.C > clonesA.log 2>&1

clonesATest: clonesA.log
	$(CMDECHO) test ! `(grep exception clonesA.log > /dev/null) && echo $?`

mytest:
	$(CMDECHO) echo mytest > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


