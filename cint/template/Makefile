#Set the list of files to be delete by clean:
CLEAN_TARGETS += *.log *.root $(ALL_LIBRARIES)

#Set the list of target to make while testing
TEST_TARGETS += operator1 simpleSeparateDict forwardDecl constructorTest typenameTest

SHELL = /bin/bash

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

SUBDIRS = $(shell $(ROOTTEST_HOME)scripts/subdirectories .)

include $(ROOTTEST_HOME)/scripts/Rules.mk



typenameTest:
	$(CMDECHO)rootcint -f typenameTestDict.C -c -D__CINT2__ typenameTest.C linkdef.h 
ifeq ($(FAIL),)
	$(WarnFailTest)
else
	$(CMDECHO)rootcint -f typenameTestDict.C -c typenameTest.C linkdef.h 
endif

operator1:
	$(CMDECHO)rm  -f operator1.cint.out operator1.comp.out
	$(CMDECHO)(echo '.x smalltest.C' | root.exe -l -b ) > operator1.cint.out 2>&1
	$(CMDECHO)(echo '.x smalltest.C++' | root.exe -l -b ) > operator1.comp.out 2>&1
ifeq ($(FAIL),)
	$(WarnFailTest)
else
	$(CMDECHO)cat operator1.comp.out | grep -v 'not declared' |grep -v 'Info in <ACLiC>:' | diff - operator1.cint.out
endif


simpleSeparateDict:
	$(CMDECHO)rootcint -f masterDict.C -c master.h masterLinkDef.h
	$(CMDECHO)rootcint -f slaveDict.C -c slave.h slaveLinkDef.h

forwardDecl: forwardDict.C

forwardDict.C: forward.C 
	$(CMDECHO)rootcint -f forwardDict.C -c forward.C linkdef.h


constructorDict.C: constructor.hh 
	$(CMDECHO) rootcint -f constructorDict.C -c constructor.hh linkdef.h

constructorTest: constructorDict.o

