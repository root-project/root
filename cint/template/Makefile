SHELL = /bin/bash

SUBDIRS = separateDict separateDictNamespace

include ../../root/Rules.mk

test: operator1 simpleSeparateDict forwardDecl constructorTest typenameTest
	true;

typenameTest:
	rootcint -f typenameTestDict.C -c -D__CINT2__ typenameTest.C linkdef.h 
ifeq ($(FAIL),)
	$(WarnFailTest)
else
	rootcint -f typenameTestDict.C -c typenameTest.C linkdef.h 
endif

operator1:
	@rm  -f operator1.cint.out operator1.comp.out
	@(echo '.x smalltest.C' | root.exe -l -b ) > operator1.cint.out 2>&1
	@(echo '.x smalltest.C++' | root.exe -l -b ) > operator1.comp.out 2>&1
ifeq ($(FAIL),)
	$(WarnFailTest)
else
	@cat operator1.comp.out | grep -v 'not declared' |grep -v 'Info in <ACLiC>:' | diff - operator1.cint.out
endif


simpleSeparateDict:
	rootcint -f masterDict.C -c master.h masterLinkDef.h
	rootcint -f slaveDict.C -c slave.h slaveLinkDef.h

forwardDecl: forwardDict.C

forwardDict.C: forward.C 
	rootcint -f forwardDict.C -c forward.C linkdef.h


constructorDict.C: constructor.hh 
	$(CMDECHO) rootcint -f constructorDict.C -c constructor.hh linkdef.h

constructorTest: constructorDict.o

