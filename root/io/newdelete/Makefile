TEST_TARGETS += OperatorNewTest
CLEAN_TARGETS += *Dict* output *.o *.so

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


ifeq ($(ARCH),linuxicc)
CXXFLAGS += -wd873
endif

OperatorNewTest : output
	$(CMDECHO) diff -bw output good.output

output: Classes.$(DllSuf) runClasses.C
	$(CMDECHO) root.exe -l -b -q runClasses.C > output 2>&1

ClassesDict.cxx:  Classes.h LinkDef.h $(ROOTSYS)/bin/rootcint
	$(CMDECHO) rootcint -f ClassesDict.cxx -c Classes.h LinkDef.h

Classes.$(DllSuf): ClassesDict.$(ObjSuf) Classes.$(ObjSuf)
	$(BuildFromObjs)

Classes.o: Classes.cxx Classes.h
