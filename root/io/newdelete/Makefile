TEST_TARGETS += OperatorNewTest
CLEAN_TARGETS += *Dict* output *.o *.so

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


ifeq ($(ARCH),linuxicc)
CXXFLAGS += -wd873
endif

OperatorNewTest : Classes.$(DllSuf)
	$(CMDECHO) echo 'gROOT->ProcessLine(".L Classes.so");gSystem->Exit(!Classes());'  | root.exe -l -b >& output  && diff output good.output

ClassesDict.cxx:  Classes.h LinkDef.h $(ROOTSYS)/bin/rootcint
	$(CMDECHO) rootcint -f ClassesDict.cxx -c Classes.h LinkDef.h

Classes.$(DllSuf): ClassesDict.$(ObjSuf) Classes.$(ObjSuf)
	$(CMDECHO) $(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" "" $@ \
		   ClassesDict.$(ObjSuf) Classes.$(ObjSuf)

Classes.o: Classes.cxx Classes.h
