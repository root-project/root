TEST_TARGETS += OperatorNewTest
CLEAN_TARGETS += *Dict* output *.o *.so

include ../../Rules.mk


OperatorNewTest : Classes.$(DllSuf)
	@echo 'gROOT->ProcessLine(".L Classes.so");gSystem->Exit(!Classes());'  | root.exe -l -b > output 2>&1 && diff output good.output

ClassesDict.cxx:  Classes.h LinkDef.h $(ROOTSYS)/bin/rootcint
	rootcint -f ClassesDict.cxx -c Classes.h LinkDef.h

Classes.$(DllSuf): ClassesDict.$(ObjSuf) Classes.$(ObjSuf)
	@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" "" $@ \
		   ClassesDict.$(ObjSuf) Classes.$(ObjSuf)

Classes.o: Classes.cxx Classes.h