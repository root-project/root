TEST_TARGETS += OperatorNewTest
CLEAN_TARGETS += *Dict* output *.o *.so *.log $(ALL_LIBRARIES)

ifeq ($(strip $(ROOTTEST_HOME)),)
   export ROOTTEST_HOME := $(shell git rev-parse --show-toplevel)/
   ifeq ($(strip $(ROOTTEST_HOME)),)
      export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
   endif
   ifeq ($(strip $(ROOTTEST_HOME)),)
      $(error The head of roottest was not found.  Set ROOTTEST_HOME)
   endif
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


ifeq ($(ARCH),linuxicc)
CXXFLAGS += -wd873
endif

OperatorNewTest : output
	$(CMDECHO) diff -bw output good.output

output: Classes.$(DllSuf) runClasses.C  $(ROOTCORELIBS) $(ROOTV) 
	$(CMDECHO) $(CALLROOTEXE) -l -b -q runClasses.C > output 2>&1

ClassesDict.cxx:  Classes.h LinkDef.h $(ROOTCINT)
	$(CMDECHO) rootcint -f ClassesDict.cxx Classes.h LinkDef.h

Classes.$(DllSuf): ClassesDict.$(ObjSuf) Classes.$(ObjSuf) $(ROOTCORELIBS) $(ROOTV) $(ROOTCINT)
	$(BuildFromObjs)

Classes.$(ObjSuf): Classes.cxx Classes.h $(ROOTCORELIBS) $(ROOTV) $(ROOTCINT)
