TEST_TARGETS = FileInfoTest
CLEAN_TARGETS = dict.* dictold.*

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

#test:
#	$(CMDECHO)echo $(CURDIR) has a failing test FileInfoTest


ifeq ($(FAIL),)
FileInfoTest:
	$(WarnFailTest)	
else
FileInfoTest: dict.o dictold.o
endif

dict.cxx: TFileInfo.h LinkDef.h $(ROOTCINT)
	$(CMDECHO)rootcint -f dict.cxx TFileInfo.h LinkDef.h

dictold.cxx: old.h TFileInfo.h oldLinkDef.h $(ROOTCINT)
	$(CMDECHO)rootcint -f dictold.cxx old.h TFileInfo.h oldLinkDef.h



