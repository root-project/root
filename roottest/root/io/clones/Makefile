TEST_TARGETS += clone
CLEAN_TARGETS += RtObjDict.* $(ALL_LIBRARIES) tester

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

ifeq ($(FAIL),)
clone:
	$(WarnFailTest)
else
clone: tester
endif


RtObjDict.C: RtObj.h RtObjLinkDef.h $(ROOTCINT)
	$(CMDECHO)rootcint -f RtObjDict.C RtObj.h RtObjLinkDef.h

.C.o:
	$(CMDECHO)$(CXX) $(CXXFLAGS) -c $<

tester: tester.o RtObjDict.o
	$(CMDECHO)$(CXX) $(CXXFLAGS) $(LDFLAGS) -o tester tester.o RtObjDict.o

