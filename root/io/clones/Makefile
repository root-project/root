TEST_TARGETS += clone
CLEAN_TARGETS += RtObjDict.* $(ALL_LIBRARIES) tester

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif
include $(ROOTTEST_HOME)/scripts/Rules.mk

ifeq ($(FAIL),)
clone:
	$(WarnFailTest)
else
clone: tester
endif
	

RtObjDict.C: RtObj.h RtObjLinkDef.h $(ROOTCINT)
	$(CMDECHO)rootcint -f RtObjDict.C -c RtObj.h RtObjLinkDef.h

.C.o:
	$(CMDECHO)$(CXX) $(CXXFLAGS) -c $<

tester: tester.o RtObjDict.o
	$(CMDECHO)$(CXX) $(CXXFLAGS) $(LDFLAGS) -o tester tester.o RtObjDict.o

