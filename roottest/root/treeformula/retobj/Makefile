local_clean = Event libEvent.$(DllSuf) *.o EventDict.cxx EventDict.h *~ Event.new.split0.root  Event.root result $(ALL_LIBRARIES)

CLEAN_TARGETS += $(local_clean) *.log
TEST_TARGETS += enum retobjTest

ifeq ($(strip $(ROOTTEST_HOME)),)
   export ROOTTEST_HOME := $(shell git rev-parse --show-toplevel)/roottest/
   ifeq ($(strip $(ROOTTEST_HOME)),)
      export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
   endif
   ifeq ($(strip $(ROOTTEST_HOME)),)
      $(error The head of roottest was not found.  Set ROOTTEST_HOME)
   endif
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

enum.log: Enum_C.$(DllSuf)

include $(ROOTTEST_HOME)/scripts/Event.mk

.SUFFIXES: .cxx .o .so

retobjTest.log: Event.new.split0.root helper.C

ifneq ($(ClingWorkAroundCallfuncAndReturnByValue),)
retobjTest: CINT_VERSION = ClingWorkAroundCallfuncAndReturnByValue
endif

retobjTest: retobjTest.log
	$(TestDiffCintSpecific)

#retobjTest: Event.new.split0.root
#	$(CMDECHO)$(CALLROOTEXE) -q -l -b run.C > result 2>&1 && diff -bw result result.good

Event.new.split0.root: ./Event$(ExeSuf) ./libEvent.$(DllSuf)
#	$(CMDECHO)echo creating Event.new.split0.root
	$(CMDECHO)./Event 10 1 0 1 >/dev/null 2>&1  && cp Event.root Event.new.split0.root 

enum: enum.log
	$(TestDiffCintSpecific)

Event.$(ObjSuf): CXXFLAGS += -DClingWorkAroundCallfuncAndInline