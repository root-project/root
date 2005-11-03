CLEAN_TARGETS += $(local_clean)
TEST_TARGETS += enum retobjTest

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

enum.log: Enum_C.$(DllSuf)

include $(ROOTTEST_HOME)/scripts/Event.mk

.SUFFIXES: .cxx .o .so

retobjTest: Event.new.split0.root
	$(CMDECHO)root.exe -q -l -b run.C > result 2>&1 && diff -bw result result.good

Event.new.split0.root: ./Event$(ExeSuf) ./libEvent.$(DllSuf)
#	$(CMDECHO)echo creating Event.new.split0.root
	$(CMDECHO)./Event 10 1 0 1 >/dev/null 2>&1  && cp Event.root Event.new.split0.root 

local_clean = Event libEvent.$(DllSuf) *.o EventDict.cxx EventDict.h *~ Event.new.split0.root  Event.root result $(ALL_LIBRARIES)

enum: enum.log
	$(TestDiff)
