local_clean = Event$(ExeSuf) libEvent.$(DllSuf) *.o EventDict.cxx EventDict.h *~ *.log *.$(DllSuf) *.root $(ALL_LIBRARIES)

CLEAN_TARGETS += $(local_clean)
TEST_TARGETS += bigeventTest

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk
include $(ROOTTEST_HOME)/scripts/Event.mk

#.SUFFIXES: .cxx .o .so

LOPT:=-g3

#all: bigeventTest

bigeventTest: ./Event$(ExeSuf) ./libEvent.$(DllSuf)
	$(CMDECHO)./Event 10 1 1 1 > Event.write.log
	$(CMDECHO)./Event 10 1 1 20 > Event.read.log
ifeq ($(FAIL),)
	$(WarnFailTest)
#	$(CMDECHO)echo Warning this is due to the problem with loading libraries with stl containers after opening a file
else
	$(CMDECHO)./DrawTest.sh
endif



