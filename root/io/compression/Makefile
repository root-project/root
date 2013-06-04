local_clean = *.root Event$(ExeSuf) libEvent.$(DllSuf) *.o EventDict.cxx EventDict.h *~ *.log *.$(ObjSuf) $(ALL_LIBRARIES)

CLEAN_TARGETS += $(local_clean)
TEST_TARGETS += compressionTest

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk
include $(ROOTTEST_HOME)/scripts/Event.mk

#.SUFFIXES: .cxx .o .so

CXXOPT:=-Z7
LDOPT:=-debug $ROOTSYS/lib/libXMLIO.so

compressionTest: ./Event$(ExeSuf) ./libEvent.$(DllSuf)
	$(CMDECHO)./Event 20   0 1  1  >  Event.write.log
	$(CMDECHO)./Event 20   0 1 20  >  Event.read.log
	$(CMDECHO)./Event 20 101 1  1  >> Event.write.log
	$(CMDECHO)./Event 20 101 1 20  >> Event.read.log
	$(CMDECHO)./Event 20 201 1  1  >> Event.write.log
	$(CMDECHO)./Event 20 201 1 20  >> Event.read.log
	$(CMDECHO)./Event 20 301 1  1  >> Event.write.log
	$(CMDECHO)./Event 20 301 1 20  >> Event.read.log
	$(CMDECHO)./Event 20   5 1  1  >> Event.write.log
	$(CMDECHO)./Event 20   5 1 20  >> Event.read.log
