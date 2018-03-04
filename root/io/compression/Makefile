local_clean = *.root Event$(ExeSuf) libEvent.$(DllSuf) *.o EventDict.cxx EventDict.h *~ *.log *.$(ObjSuf) $(ALL_LIBRARIES)

CLEAN_TARGETS += $(local_clean)
TEST_TARGETS += compressionTest

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
include $(ROOTTEST_HOME)/scripts/Event.mk

#.SUFFIXES: .cxx .o .so

CXXOPT:=-Z7
LDOPT:=-debug $ROOTSYS/lib/libXMLIO.so

compressionTest: ./Event$(ExeSuf) ./libEvent.$(DllSuf)
	$(CMDECHO)./Event 20   0 1  1  >  Event.write.log
	$(CMDECHO)./Event 20   0 1 20  >  Event.read.log
	$(CMDECHO)./Event 20 101 1  1  >> Event.write.log
	$(CMDECHO)./Event 20 101 1 20  >> Event.read.log
	$(CMDECHO)./Event 20 208 1  1  >> Event.write.log
	$(CMDECHO)./Event 20 208 1 20  >> Event.read.log
	$(CMDECHO)./Event 20 301 1  1  >> Event.write.log
	$(CMDECHO)./Event 20 301 1 20  >> Event.read.log
	$(CMDECHO)./Event 20 404 1  1  >> Event.write.log
	$(CMDECHO)./Event 20 404 1 20  >> Event.read.log
	$(CMDECHO)./Event 20   5 1  1  >> Event.write.log
	$(CMDECHO)./Event 20   5 1 20  >> Event.read.log
