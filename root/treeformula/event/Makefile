CLEAN_TARGETS += $(local_clean)
TEST_TARGETS += bigeventTest

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk
include $(ROOTSYS)/test/Makefile.arch

#.SUFFIXES: .cxx .o .so

LOPT:=-g3

#all: bigeventTest

bigeventTest: ./Event$(ExeSuf) ./libEvent.$(DllSuf)
	$(CMDECHO)./Event 10 1 1 1 > Event.write.log
	$(CMDECHO)./Event 10 1 1 20 > Event.read.log
ifeq ($(FAIL),)
	$(WarnFailTest)
	$(CMDECHO)echo This is due to the problem with loading libraries with stl containers after opening a file
else
	$(CMDECHO)./DrawTest.sh
endif


#Event : MainEvent.o Event.o EventDict.o
#	$(CMDECHO) $(CXX) $(OPT) $(OutPutOpt)Event MainEvent.o Event.o EventDict.o -L$(ROOTSYS)/lib \
#    -lCore -lCint -lTree -lHist -lMatrix -lPhysics -lGraf -lm -ldl 

#libEvent.$(DllSuf) :  Event.o EventDict.o
#	$(CMDECHO) $(CXX) $(OPT) $(SOFLAGS) Event.o EventDict.o $(OutPutOpt)libEvent.$(DllSuf) -L$(ROOTSYS)/lib -lPhysics

EventDict.cxx: Event.h EventLinkDef.h $(ROOTSYS)/bin/rootcint $(ROOTSYS)/lib/libCint.$(DllSuf)
	$(CMDECHO) rootcint -f EventDict.cxx -c Event.h EventLinkDef.h

local_clean = Event$(ExeSuf) libEvent.$(DllSuf) *.o EventDict.cxx EventDict.h *~ *.log *.$(DllSuf) *.root $(ALL_LIBRARIES)


EVENTO        = Event.$(ObjSuf) EventDict.$(ObjSuf)
EVENTS        = Event.$(SrcSuf) EventDict.$(SrcSuf)
EVENTSO       = libEvent.$(DllSuf)
EVENT         = Event$(ExeSuf)
ifeq ($(PLATFORM),win32)
EVENTLIB      = libEvent.lib
else
EVENTLIB      = $(EVENTSO)
endif
MAINEVENTO    = MainEvent.$(ObjSuf)
MAINEVENTS    = MainEvent.$(SrcSuf)

$(EVENTSO):     $(EVENTO)
ifeq ($(ARCH),aix)
		$(CMDECHO) /usr/ibmcxx/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
else
ifeq ($(ARCH),aix5)
		$(CMDECHO) /usr/vacpp/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
else
ifeq ($(PLATFORM),macosx)
# We need to make both the .dylib and the .so
		$(CMDECHO) $(LD) $(SOFLAGS) $(EVENTO) $(OutPutOpt) $(EVENTSO)
		$(CMDECHO) $(LD) -bundle -undefined $(UNDEFOPT) $(LDFLAGS) $^ \
		   $(OutPutOpt) $(subst .$(DllSuf),.so,$@)
else
ifeq ($(PLATFORM),win32)
		$(CMDECHO) bindexplib $* $^ > $*.def
		$(CMDECHO) lib -nologo -MACHINE:IX86 $^ -def:$*.def \
		   $(OutPutOpt)$(EVENTLIB)
		$(CMDECHO) $(LD) $(SOFLAGS) $(LDFLAGS) $^ $*.exp $(LIBS) \
		   $(OutPutOpt)$@
else
		$(CMDECHO) $(LD) $(SOFLAGS) $(LDFLAGS) $^ $(OutPutOpt) $@ $(EXPLLINKLIBS)
endif
endif
endif
endif

$(EVENT):       $(EVENTSO) $(MAINEVENTO)
		$(CMDECHO) $(LD) $(LDFLAGS) $(MAINEVENTO) $(EVENTLIB) $(LIBS) \
		   $(OutPutOpt)$(EVENT)

