include $(ROOTSYS)/test/Makefile.arch

EventDict.cxx: Event.h EventLinkDef.h $(ROOTSYS)/bin/rootcint $(ROOTSYS)/lib/libCint.$(DllSuf)
	$(CMDECHO) rootcint -f EventDict.cxx -c Event.h EventLinkDef.h

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

$(EVENTO) : Event.h

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
