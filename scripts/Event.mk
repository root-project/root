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

ifeq ($(findstring clean,$(MAKECMDGOALS)),)
-include Event.d
-include MainEvent.d
-include EventDict.d
endif

Event.d: Event.cxx Event.h
	@touch Event.dd; rmkdepend -f Event.dd -I$(ROOTSYS)/include Event.cxx 2>/dev/null && \
	cat Event.dd | sed -e s/Event\\\.o/Event\\\.$(ObjSuf)/g > Event.d; rm Event.dd Event.dd.bak

MainEvent.d: MainEvent.cxx Event.h
	@touch MainEvent.dd; rmkdepend -f MainEvent.dd -I$(ROOTSYS)/include MainEvent.cxx 2>/dev/null && \
	cat MainEvent.dd | sed -e s/MainEvent\\\.o/MainEvent\\\.$(ObjSuf)/g > MainEvent.d; rm MainEvent.dd MainEvent.dd.bak

EventDict.d: EventDict.cxx Event.h
	@touch EventDict.dd; rmkdepend -f EventDict.dd -I$(ROOTSYS)/include EventDict.cxx 2>/dev/null && \
	cat EventDict.dd | sed -e s/EventDict\\\.o/EventDict\\\.$(ObjSuf)/g > EventDict.d; rm EventDict.dd EventDict.dd.bak

Event.$(ObjSuf): $(ROOTCORELIBS)
EventDict.$(ObjSuf): $(ROOTCORELIBS)
MainEvent.$(ObjSuf): $(ROOTCORELIBS)
$(EVENTO): Event.d EventDict.d

$(MAINEVENTO): MainEvent.d

$(EVENTSO):     $(EVENTO) $(ROOTCORELIBS)
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
		$(CMDECHO) bindexplib $* $(EVENTO) > $*.def
		$(CMDECHO) lib -nologo -MACHINE:IX86 $(EVENTO) -def:$*.def \
		   $(OutPutOpt)$(EVENTLIB)
		$(CMDECHO) $(LD) $(SOFLAGS) $(LDFLAGS) $(EVENTO) $*.exp $(LIBS) \
		   $(OutPutOpt)$@
else
		$(CMDECHO) $(LD) $(SOFLAGS) $(LDFLAGS) $^ $(OutPutOpt) $@ $(EXPLLINKLIBS)
endif
endif
endif
endif

$(EVENT):       $(EVENTSO) $(MAINEVENTO) $(ROOTCORELIBS)
		$(CMDECHO) $(LD) $(LDFLAGS) $(MAINEVENTO) $(EVENTLIB) $(LIBS) \
		   $(OutPutOpt)$(EVENT)
