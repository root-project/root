EventDict.cxx: Event.h EventLinkDef.h $(ROOTCINT) 
	$(CMDECHO) rootcint -f EventDict.cxx Event.h EventLinkDef.h

EVENTO        = Event.$(ObjSuf) EventDict.$(ObjSuf)
EVENTS        = Event.$(SrcSuf) EventDict.$(SrcSuf)
EVENTSO       = libEvent.$(DllSuf)
EVENT         = Event$(ExeSuf)
EVENTLIB      = libEvent.$(LibSuf)
MAINEVENTO    = MainEvent.$(ObjSuf)
MAINEVENTS    = MainEvent.$(SrcSuf)
LIBS          = $(ROOTLIBS)

ifeq ($(findstring clean,$(MAKECMDGOALS)),)
-include Event.d
-include MainEvent.d
-include EventDict.d
endif

ifeq ($(PLATFORM),macosx)
ifeq ($(MACOSX_MINOR),) 
  export MACOSX_MINOR := $(shell sw_vers | sed -n 's/ProductVersion://p' | cut -d . -f 2)
endif
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
$(EVENTO): %.$(ObjSuf): %.d

$(MAINEVENTO): MainEvent.d

$(EVENTSO):     $(EVENTO) $(ROOTCORELIBS)
ifeq ($(ARCH),aix)
		$(CMDECHO) /usr/ibmcxx/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
else
ifeq ($(ARCH),aix5)
		$(CMDECHO) /usr/vacpp/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
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

$(EVENT):       $(EVENTSO) $(MAINEVENTO) $(ROOTCORELIBS)
		$(CMDECHO) $(LD) $(LDFLAGS) $(MAINEVENTO) $(EVENTLIB) $(LIBS) \
		   $(OutPutOpt)$(EVENT)
