# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES)

#Set the list of target to make while testing
TEST_TARGETS += mytest

# adjust the location of Rules.mk as needed.
include ../../Rules.mk

mytest:
	$(CMDECHO) echo mytest > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

schemaRun.ref:
	$(CMDECHO) root.exe -q -b -l schemaRun1.C \
	| grep -v Processing | grep -v TRefArray | grep -v fType | grep -v TH1F \
	> schemaRun.ref

schemaRun1.log:
	$(CMDECHO) root.exe -q -b -l schemaRun1.C \
	| grep -v Processing | grep -v TRefArray | grep -v fType | grep -v TH1F \
	> schemaRun1.log

schemaRun2.log:
	$(CMDECHO) root.exe -q -b -l schemaRun1.C \
	| grep -v Processing | grep -v TRefArray | grep -v fType | grep -v TH1F \
	> schemaRun2.log

schemaRun3.log:
	$(CMDECHO) root.exe -q -b -l schemaRun1.C \
	| grep -v Processing | grep -v TRefArray | grep -v fType | grep -v TH1F \
	> schemaRun3.log

schemaRun4.log:
	$(CMDECHO) root.exe -q -b -l schemaRun1.C \
	| grep -v Processing | grep -v TRefArray | grep -v fType | grep -v TH1F \
	> schemaRun4.log

schemaTest: schemaRun1.log schemaRun2.log schemaRun3.log schemaRun4.log
	$(CMDECHO) diff -b schemaRun.ref schemaRun1.log && \
	difff -b schemaRun.ref schemaRun2.log && \
	&& diff -b schemaRun.ref schemaRun3.log && \
	&& diff -b schemaRun.ref schemaRun4.log

include $(ROOTSYS)/test/Makefile.arch

EVENTO        = Event.$(ObjSuf) EventDict.$(ObjSuf)
EVENTS        = Event.$(SrcSuf) EventDict.$(SrcSuf)
EVENTSO       = libEvent.$(DllSuf)

EVENTS2       = Event_2.$(SrcSuf) EventDict_2.$(SrcSuf) Event.$(ObjSuf)
EVENT2O       = Event_2.$(ObjSuf) EventDict_2.$(ObjSuf)
EVENTS2O      = libEvent_2.$(DllSuf)

EVENT         = Event$(ExeSuf)
ifeq ($(ARCH),win32)
EVENTLIB      = libEvent.lib
else
EVENTLIB      = $(EVENTSO)
endif

MAINEVENTO    = MainEvent.$(ObjSuf)
MAINEVENTS    = MainEvent.$(SrcSuf)

Event.$(ObjSuf): Event.h
MainEvent.$(ObjSuf): Event.h

EventDict.$(SrcSuf): Event.h EventLinkDef.h
	@echo "Generating dictionary $@..."
	@rootcint -f $@ -c $^

EventDict_2.$(SrcSuf): Event_2.h EventLinkDef.h
	@echo "Generating dictionary $@..."
	@rootcint -f $@ -c $^

$(EVENTSO):     $(EVENTO)
ifeq ($(ARCH),aix)
		/usr/ibmcxx/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
else
ifeq ($(ARCH),aix5)
		/usr/vacpp/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
else
ifeq ($(ARCH),macosx)
# We need to make both the .dylib and the .so
		$(LD) $(SOFLAGS) $(EVENTO) $(OutPutOpt) $(EVENTSO)
		$(LD) -bundle -undefined suppress -Wl,-x $(LDFLAGS) $^ \
		   $(OutPutOpt) $(subst .$(DllSuf),.so,$@)
else
ifeq ($(ARCH),win32)
		bindexplib $* $^ > $*.def
		lib -nologo -MACHINE:IX86 $^ -def:$*.def \
		   $(OutPutOpt)$(EVENTLIB)
		$(LD) $(SOFLAGS) $(LDFLAGS) $^ $*.exp $(LIBS) \
		   $(OutPutOpt)$@
else
		$(LD) $(SOFLAGS) $(LDFLAGS) $^ $(OutPutOpt) $@ $(EXPLLINKLIBS)
endif
endif
endif
endif
		@echo "$@ done"

$(EVENT):       $(EVENTSO) $(MAINEVENTO)
		$(LD) $(LDFLAGS) $(MAINEVENTO) $(EVENTLIB) $(LIBS) \
		   $(OutPutOpt)$(EVENT)
		@echo "$@ done"

$(EVENTS2O):     $(EVENT2O)
ifeq ($(ARCH),aix)
		/usr/ibmcxx/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
else
ifeq ($(ARCH),aix5)
		/usr/vacpp/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
else
ifeq ($(ARCH),macosx)
# We need to make both the .dylib and the .so
		$(LD) $(SOFLAGS) $(EVENTO) $(OutPutOpt) $(EVENTSO)
		$(LD) -bundle -undefined suppress -Wl,-x $(LDFLAGS) $^ \
		   $(OutPutOpt) $(subst .$(DllSuf),.so,$@)
else
ifeq ($(ARCH),win32)
		bindexplib $* $^ > $*.def
		lib -nologo -MACHINE:IX86 $^ -def:$*.def \
		   $(OutPutOpt)$(EVENTLIB)
		$(LD) $(SOFLAGS) $(LDFLAGS) $^ $*.exp $(LIBS) \
		   $(OutPutOpt)$@
else
		$(LD) $(SOFLAGS) $(LDFLAGS) $^ $(OutPutOpt) $@ $(EXPLLINKLIBS)
endif
endif
endif
endif
		@echo "$@ done"


Event.root: $(EVENT)
	$(CMDECHO) ./Event 10 0 7 1 > Event_root.log

