# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log Event.$(ExeSuf) *.root EventDict*

#Set the list of target to make while testing
TEST_TARGETS += schemaTest

# adjust the location of Rules.mk as needed.
ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


mytest:
	$(CMDECHO) echo mytest > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

include $(ROOTSYS)/test/Makefile.arch

EVENTO        = Event.$(ObjSuf) EventDict.$(ObjSuf)
EVENTS        = Event.$(SrcSuf) EventDict.$(SrcSuf)
EVENTSO       = libEvent.$(DllSuf)

EVENTS2       = Event_2.$(SrcSuf) EventDict_2.$(SrcSuf) Event.$(ObjSuf)
EVENT2O       = Event_2.$(ObjSuf) EventDict_2.$(ObjSuf)
EVENT2SO      = libEvent_2.$(DllSuf)

EVENT         = Event$(ExeSuf)
ifeq ($(ARCH),win32)
EVENTLIB      = libEvent.lib
else
EVENTLIB      = $(EVENTSO)
endif

schemaRun.ref: schemaRun1.C Event.root 
	$(CMDECHO) root.exe -q -b -l schemaRun1.C 2>&1 \
	| grep -v fType | grep -v 'no dictionary for class' | grep -v Processing | grep -v TRefArray | grep -v TH1F \
	> schemaRun.ref

SCRIPTS = $(wildcard schemaRun*.C)
LOGFILES = $(subst .C,.log,$(SCRIPTS))
LOGFILES_TEST = $(subst .log,.test,$(LOGFILES))

#debug := $(shell echo $(LOGFILES_TEST) 1>&2)

$(LOGFILES) : %.log : %.C $(EVENT2SO) $(EVENTSO) 
	$(CMDECHO) root.exe -q -b -l  $< 2>&1 \
	| grep -v fType | grep -v 'no dictionary for class' | grep -v Processing | grep -v TRefArray | grep -v TH1F \
	> $@

$(LOGFILES_TEST) : %.test : %.log
	$(CMDECHO) diff -b schemaRun.ref $^

schemaTest: Event.root $(LOGFILES_TEST)
	$(CMDECHO) diff -b schemaRun.ref schemaRun1.log && \
	diff -b schemaRun.ref schemaRun2.log && \
	diff -b schemaRun.ref schemaRun3.log && \
	diff -b schemaRun.ref schemaRun4.log

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
		$(CMDECHO) $(LD) $(SOFLAGS) $(EVENTO) $(OutPutOpt) $(EVENTSO)
		$(CMDECHO) $(LD) -bundle -undefined suppress -Wl,-x $(LDFLAGS) $^ \
		   $(OutPutOpt) $(subst .$(DllSuf),.so,$@)
else
ifeq ($(ARCH),win32)
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
		@echo "$@ done"

$(EVENT):       $(EVENTSO) $(MAINEVENTO)
		$(CMDECHO) $(LD) $(LDFLAGS) $(MAINEVENTO) $(EVENTLIB) $(LIBS) \
		   $(OutPutOpt)$(EVENT)
		@echo "$@ done"

$(EVENT2SO):     $(EVENT2O)
ifeq ($(ARCH),aix)
		/usr/ibmcxx/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
else
ifeq ($(ARCH),aix5)
		/usr/vacpp/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
else
ifeq ($(ARCH),macosx)
# We need to make both the .dylib and the .so
		$(CMDECHO) $(LD) $(SOFLAGS) $(EVENTO) $(OutPutOpt) $(EVENTSO)
		$(CMDECHO) $(LD) -bundle -undefined suppress -Wl,-x $(LDFLAGS) $^ \
		   $(OutPutOpt) $(subst .$(DllSuf),.so,$@)
else
ifeq ($(ARCH),win32)
		bindexplib $* $^ > $*.def
		lib -nologo -MACHINE:IX86 $^ -def:$*.def \
		   $(OutPutOpt)$(EVENTLIB)
		$(CMDECHO) $(LD) $(SOFLAGS) $(LDFLAGS) $^ $*.exp $(LIBS) \
		   $(OutPutOpt)$@
else
		$(CMDECHO) $(LD) $(SOFLAGS) $(LDFLAGS) $^ $(OutPutOpt) $@ $(EXPLLINKLIBS)
endif
endif
endif
endif
		@echo "$@ done"


Event.root: $(EVENT)
	$(CMDECHO) ./Event 10 0 7 1 > Event_root.log
	@echo "$@ done"

