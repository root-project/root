# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log Event$(ExeSuf) *.root EventDict* schemaRun.ref

#Set the list of target to make while testing
TEST_TARGETS += schemaTest

# adjust the location of Rules.mk as needed.
ifeq ($(strip $(ROOTTEST_HOME)),)
   export ROOTTEST_HOME := $(shell git rev-parse --show-toplevel)/roottest/
   ifeq ($(strip $(ROOTTEST_HOME)),)
      export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
   endif
   ifeq ($(strip $(ROOTTEST_HOME)),)
      $(error The head of roottest was not found.  Set ROOTTEST_HOME)
   endif
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

LIBS = $(ROOTLIBS)

EVENTO        = Event.$(ObjSuf) EventDict.$(ObjSuf)
EVENTS        = Event.$(SrcSuf) EventDict.$(SrcSuf)
EVENTSO       = libEvent.$(DllSuf)

EVENTS2       = Event_2.$(SrcSuf) EventDict_2.$(SrcSuf) Event.$(ObjSuf)
EVENT2O       = Event_2.$(ObjSuf) EventDict_2.$(ObjSuf)
EVENT2SO      = libEvent_2.$(DllSuf)

EVENT         = Event$(ExeSuf)
EVENTLIB      = libEvent.$(LibSuf)
EVENT2LIB     = libEvent_2.$(LibSuf)

ifeq ($(PLATFORM),macosx)
ifeq ($(MACOSX_MINOR),)
  export MACOSX_MINOR := $(shell sw_vers | sed -n 's/ProductVersion://p' | cut -d . -f 2)
endif
endif

schemaRun.ref: schemaRun1.C Event.root 
	$(CMDECHO) $(CALLROOTEXE) -q -b -l schemaRun1.C 2>&1 \
	| grep -a -v fType | grep -v 'no dictionary for class' | grep -v Processing | grep -v TRefArray | grep -v TH1F | sed -e 's/45[0-9],/45,/g'  \
	| sed -e 's/\([0-9]\.[0-9][0-9][0-9][0-9]\)[0-9]*/\1/g'| sed -e 's:0x[^/w]*:0xRemoved:' | sed -e 's:33554456*:50331672:g' \
	| sed -e 's:33554432:50331648:g' \
	> schemaRun.ref

SCRIPTS = $(wildcard schemaRun*.C)
LOGFILES = $(subst .C,.log,$(SCRIPTS))
LOGFILES_TEST = $(subst .log,.test,$(LOGFILES))

#debug := $(shell echo $(LOGFILES_TEST) 1>&2)

$(LOGFILES) : %.log : %.C $(EVENT2SO) $(EVENTSO) Event.root schemaRun.ref
	$(CMDECHO) $(CALLROOTEXE) -q -b -l  $< 2>&1 \
	| grep -a -v fType | grep -v 'no dictionary for class' | grep -v Processing | grep -v TRefArray | grep -v TH1F | sed -e 's/45[0-9],/45,/g'  \
	| sed -e 's/\([0-9]\.[0-9][0-9][0-9][0-9]\)[0-9]*/\1/g'| sed -e 's:0x[^/w]*:0xRemoved:' | sed -e 's:33554456*:50331672:g' \
	| sed -e 's:33554432:50331648:g' \
	> $@

$(LOGFILES_TEST) : %.test : %.log
	$(CMDECHO) diff --strip-trailing-cr -b schemaRun.ref $^

schemaTest: Event.root $(LOGFILES_TEST)
	$(CMDECHO) diff --strip-trailing-cr -b schemaRun.ref schemaRun1.log && \
	diff --strip-trailing-cr -b schemaRun.ref schemaRun2.log && \
	diff --strip-trailing-cr -b schemaRun.ref schemaRun3.log && \
	diff --strip-trailing-cr -b schemaRun.ref schemaRun4.log

MAINEVENTO    = MainEvent.$(ObjSuf)
MAINEVENTS    = MainEvent.$(SrcSuf)

Event.$(ObjSuf): Event.h
MainEvent.$(ObjSuf): Event.h

EventDict.$(SrcSuf): Event.h EventLinkDef.h
#	$(CMDECHO)echo "Generating dictionary $@..."
	$(CMDECHO)rootcint -f $@ $^

EventDict_2.$(SrcSuf): Event_2.h EventLinkDef.h
#	$(CMDECHO)echo "Generating dictionary $@..."
	$(CMDECHO)rootcint -f $@ $^

$(EVENTSO):     $(EVENTO)
ifeq ($(ARCH),aix)
		$(CMDECHO)/usr/ibmcxx/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
else
ifeq ($(ARCH),aix5)
		$(CMDECHO)/usr/vacpp/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
else
ifeq ($(ARCH),win32)
		$(CMDECHO) bindexplib $* $^ > $*.def
		$(CMDECHO) lib -nologo -MACHINE:IX86 $^ -def:$*.def \
		   $(OutPutOpt)$(EVENTLIB)
		$(CMDECHO) $(LD) $(SOFLAGS) $(LDFLAGS) $^ $*.exp $(LIBS) \
		   $(OutPutOpt)$@
else
		$(CMDECHO) $(LD) $(SOFLAGS) $(LDFLAGS) $^ $(OutPutOpt) $@ $(LIBS)
endif
endif
endif
#		$(CMDECHO)echo "$@ done"

$(EVENT):       $(EVENTSO) $(MAINEVENTO)
		$(CMDECHO) $(LD) $(LDFLAGS) $(MAINEVENTO) $(EVENTLIB) $(LIBS) \
		   $(OutPutOpt)$(EVENT)
#		$(CMDECHO)echo "$@ done"

$(EVENT2SO):     $(EVENT2O)
ifeq ($(ARCH),aix)
		$(CMDECHO)/usr/ibmcxx/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
else
ifeq ($(ARCH),aix5)
		$(CMDECHO)/usr/vacpp/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
else
ifeq ($(ARCH),win32)
		$(CMDECHO)bindexplib $* $^ > $*.def
		$(CMDECHO)lib -nologo -MACHINE:IX86 $^ -def:$*.def \
		   $(OutPutOpt)$(EVENT2LIB)
		$(CMDECHO) $(LD) $(SOFLAGS) $(LDFLAGS) $^ $*.exp $(LIBS) \
		   $(OutPutOpt)$@
else
		$(CMDECHO) $(LD) $(SOFLAGS) $(LDFLAGS) $^ $(OutPutOpt) $@ $(LIBS)
endif
endif
endif
#		$(CMDECHO)echo "$@ done"


Event.root: $(EVENT)
	$(CMDECHO) ./Event 10 0 7 1 > Event_root.log
#	$(CMDECHO)echo "$@ done"

