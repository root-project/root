# Makefile for the ROOT test programs.
# This Makefile shows nicely how to compile and link applications
# using the ROOT libraries on all supported platforms.
#
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

include Makefile.arch

#------------------------------------------------------------------------------

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

HWORLDO       = hworld.$(ObjSuf)
HWORLDS       = hworld.$(SrcSuf)
HWORLD        = hworld$(ExeSuf)

CTORTUREO     = ctorture.$(ObjSuf)
CTORTURES     = ctorture.$(SrcSuf)
CTORTURE      = ctorture$(ExeSuf)

HSIMPLEO      = hsimple.$(ObjSuf)
HSIMPLES      = hsimple.$(SrcSuf)
HSIMPLE       = hsimple$(ExeSuf)

MINEXAMO      = minexam.$(ObjSuf)
MINEXAMS      = minexam.$(SrcSuf)
MINEXAM       = minexam$(ExeSuf)

TSTRINGO      = tstring.$(ObjSuf)
TSTRINGS      = tstring.$(SrcSuf)
TSTRING       = tstring$(ExeSuf)

TCOLLEXO      = tcollex.$(ObjSuf)
TCOLLEXS      = tcollex.$(SrcSuf)
TCOLLEX       = tcollex$(ExeSuf)

TCOLLBMO      = tcollbm.$(ObjSuf)
TCOLLBMS      = tcollbm.$(SrcSuf)
TCOLLBM       = tcollbm$(ExeSuf)

VVECTORO      = vvector.$(ObjSuf)
VVECTORS      = vvector.$(SrcSuf)
VVECTOR       = vvector$(ExeSuf)

VMATRIXO      = vmatrix.$(ObjSuf)
VMATRIXS      = vmatrix.$(SrcSuf)
VMATRIX       = vmatrix$(ExeSuf)

STRESSLO      = stressLinear.$(ObjSuf)
STRESSLS      = stressLinear.$(SrcSuf)
STRESSL       = stressLinear$(ExeSuf)

VLAZYO        = vlazy.$(ObjSuf)
VLAZYS        = vlazy.$(SrcSuf)
VLAZY         = vlazy$(ExeSuf)

GUITESTO      = guitest.$(ObjSuf)
GUITESTS      = guitest.$(SrcSuf)
GUITEST       = guitest$(ExeSuf)

GUIVIEWERO    = guiviewer.$(ObjSuf) guiviewerDict.$(ObjSuf)
GUIVIEWERS    = guiviewer.$(SrcSuf) guiviewerDict.$(SrcSuf)
GUIVIEWER     = guiviewer$(ExeSuf)

HELLOO        = Hello.$(ObjSuf) HelloDict.$(ObjSuf)
HELLOS        = Hello.$(SrcSuf) HelloDict.$(SrcSuf)
HELLOSO       = Hello.$(DllSuf)

ACLOCKO       = Aclock.$(ObjSuf) AclockDict.$(ObjSuf)
ACLOCKS       = Aclock.$(SrcSuf) AclockDict.$(SrcSuf)
ACLOCKSO      = Aclock.$(DllSuf)

TETRISO       = Tetris.$(ObjSuf) TetrisDict.$(ObjSuf)
TETRISS       = Tetris.$(SrcSuf) TetrisDict.$(SrcSuf)
TETRISSO      = Tetris.$(DllSuf)

STRESSO       = stress.$(ObjSuf)
STRESSS       = stress.$(SrcSuf)
STRESS        = stress$(ExeSuf)

STRESSGEOMO   = stressgeom.$(ObjSuf)
STRESSGEOMS   = stressgeom.$(SrcSuf)
STRESSGEOM    = stressgeom$(ExeSuf)

TBENCHO       = TBench.$(ObjSuf) TBenchDict.$(ObjSuf)
TBENCHS       = TBench.$(SrcSuf) TBenchDict.$(SrcSuf)
TBENCHSO      = TBench.$(DllSuf)

BENCHO        = bench.$(ObjSuf)
BENCHS        = bench.$(SrcSuf)
BENCH         = bench$(ExeSuf)

TESTBITSO     = testbits.$(ObjSuf)
TESTBITSS     = testbits.$(SrcSuf)
TESTBITS      = testbits$(ExeSuf)

QPRANDOMO     = QpRandomDriver.$(ObjSuf)
QPRANDOMS     = QpRandomDriver.$(SrcSuf)
QPRANDOM      = QpRandomDriver$(ExeSuf)

THREADSO      = threads.$(ObjSuf)
THREADSS      = threads.$(SrcSuf)
THREADS       = threads$(ExeSuf)

OBJS          = $(EVENTO) $(MAINEVENTO) $(HWORLDO) $(HSIMPLEO) $(MINEXAMO) \
                $(TSTRINGO) $(TCOLLEXO) $(VVECTORO) $(VMATRIXO) $(VLAZYO) \
                $(HELLOO) $(ACLOCKO) $(STRESSO) $(TBENCHO) $(BENCHO) \
                $(TCOLLBMO) $(STRESSGEOMO) $(STRESSLO) $(TESTBITSO) \
                $(CTORTUREO) $(QPRANDOMO) $(THREADO)

PROGRAMS      = $(EVENT) $(HWORLD) $(HSIMPLE) $(MINEXAM) $(TSTRING) \
                $(TCOLLEX) $(TCOLLBM) $(VVECTOR) $(VMATRIX) $(VLAZY) \
                $(HELLOSO) $(ACLOCKSO) $(STRESS) $(TBENCHSO) $(BENCH) \
                $(STRESSGEOM) $(STRESSL) $(TESTBITS) $(CTORTURE) $(QPRANDOM) \
                $(THREADS)

OBJS         += $(GUITESTO) $(GUIVIEWERO) $(TETRISO)
PROGRAMS     += $(GUITEST) $(GUIVIEWER) $(TETRISSO)

#------------------------------------------------------------------------------

.SUFFIXES: .$(SrcSuf) .$(ObjSuf) .$(DllSuf)
.PHONY:    Aclock Hello Tetris

all:            $(PROGRAMS)

$(EVENTSO):     $(EVENTO)
ifeq ($(ARCH),aix)
		/usr/ibmcxx/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
else
ifeq ($(ARCH),aix5)
		/usr/vacpp/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
else
ifeq ($(PLATFORM),macosx)
# We need to make both the .dylib and the .so
		$(LD) $(SOFLAGS) $(EVENTO) $(OutPutOpt) $(EVENTSO)
		$(LD) -bundle -undefined $(UNDEFOPT) $(LDFLAGS) $^ \
		   $(OutPutOpt) $(subst .$(DllSuf),.so,$@)
else
ifeq ($(PLATFORM),win32)
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

$(HWORLD):      $(HWORLDO)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(CTORTURE):    $(CTORTUREO)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(HSIMPLE):     $(HSIMPLEO)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(MINEXAM):     $(MINEXAMO)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(TSTRING):     $(TSTRINGO)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(TCOLLEX):     $(TCOLLEXO)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(TCOLLBM):     $(TCOLLBMO)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(VVECTOR):     $(VVECTORO)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(VMATRIX):     $(VMATRIXO)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(VLAZY):       $(VLAZYO)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(STRESSL):     $(STRESSLO)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(TESTBITS):    $(TESTBITSO)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(THREADS):     $(THREADSO)
ifeq ($(HASTHREAD),yes)
ifeq ($(PLATFORM),win32)
		$(LD) $(LDFLAGS) $^ $(LIBS) ../lib/libThread.lib $(OutPutOpt)$@
else
		$(LD) $(LDFLAGS) $^ $(LIBS) -lThread $(OutPutOpt)$@
endif
		@echo "$@ done"
else
		@echo "This version of ROOT has no thread support, $@ not built"
endif

$(QPRANDOM):    $(QPRANDOMO)
ifeq ($(PLATFORM),win32)
		$(LD) $(LDFLAGS) $^ $(LIBS) ../lib/libQuadp.lib $(OutPutOpt)$@
else
		$(LD) $(LDFLAGS) $^ $(LIBS) -lQuadp $(OutPutOpt)$@
endif
		@echo "$@ done"

$(GUITEST):     $(GUITESTO)
		$(LD) $(LDFLAGS) $^ $(GLIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(GUIVIEWER):   $(GUIVIEWERO)
		$(LD) $(LDFLAGS) $^ $(GLIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(STRESS):      $(STRESSO) $(EVENT)
		$(LD) $(LDFLAGS) $(STRESSO) $(EVENTLIB) $(LIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(STRESSGEOM):  $(STRESSGEOMO)
ifeq ($(PLATFORM),win32)
		$(LD) $(LDFLAGS) $^ $(LIBS) ../lib/libGeom.lib $(OutPutOpt)$@
else
		$(LD) $(LDFLAGS) $^ $(LIBS) -lGeom $(OutPutOpt)$@
endif
		@echo "$@ done"

$(BENCH):       $(BENCHO) $(TBENCHSO)
		$(LD) $(LDFLAGS) $(BENCHO) $(TBENCHO) $(LIBS) $(OutPutOpt)$@
		@echo "$@ done"

Hello:          $(HELLOSO)
$(HELLOSO):     $(HELLOO)
ifeq ($(ARCH),aix)
		/usr/ibmcxx/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
else
ifeq ($(ARCH),aix5)
		/usr/vacpp/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
else
ifeq ($(PLATFORM),macosx)
# We need to make both the .dylib and the .so
		$(LD) $(SOFLAGS) $^ $(OutPutOpt) $@
		$(LD) -bundle -undefined $(UNDEFOPT) $(LDFLAGS) $^ \
		   $(OutPutOpt) $(subst .$(DllSuf),.so,$@)
else
		$(LD) $(SOFLAGS) $(LDFLAGS) $^ $(EXPLLINKLIBS) $(OutPutOpt)$@
endif
endif
endif

Aclock:         $(ACLOCKSO)
$(ACLOCKSO):    $(ACLOCKO)
ifeq ($(ARCH),aix)
		/usr/ibmcxx/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
else
ifeq ($(ARCH),aix5)
		/usr/vacpp/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
else
ifeq ($(PLATFORM),macosx)
# We need to make both the .dylib and the .so
		$(LD) $(SOFLAGS) $^ $(OutPutOpt) $@
		$(LD) -bundle -undefined $(UNDEFOPT) $(LDFLAGS) $^ \
		   $(OutPutOpt) $(subst .$(DllSuf),.so,$@)
else
		$(LD) $(SOFLAGS) $(LDFLAGS) $^ $(EXPLLINKLIBS) $(OutPutOpt)$@
endif
endif
endif

Tetris:         $(TETRISSO)
$(TETRISSO):    $(TETRISO)
ifeq ($(ARCH),aix)
		/usr/ibmcxx/bin/makeC++SharedLib $(OutPutOpt) $@ $(GLIBS) -p 0 $^
else
ifeq ($(ARCH),aix5)
		/usr/vacpp/bin/makeC++SharedLib $(OutPutOpt) $@ $(GLIBS) -p 0 $^
else
ifeq ($(PLATFORM),macosx)
# We need to make both the .dylib and the .so
		$(LD) $(SOFLAGS) $^ $(OutPutOpt) $@
		$(LD) -bundle -undefined $(UNDEFOPT) $(LDFLAGS) $^ \
	 	   $(OutPutOpt) $(subst .$(DllSuf),.so,$@)
else
		$(LD) $(SOFLAGS) $(LDFLAGS) $^ $(EXPLLINKLIBS) $(OutPutOpt)$@
endif
endif
endif

$(TBENCHSO):    $(TBENCHO)
ifeq ($(ARCH),aix)
		/usr/ibmcxx/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
else
ifeq ($(ARCH),aix5)
		/usr/vacpp/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
else
ifeq ($(PLATFORM),macosx)
# We need to make both the .dylib and the .so
		$(LD) $(SOFLAGS) $^ -lstdc++ $(OutPutOpt) $@
		$(LD) -bundle -undefined $(UNDEFOPT) $(LDFLAGS) $^ \
		   $(OutPutOpt) $(subst .$(DllSuf),.so,$@)
else
		$(LD) $(SOFLAGS) $(LDFLAGS) $^ $(EXPLLINKLIBS) $(OutPutOpt)$@
endif
endif
endif

clean:
		@rm -f $(OBJS) core

distclean:      clean
		@rm -f $(PROGRAMS) $(EVENTSO) $(EVENTLIB) *Dict.* *.def *.exp \
		   *.root *.ps *.so *.lib *.dll *.d .def so_locations
		@rm -rf cxx_repository
		-@cd RootShower && $(MAKE) distclean

.SUFFIXES: .$(SrcSuf)

###

Event.$(ObjSuf): Event.h
MainEvent.$(ObjSuf): Event.h

EventDict.$(SrcSuf): Event.h EventLinkDef.h
	@echo "Generating dictionary $@..."
	@rootcint -f $@ -c $^

Hello.$(ObjSuf): Hello.h
HelloDict.$(SrcSuf): Hello.h
	@echo "Generating dictionary $@..."
	@rootcint -f $@ -c $^

Aclock.$(ObjSuf): Aclock.h
AclockDict.$(SrcSuf): Aclock.h
	@echo "Generating dictionary $@..."
	@rootcint -f $@ -c $^

Tetris.$(ObjSuf): Tetris.h
TetrisDict.$(SrcSuf): Tetris.h
	@echo "Generating dictionary $@..."
	@rootcint -f $@ -c $^

TBench.$(ObjSuf): TBench.h
TBenchDict.$(SrcSuf): TBench.h benchLinkDef.h
	@echo "Generating dictionary $@..."
	@rootcint -f $@ -c $^

guiviewer.$(ObjSuf): guiviewer.h
guiviewerDict.$(SrcSuf): guiviewer.h guiviewerLinkDef.h
	@echo "Generating dictionary $@..."
	@rootcint -f $@ -c $^

.$(SrcSuf).$(ObjSuf):
	$(CXX) $(CXXFLAGS) -c $<
