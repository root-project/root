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
EVENTLIB      = $(shell pwd)/$(EVENTSO)
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

STRESSGO      = stressGraphics.$(ObjSuf)
STRESSGS      = stressGraphics.$(SrcSuf)
STRESSG       = stressGraphics$(ExeSuf)

ifeq ($(shell $(ROOTCONFIG) --has-mathcore),yes)
STRESSVECO   = stressVector.$(ObjSuf)
STRESSVECS   = stressVector.$(SrcSuf)
STRESSVEC    = stressVector$(ExeSuf)
endif

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

STRESSGEOMETRYO   = stressGeometry.$(ObjSuf)
STRESSGEOMETRYS   = stressGeometry.$(SrcSuf)
STRESSGEOMETRY    = stressGeometry$(ExeSuf)

STRESSSHAPESO   = stressShapes.$(ObjSuf)
STRESSSHAPESS   = stressShapes.$(SrcSuf)
STRESSSHAPES    = stressShapes$(ExeSuf)

STRESSFITO    = stressFit.$(ObjSuf)
STRESSFITS    = stressFit.$(SrcSuf)
STRESSFIT     = stressFit$(ExeSuf)

STRESSENTRYLISTO = stressEntryList.$(ObjSuf)
STRESSENTRYLISTS = stressEntryList.$(SrcSuf)
STRESSENTRYLIST  = stressEntryList$(ExeSuf)

STRESSHEPIXO  = stressHepix.$(ObjSuf)
STRESSHEPIXS  = stressHepix.$(SrcSuf)
STRESSHEPIX   = stressHepix$(ExeSuf)

STRESSSPO     = stressSpectrum.$(ObjSuf)
STRESSSPS     = stressSpectrum.$(SrcSuf)
STRESSSP      = stressSpectrum$(ExeSuf)

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
                $(STRESSSHAPESO) $(TCOLLBMO) $(STRESSGEOMETRYO) $(STRESSLO) \
                $(STRESSGO) $(STRESSSPO) $(TESTBITSO) \
                $(CTORTUREO) $(QPRANDOMO) $(THREADO) $(STRESSVECO) \
                $(STRESSFITO) $(STRESSHEPIXO) $(STRESSENTRYLISTO)

PROGRAMS      = $(EVENT) $(HWORLD) $(HSIMPLE) $(MINEXAM) $(TSTRING) \
                $(TCOLLEX) $(TCOLLBM) $(VVECTOR) $(VMATRIX) $(VLAZY) \
                $(HELLOSO) $(ACLOCKSO) $(STRESS) $(TBENCHSO) $(BENCH) \
                $(STRESSSHAPES) $(STRESSGEOMETRY) $(STRESSL) $(STRESSG) \
                $(TESTBITS) $(CTORTURE) $(QPRANDOM) $(THREADS) $(STRESSSP) \
                $(STRESSVEC) $(STRESSFIT) $(STRESSHEPIX) $(STRESSENTRYLIST)

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
		$(LD) $(SOFLAGS) $^ $(OutPutOpt) $@
ifeq ($(MACOSX_MINOR),4)
		ln -sf $@ $(subst .$(DllSuf),.so,$@)
else
		$(LD) -bundle -undefined $(UNDEFOPT) $(LDFLAGS) $^ \
		   $(OutPutOpt) $(subst .$(DllSuf),.so,$@)
endif
else
ifeq ($(PLATFORM),win32)
		bindexplib $* $^ > $*.def
		lib -nologo -MACHINE:IX86 $^ -def:$*.def \
		   $(OutPutOpt)$(EVENTLIB)
		$(LD) $(SOFLAGS) $(LDFLAGS) $^ $*.exp $(LIBS) \
		   $(OutPutOpt)$@
		$(MT_DLL)
else
		$(LD) $(SOFLAGS) $(LDFLAGS) $^ $(OutPutOpt) $@ $(EXPLLINKLIBS)
endif
endif
endif
endif
		@echo "$@ done"

$(EVENT):       $(EVENTSO) $(MAINEVENTO)
		$(LD) $(LDFLAGS) $(MAINEVENTO) $(EVENTLIB) $(LIBS) $(OutPutOpt)$@
		$(MT_EXE)
		@echo "$@ done"

$(HWORLD):      $(HWORLDO)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		$(MT_EXE)
		@echo "$@ done"

$(CTORTURE):    $(CTORTUREO)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		$(MT_EXE)
		@echo "$@ done"

$(HSIMPLE):     $(HSIMPLEO)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		$(MT_EXE)
		@echo "$@ done"

$(MINEXAM):     $(MINEXAMO)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		$(MT_EXE)
		@echo "$@ done"

$(TSTRING):     $(TSTRINGO)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		$(MT_EXE)
		@echo "$@ done"

$(TCOLLEX):     $(TCOLLEXO)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		$(MT_EXE)
		@echo "$@ done"

$(TCOLLBM):     $(TCOLLBMO)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		$(MT_EXE)
		@echo "$@ done"

$(VVECTOR):     $(VVECTORO)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		$(MT_EXE)
		@echo "$@ done"

$(VMATRIX):     $(VMATRIXO)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		$(MT_EXE)
		@echo "$@ done"

$(VLAZY):       $(VLAZYO)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		$(MT_EXE)
		@echo "$@ done"

$(STRESSL):     $(STRESSLO)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		$(MT_EXE)
		@echo "$@ done"

$(STRESSG):     $(STRESSGO)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		$(MT_EXE)
		@echo "$@ done"

$(STRESSSP):    $(STRESSSPO)
ifeq ($(PLATFORM),win32)
		$(LD) $(LDFLAGS) $^ $(LIBS) ../lib/libSpectrum.lib $(OutPutOpt)$@
		$(MT_EXE)
else
		$(LD) $(LDFLAGS) $^ $(LIBS) -lSpectrum $(OutPutOpt)$@
endif
		@echo "$@ done"

$(STRESSVEC):   $(STRESSVECO)
ifeq ($(PLATFORM),win32)
		$(LD) $(LDFLAGS) $^ $(LIBS) ../lib/libMathCore.lib $(OutPutOpt)$@
		$(MT_EXE)
else
		$(LD) $(LDFLAGS) $^ $(LIBS) -lMathCore $(OutPutOpt)$@
endif
		@echo "$@ done"

$(TESTBITS):    $(TESTBITSO)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		$(MT_EXE)
		@echo "$@ done"

$(THREADS):     $(THREADSO)
ifeq ($(PLATFORM),win32)
		$(LD) $(LDFLAGS) $^ $(LIBS) ../lib/libThread.lib $(OutPutOpt)$@
		$(MT_EXE)
		@echo "$@ done"
else
ifeq ($(HASTHREAD),yes)
		$(LD) $(LDFLAGS) $^ $(LIBS) -lThread $(OutPutOpt)$@
		@echo "$@ done"
else
		@echo "This version of ROOT has no thread support, $@ not built"
endif
endif

$(QPRANDOM):    $(QPRANDOMO)
ifeq ($(PLATFORM),win32)
		$(LD) $(LDFLAGS) $^ $(LIBS) ../lib/libQuadp.lib $(OutPutOpt)$@
		$(MT_EXE)
else
		$(LD) $(LDFLAGS) $^ $(LIBS) -lQuadp $(OutPutOpt)$@
endif
		@echo "$@ done"

$(GUITEST):     $(GUITESTO)
		$(LD) $(LDFLAGS) $^ $(GLIBS) $(OutPutOpt)$@
		$(MT_EXE)
		@echo "$@ done"

$(GUIVIEWER):   $(GUIVIEWERO)
		$(LD) $(LDFLAGS) $^ $(GLIBS) $(OutPutOpt)$@
		$(MT_EXE)
		@echo "$@ done"

$(STRESS):      $(STRESSO) $(EVENT)
		$(LD) $(LDFLAGS) $(STRESSO) $(EVENTLIB) $(LIBS) $(OutPutOpt)$@
		$(MT_EXE)
		@echo "$@ done"

$(STRESSGEOMETRY):  $(STRESSGEOMETRYO)
ifeq ($(PLATFORM),win32)
		$(LD) $(LDFLAGS) $^ $(LIBS) ../lib/libGeom.lib $(OutPutOpt)$@
		$(MT_EXE)
else
		$(LD) $(LDFLAGS) $^ $(LIBS) -lGeom $(OutPutOpt)$@
endif
		@echo "$@ done"

$(STRESSSHAPES):  $(STRESSSHAPESO)
ifeq ($(PLATFORM),win32)
		$(LD) $(LDFLAGS) $^ $(LIBS) ../lib/libGeom.lib $(OutPutOpt)$@
		$(MT_EXE)
else
		$(LD) $(LDFLAGS) $^ $(LIBS) -lGeom $(OutPutOpt)$@
endif
		@echo "$@ done"

$(STRESSFIT):   $(STRESSFITO)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		$(MT_EXE)
		@echo "$@ done"

$(STRESSENTRYLIST):	$(STRESSENTRYLISTO)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(STRESSHEPIX): $(STRESSHEPIXO) $(STRESSGEOMETRY) $(STRESSFIT) $(STRESSL) \
                $(STRESSSP) $(STRESS)
		$(LD) $(LDFLAGS) $(STRESSHEPIXO) $(LIBS) $(OutPutOpt)$@
		$(MT_EXE)
		@echo "$@ done"

$(BENCH):       $(BENCHO) $(TBENCHSO)
		$(LD) $(LDFLAGS) $(BENCHO) $(TBENCHO) $(LIBS) $(OutPutOpt)$@
		$(MT_EXE)
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
ifeq ($(MACOSX_MINOR),4)
		ln -sf $@ $(subst .$(DllSuf),.so,$@)
else
		$(LD) -bundle -undefined $(UNDEFOPT) $(LDFLAGS) $^ \
		   $(OutPutOpt) $(subst .$(DllSuf),.so,$@)
endif
else
		$(LD) $(SOFLAGS) $(LDFLAGS) $^ $(EXPLLINKLIBS) $(OutPutOpt)$@
		$(MT_DLL)
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
ifeq ($(MACOSX_MINOR),4)
		ln -sf $@ $(subst .$(DllSuf),.so,$@)
else
		$(LD) -bundle -undefined $(UNDEFOPT) $(LDFLAGS) $^ \
		   $(OutPutOpt) $(subst .$(DllSuf),.so,$@)
endif
else
		$(LD) $(SOFLAGS) $(LDFLAGS) $^ $(EXPLLINKLIBS) $(OutPutOpt)$@
		$(MT_DLL)
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
ifeq ($(MACOSX_MINOR),4)
		ln -sf $@ $(subst .$(DllSuf),.so,$@)
else
		$(LD) -bundle -undefined $(UNDEFOPT) $(LDFLAGS) $^ \
	 	   $(OutPutOpt) $(subst .$(DllSuf),.so,$@)
endif
else
		$(LD) $(SOFLAGS) $(LDFLAGS) $^ $(EXPLLINKLIBS) $(OutPutOpt)$@
		$(MT_DLL)
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
ifeq ($(MACOSX_MINOR),4)
		ln -sf $@ $(subst .$(DllSuf),.so,$@)
else
		$(LD) -bundle -undefined $(UNDEFOPT) $(LDFLAGS) $^ \
		   $(OutPutOpt) $(subst .$(DllSuf),.so,$@)
endif
else
		$(LD) $(SOFLAGS) $(LDFLAGS) $^ $(EXPLLINKLIBS) $(OutPutOpt)$@
		$(MT_DLL)
endif
endif
endif

clean:
		@rm -f $(OBJS) core

distclean:      clean
		-@mv -f linearIO.root linearIO.roott
		@rm -f $(PROGRAMS) $(EVENTSO) $(EVENTLIB) *Dict.* *.def *.exp \
		   *.root *.ps *.so *.lib *.dll *.d *.log .def so_locations
		@rm -rf cxx_repository
		-@mv -f linearIO.roott linearIO.root
		-@cd RootShower && $(MAKE) distclean
		-@cd rhtml && $(MAKE) distclean

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
