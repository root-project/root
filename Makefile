# Top level Makefile for ROOT System
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000


##### include path/location macros (result of ./configure) ####

include config/Makefile.config

##### include machine dependent macros ####

include config/Makefile.$(ARCH)

##### allow local macros ####

-include MyConfig.mk

##### Modules to build #####

MODULES       = build cint utils base cont meta net zip clib new hist \
                tree graf g3d gpad gui matrix minuit histpainter proof \
                treeplayer treeviewer physics postscript rint html eg

ifneq ($(ARCH),win32)
MODULES      += unix x11 x3d rootx rootd proofd
SYSTEMO       = $(UNIXO)
SYSTEMDO      = $(UNIXDO)
else
MODULES      += winnt win32
SYSTEMO       = $(WINNTO)
SYSTEMDO      = $(WINNTDO)
endif
ifneq ($(TTFINCDIR),)
MODULES      += x11ttf
endif
ifneq ($(OPENGLINCDIR),)
MODULES      += gl
endif
ifneq ($(RFIO),)
MODULES      += rfio
endif
ifneq ($(THREAD),)
MODULES      += thread
endif
ifneq ($(PYTHIA),)
MODULES      += pythia
endif
ifneq ($(PYTHIA6),)
MODULES      += pythia6
endif
ifneq ($(VENUS),)
MODULES      += venus
endif
ifneq ($(STAR),)
MODULES      += star
endif
ifneq ($(MYSQLINCDIR),)
MODULES      += mysql
endif
ifneq ($(SRPDIR),)
MODULES      += srputils
endif

ifneq ($(findstring $(MAKECMDGOALS),distclean distsrc),)
MODULES      += unix winnt x11 x11ttf win32 gl rfio thread pythia pythia6 \
                venus star mysql srputils x3d rootx rootd proofd
MODULES      := $(sort $(MODULES))  # removes duplicates
endif

MODULES      += main   # must be last, $(ALLLIBS) must be fully formed

##### ROOT libraries #####

LPATH         = lib

ifneq ($(ARCH),win32)
RPATH        := -L$(LPATH)
CINTLIBS     := -lCint
ROOTLIBS     := -lNew -lCore -lCint -lHist -lGraf -lGraf3d -lTree
RINTLIBS     := -lRint
PROOFLIBS    := -lGpad -lProof -lTreePlayer
CERNPATH     := -L$(CERNLIBDIR)
CERNLIBS     := -lpacklib -lkernlib
else
CINTLIBS     := $(LPATH)/libCint.lib
ROOTLIBS     := $(LPATH)/libNew.lib $(LPATH)/libCore.lib $(LPATH)/libCint.lib \
                $(LPATH)/libHist.lib $(LPATH)/libGraf.lib \
                $(LPATH)/libGraf3d.lib $(LPATH)/libTree.lib
RINTLIBS     := $(LPATH)/libRint.lib
PROOFLIBS    := $(LPATH)/libGpad.lib $(LPATH)/libProof.lib \
                $(LPATH)/libTreePlayer.lib
CERNLIBS     := '$(shell cygpath -w -- $(CERNLIBDIR)/packlib.lib)' \
                '$(shell cygpath -w -- $(CERNLIBDIR)/kernlib.lib)'
endif

##### f77 options #####

ifeq ($(F77LD),)
F77LD        := $(LD)
endif
ifeq ($(F77OPT),)
F77OPT       := $(OPT)
endif
ifeq ($(F77LDFLAGS),)
F77LDFLAGS   := $(LDFLAGS)
endif

##### utilities #####

MAKEDEP       = build/unix/depend.sh
MAKELIB       = build/unix/makelib.sh
MAKEDIST      = build/unix/makedist.sh
MAKEDISTSRC   = build/unix/makedistsrc.sh
MAKEVERSION   = build/unix/makeversion.sh
IMPORTCINT    = build/unix/importcint.sh
MAKECOMPDATA  = build/unix/compiledata.sh
MAKEMAKEINFO  = build/unix/makeinfo.sh
MAKECHANGELOG = build/unix/makechangelog.sh
MAKEHTML      = build/unix/makehtml.sh
MAKELOGHTML   = build/unix/makeloghtml.sh
ifeq ($(ARCH),win32)
MAKELIB       = build/win/makelib.sh
MAKEDIST      = build/win/makedist.sh
MAKEMAKEINFO  = build/win/makeinfo.sh
endif

##### compiler directives #####

COMPILEDATA   = include/compiledata.h
MAKEINFO      = cint/MAKEINFO

##### libCore #####

COREO         = $(BASEO) $(CONTO) $(METAO) $(NETO) $(SYSTEMO) $(ZIPO) $(CLIBO)
COREDO        = $(BASEDO) $(CONTDO) $(METADO) $(NETDO) $(SYSTEMDO) $(CLIBDO)

CORELIB      := $(LPATH)/libCore.$(SOEXT)

##### if shared libs need to resolve all symbols (e.g.: aix, win32) #####

ifneq ($(EXPLICITLINK),)
MAINLIBS      = $(CORELIB) $(CINTLIB)
else
MAINLIBS      =
endif

##### all #####

ALLHDRS      :=
ALLLIBS      := $(CORELIB)
ALLEXECS     :=
INCLUDEFILES :=

##### RULES #####

.SUFFIXES: .cxx .d
.PRECIOUS: include/%.h

# special rules (need to be defined before generic ones)
cint/src/%.o: cint/src/%.cxx
	$(CXX) $(OPT) $(CINTCXXFLAGS) -o $@ -c $<

cint/src/%.o: cint/src/%.c
	$(CC) $(OPT) $(CINTCFLAGS) -o $@ -c $<

%.o: %.cxx
	$(CXX) $(OPT) $(CXXFLAGS) -o $@ -c $<

%.o: %.c
	$(CC) $(OPT) $(CFLAGS) -o $@ -c $<

%.o: %.f
ifeq ($(F77),f2c)
	f2c -a -A $<
	$(CC) $(F77OPT) $(CFLAGS) -o $@ -c $*.c
else
	$(F77) $(F77OPT) $(F77FLAGS) -o $@ -c $<
endif


##### TARGETS #####

.PHONY:         all fast config rootcint rootlibs rootexecs dist distsrc \
                clean distclean compiledata importcint version html changelog \
                $(patsubst %,all-%,$(MODULES)) \
                $(patsubst %,clean-%,$(MODULES)) \
                $(patsubst %,distclean-%,$(MODULES))

all:            rootexecs

fast:           rootexecs

include $(patsubst %,%/Module.mk,$(MODULES))

-include MyRules.mk            # allow local rules

ifeq ($(findstring $(MAKECMDGOALS),clean distclean dist distsrc version \
      importcint install showbuild),)
ifeq ($(findstring $(MAKECMDGOALS),fast),)
include $(INCLUDEFILES)
endif
include build/dummy.d          # must be last include
endif


rootcint:       all-cint $(ROOTCINTTMP) $(ROOTCINT) $(CINTTMP) $(CINT)

rootlibs:       rootcint compiledata $(ALLLIBS)

rootexecs:      rootlibs $(ALLEXECS)

compiledata:    $(COMPILEDATA) $(MAKEINFO)

config config/Makefile.:
	@(if [ ! -f config/Makefile.config ] ; then \
	   echo ""; echo "Please, run ./configure first"; echo ""; \
	   exit 1; \
	fi)

$(COMPILEDATA): config/Makefile.$(ARCH)
	@$(MAKECOMPDATA) $(COMPILEDATA) $(CXX) "$(OPT)" "$(CXXFLAGS)" \
	   "$(SOFLAGS)" "$(LDFLAGS)" "$(SOEXT)" "$(SYSLIBS)" "$(LIBDIR)" \
	   "$(ROOTLIBS)" "$(RINTLIBS)" "$(INCDIR)"

$(MAKEINFO): config/Makefile.$(ARCH)
	@$(MAKEMAKEINFO) $(MAKEINFO) $(CXX) $(CC)

build/dummy.d: config $(RMKDEP) $(BINDEXP) $(ALLHDRS)
	@(if [ ! -f $@ ] ; then \
	   touch $@; \
	fi)

%.d: %.c $(RMKDEP)
	$(MAKEDEP) $@ "$(CFLAGS)" $*.c > $@

%.d: %.cxx $(RMKDEP)
	$(MAKEDEP) $@ "$(CXXFLAGS)" $*.cxx > $@

$(CORELIB): $(COREO) $(COREDO) $(CINTLIB) $(CORELIBDEP)
	@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
	   "$(SOFLAGS)" libCore.$(SOEXT) $@ "$(COREO) $(COREDO)" \
	   "$(CORELIBEXTRA)"

dist:
	@$(MAKEDIST)

distsrc:
	@$(MAKEDISTSRC)

clean::
	@rm -f __compiledata __makeinfo *~ core

ifeq ($(CXX),KCC)
clean::
	@find . -name "ti_files" -exec rm -rf {} \;
endif

distclean:: clean
	@mv -f include/config.h include/config.hh
	@rm -f include/*.h $(MAKEINFO) $(CORELIB)
	@mv -f include/config.hh include/config.h
	@rm -f build/dummy.d bin/*.dll lib/*.def lib/*.exp lib/*.lib .def
	@rm -f tutorials/*.root tutorials/*.ps tutorials/*.gif so_locations
	@rm -rf htmldoc
	@cd test && $(MAKE) distclean

version: $(CINTTMP)
	@$(MAKEVERSION)

importcint: distclean-cint
	@$(IMPORTCINT)

html: $(ROOTEXE)
	@$(MAKELOGHTML)
	@$(MAKEHTML)

changelog:
	@$(MAKECHANGELOG)

install:
	@(inode1=`ls -id $(BINDIR) | awk '{ print $$1 }'`; \
	inode2=`ls -id $$(pwd)/bin | awk '{ print $$1 }'`;\
	if [ -d $(BINDIR) ] && [ $$inode1 -eq $$inode2 ]; then \
		echo "Everything already installed..."; \
	else \
		echo "Installing binaries in $(BINDIR)"; \
		$(INSTALL) $(ALLEXECS) $(CINT) $(ROOTCINT) $(BINDIR); \
		echo "Installing libraries in $(LIBDIR)"; \
		chmod u+w $(LIBDIR)/*; \
		$(INSTALL) $(ALLLIBS) $(LIBDIR); \
		$(INSTALL) $(CINTLIB) $(LIBDIR); \
		echo "Installing headers in $(INCDIR)"; \
		$(INSTALLDATA) include/*.h $(INCDIR); \
		echo "Installing main/src/rmain.cxx in $(INCDIR)"; \
		$(INSTALLDATA) main/src/rmain.cxx $(INCDIR); \
		echo "Installing $(MAKEINFO) in $(CINTINCDIR)"; \
		$(INSTALLDATA) $(MAKEINFO) $(CINTINCDIR); \
		echo "Installing cint/include lib and stl in $(CINTINCDIR)"; \
		$(INSTALLDATA) cint/include cint/lib cint/stl $(CINTINCDIR); \
	fi)

showbuild:
	@echo "ROOTSYS            = $(ROOTSYS)"
	@echo "PLATFORM           = $(PLATFORM)"
	@echo "OPT                = $(OPT)"
	@echo ""
	@echo "CXX                = $(CXX)"
	@echo "CC                 = $(CC)"
	@echo "F77                = $(F77)"
	@echo "CPP                = $(CPP)"
	@echo "LD                 = $(LD)"
	@echo "F77LD              = $(F77LD)"
	@echo ""
	@echo "CXXFLAGS           = $(CXXFLAGS)"
	@echo "CINTCXXFLAGS       = $(CINTCXXFLAGS)"
	@echo "EXTRA_CXXFLAGS     = $(EXTRA_CXXFLAGS)"
	@echo "CFLAGS             = $(CFLAGS)"
	@echo "CINTCFLAGS         = $(CINTCFLAGS)"
	@echo "EXTRA_CFLAGS       = $(EXTRA_CFLAGS)"
	@echo "F77FLAGS           = $(F77FLAGS)"
	@echo "LDFLAGS            = $(LDFLAGS)"
	@echo "EXTRA_LDFLAGS      = $(EXTRA_LDFLAGS)"
	@echo "SOFLAGS            = $(SOFLAGS)"
	@echo "SOEXT              = $(SOEXT)"
	@echo ""
	@echo "SYSLIBS            = $(SYSLIBS)"
	@echo "XLIBS              = $(XLIBS)"
	@echo "CILIBS             = $(CILIBS)"
	@echo "F77LIBS            = $(F77LIBS)"
	@echo ""
	@echo "PYTHIA             = $(PYTHIA)"
	@echo "PYTHIA6            = $(PYTHIA6)"
	@echo "VENUS              = $(VENUS)"
	@echo "STAR               = $(STAR)"
	@echo "XPMLIBDIR          = $(XPMLIBDIR)"
	@echo "TTFLIBDIR          = $(TTFLIBDIR)"
	@echo "TTFINCDIR          = $(TTFINCDIR)"
	@echo "TTFFONTDIR         = $(TTFFONTDIR)"
	@echo "OPENGLLIBDIR       = $(OPENGLLIBDIR)"
	@echo "OPENGLINCDIR       = $(OPENGLINCDIR)"
	@echo "CERNLIBDIR         = $(CERNLIBDIR)"
	@echo "THREAD             = $(THREAD)"
	@echo "RFIO               = $(RFIO)"
	@echo "MYSQLINCDIR        = $(MYSQLINCDIR)"
	@echo "SRPDIR             = $(SRPDIR)"
	@echo "AFSDIR             = $(AFSDIR)"
	@echo ""
	@echo "INSTALL            = $(INSTALL)"
	@echo "MAKEDEP            = $(MAKEDEP)"
	@echo "MAKELIB            = $(MAKELIB)"
	@echo "MAKEDIST           = $(MAKEDIST)"
	@echo "MAKEDISTSRC        = $(MAKEDISTSRC)"
	@echo "MAKEVERSION        = $(MAKEVERSION)"
	@echo "IMPORTCINT         = $(IMPORTCINT)"
