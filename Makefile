# Top level Makefile for ROOT System
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000


##### Include path/location macros (result of ./configure) #####
##### However, if we are building packages or cleaning,    #####
##### config/Makefile.config isn't made yet - the package  #####
##### scripts want's to make it them selves - so we don't  #####

ifeq ($(findstring $(MAKECMDGOALS), maintainer-clean debian redhat),)
include config/Makefile.config
endif

##### Include machine dependent macros                     #####
##### However, if we are building packages or cleaning, we #####
##### don't include this file since it may screw up things #####

ifeq ($(findstring $(MAKECMDGOALS), maintainer-clean debian redhat),)
include config/Makefile.$(ARCH)
endif

##### Allow local macros #####

-include MyConfig.mk

##### Modules to build #####

MODULES       = build cint utils base cont meta net zip clib matrix newdelete \
                hist tree graf g3d gpad gui minuit histpainter proof \
                treeplayer treeviewer physics postscript rint html eg mc

ifeq ($(ARCH),win32)
MODULES      += winnt win32 gl
SYSTEMO       = $(WINNTO)
SYSTEMDO      = $(WINNTDO)
else
ifeq ($(ARCH),win32gdk)
MODULES      += winnt win32gdk gl
SYSTEMO       = $(WINNTO)
SYSTEMDO      = $(WINNTDO)
else
MODULES      += unix x11 x3d rootx rootd proofd
SYSTEMO       = $(UNIXO)
SYSTEMDO      = $(UNIXDO)
endif
endif
ifneq ($(TTFINCDIR),)
ifneq ($(TTFLIB),)
MODULES      += x11ttf
endif
endif
ifneq ($(OPENGLINCDIR),)
ifneq ($(OPENGLULIB),)
ifneq ($(OPENGLLIB),)
MODULES      += gl
endif
endif
endif
ifneq ($(MYSQLINCDIR),)
ifneq ($(MYSQLCLILIB),)
MODULES      += mysql
endif
endif
ifneq ($(PGSQLINCDIR),)
ifneq ($(PGSQLCLILIB),)
MODULES      += pgsql
endif
endif
ifneq ($(SAPDBINCDIR),)
ifneq ($(SAPDBCLILIB),)
MODULES      += sapdb
endif
endif
ifneq ($(SHIFTLIB),)
MODULES      += rfio
endif
ifneq ($(DCAPLIB),)
MODULES      += dcache
endif
ifneq ($(OSTHREADLIB),)
MODULES      += thread
endif
ifneq ($(FPYTHIALIB),)
MODULES      += pythia
endif
ifneq ($(FPYTHIA6LIB),)
MODULES      += pythia6
endif
ifneq ($(FVENUSLIB),)
MODULES      += venus
endif
ifneq ($(STAR),)
MODULES      += star
endif
ifneq ($(SRPUTILLIB),)
MODULES      += srputils
endif
ifneq ($(KRB5LIB),)
MODULES      += krb5auth
endif
ifneq ($(CERNLIBS),)
MODULES      += hbook
endif

ifneq ($(findstring $(MAKECMDGOALS),distclean maintainer-clean),)
MODULES      += unix winnt x11 x11ttf win32 win32gdk gl rfio thread pythia \
                pythia6 venus star mysql pgsql sapdb srputils x3d rootx \
                rootd proofd dcache hbook
MODULES      := $(sort $(MODULES))  # removes duplicates
endif

MODULES      += main   # must be last, $(ALLLIBS) must be fully formed

##### ROOT libraries #####

LPATH         = lib

ifneq ($(PLATFORM),win32)
RPATH        := -L$(LPATH)
CINTLIBS     := -lCint
NEWLIBS      := -lNew
ROOTLIBS     := -lCore -lCint -lHist -lGraf -lGraf3d -lTree -lMatrix
RINTLIBS     := -lRint
PROOFLIBS    := -lGpad -lProof -lTreePlayer
else
CINTLIBS     := $(LPATH)/libCint.lib
NEWLIBS      := $(LPATH)/libNew.lib
ROOTLIBS     := $(LPATH)/libCore.lib $(LPATH)/libCint.lib \
                $(LPATH)/libHist.lib $(LPATH)/libGraf.lib \
                $(LPATH)/libGraf3d.lib $(LPATH)/libTree.lib \
                $(LPATH)/libMatrix.lib
RINTLIBS     := $(LPATH)/libRint.lib
PROOFLIBS    := $(LPATH)/libGpad.lib $(LPATH)/libProof.lib \
                $(LPATH)/libTreePlayer.lib
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
MAKELIB       = build/unix/makelib.sh $(MKLIBOPTIONS)
MAKEDIST      = build/unix/makedist.sh
MAKEDISTSRC   = build/unix/makedistsrc.sh
MAKEVERSION   = build/unix/makeversion.sh
IMPORTCINT    = build/unix/importcint.sh
MAKECOMPDATA  = build/unix/compiledata.sh
MAKEMAKEINFO  = build/unix/makeinfo.sh
MAKECHANGELOG = build/unix/makechangelog.sh
MAKEHTML      = build/unix/makehtml.sh
MAKELOGHTML   = build/unix/makeloghtml.sh
MAKECINTDLLS  = build/unix/makecintdlls.sh
MAKESTATIC    = build/unix/makestatic.sh
ifeq ($(PLATFORM),win32)
MAKELIB       = build/win/makelib.sh
MAKEDIST      = build/win/makedist.sh
MAKECOMPDATA  = build/win/compiledata.sh
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
                clean distclean maintainer-clean compiledata importcint \
                version html changelog install uninstall showbuild cintdlls \
                static debian redhat skip \
                $(patsubst %,all-%,$(MODULES)) \
                $(patsubst %,clean-%,$(MODULES)) \
                $(patsubst %,distclean-%,$(MODULES))

all:            rootexecs

fast:           rootexecs

skip:
		@true;

include $(patsubst %,%/Module.mk,$(MODULES))

-include MyRules.mk            # allow local rules

ifeq ($(findstring $(MAKECMDGOALS),clean distclean maintainer-clean dist \
      distsrc version importcint install uninstall showbuild changelog html \
      debian redhat),)
ifeq ($(findstring skip,$(MAKECMDGOALS))$(findstring fast,$(MAKECMDGOALS)),)
include $(INCLUDEFILES)
endif
include build/dummy.d          # must be last include
endif


rootcint:       all-cint $(ROOTCINTTMP) $(ROOTCINT)

rootlibs:       rootcint compiledata $(ALLLIBS)

rootexecs:      rootlibs $(ALLEXECS)

compiledata:    $(COMPILEDATA) $(MAKEINFO)

config config/Makefile.:
	@(if [ ! -f config/Makefile.config ] ; then \
	   echo ""; echo "Please, run ./configure first"; echo ""; \
	   exit 1; \
	fi)

$(COMPILEDATA): config/Makefile.$(ARCH) $(MAKECOMPDATA)
	@$(MAKECOMPDATA) $(COMPILEDATA) $(CXX) "$(OPT)" "$(CXXFLAGS)" \
	   "$(SOFLAGS)" "$(LDFLAGS)" "$(SOEXT)" "$(SYSLIBS)" "$(LIBDIR)" \
	   "$(ROOTLIBS)" "$(RINTLIBS)" "$(INCDIR)" "$(MAKESHAREDLIB)" \
	   "$(MAKEEXE)" "$(ARCH)"

$(MAKEINFO): config/Makefile.$(ARCH)
	@$(MAKEMAKEINFO) $(MAKEINFO) $(CXX) $(CC) "$(CPPPREP)"

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

debian:
	@if [ ! -x `which debuild` ] || [ ! -x `which dh_testdir` ]; then \
	   echo "You must have debuild and debhelper installed to"; \
	   echo "make the Debian GNU/Linux package"; exit 1; fi
	@echo "OK, you're on a Debian GNU/Linux system - cool"
	@vers=`sed 's|\(.*\)/\(.*\)|\1.\2|' < build/version_number` ; \
	  dirvers=`basename $$PWD | sed 's|root-\(.*\)|\1|'` ; \
	  if [ "$$vers" != "$$dirvers" ] ; then \
	    echo "Must have ROOT source tree in root-$$vers" ; \
	    echo "Please rename this directory to `basename $$PWD` to"; \
	    echo "root-$$vers and try again"; exit 1 ; fi
	build/package/lib/makedebclean.sh
	build/package/lib/makedebdir.sh
	debuild -rfakeroot -us -uc -i"G__|^debian|\.d$$"
	@echo "Debian GNU/Linux packages done. They are put in '../'"

redhat:
	@if [ ! -x `which rpm` ]; then \
	   echo "You must have rpm installed to make the Redhat package"; \
	   exit 1; fi
	@echo "OK, you have RPM on your system - good"
	build/package/lib/makerpmclean.sh
	build/package/lib/makerpmspec.sh
	@echo "To build the packages, make a gzipped tar ball of the sources"
	@vers=`sed 's|\(.*\)/\(.*\)|\1.\2|' < build/version_number` ; \
	  echo "called root-v$$vers.source.tar.gz and put it in you RPM "
	@echo "source directory (default /usr/src/rpm/SOURCES) and the "
	@echo "spec-file root.spec in your RPM spec directory"
	@echo "(default /usr/src/RPM/SPECS). If you want to build outside"
	@echo "the regular tree, please refer to the RPM documentation."
	@echo "After that, do"
	@echo "   rpm -ba root.spec "
	@echo "to actually build the packages. More details are given in"
	@echo "README/INSTALL"
#	rpm -bb --rcfile rpm/rpmrc --buildroot `pwd`/rpm/tmp rpm/root.spec
#	@echo "Redhat Linux packages done. They are put in '../<arch>'"

clean::
	@rm -f __compiledata __makeinfo *~ core

ifeq ($(CXX),KCC)
clean::
	@(find . -name "ti_files" -exec rm -rf {} \; >/dev/null 2>&1;true)
endif
ifeq ($(SUNCC5),true)
clean::
	@(find . -name "SunWS_cache" -exec rm -rf {} \; >/dev/null 2>&1;true)
endif

distclean:: clean
	-@mv -f include/config.h include/config.hh
	@rm -f include/*.h $(MAKEINFO) $(CORELIB)
	-@mv -f include/config.hh include/config.h
	@rm -f build/dummy.d bin/*.dll lib/*.def lib/*.exp lib/*.lib .def
	@rm -f tutorials/*.root tutorials/*.ps tutorials/*.gif so_locations
	@rm -f tutorials/pca.C tutorials/*.so work.pc work.pcl
	@rm -f bin/roota lib/libRoot.a
	@rm -f $(CINTDIR)/include/*.dll $(CINTDIR)/include/sys/*.dll
	@rm -f $(CINTDIR)/stl/*.dll README/ChangeLog
	-@cd test && $(MAKE) distclean

maintainer-clean:: distclean
	-build/package/lib/makedebclean.sh
	-build/package/lib/makerpmclean.sh
	@rm -rf bin lib include system.rootrc config/Makefile.config \
	   test/Makefile etc/system.rootrc etc/root.mimes

version: $(CINTTMP)
	@$(MAKEVERSION)

cintdlls: $(CINTTMP)
	@$(MAKECINTDLLS) $(PLATFORM) $(CINTTMP) $(MAKELIB) $(CXX) \
	   $(CC) $(LD) "$(OPT)" "$(CINTCXXFLAGS)" "$(CINTCFLAGS)" \
	   "$(LDFLAGS)" "$(SOFLAGS)" "$(SOEXT)"

static: rootlibs
	@$(MAKESTATIC) $(PLATFORM) $(CXX) $(CC) $(LD) "$(LDFLAGS)" \
	   "$(XLIBS)" "$(SYSLIBS)"

importcint: distclean-cint
	@$(IMPORTCINT)

changelog:
	@$(MAKECHANGELOG)

html: $(ROOTEXE) changelog
	@$(MAKELOGHTML)
	@$(MAKEHTML)

install:
	@if [ -d $(BINDIR) ]; then \
	   inode1=`ls -id $(BINDIR) | awk '{ print $$1 }'`; \
	fi; \
	inode2=`ls -id $$PWD/bin | awk '{ print $$1 }'`; \
	if [ -d $(BINDIR) ] && [ $$inode1 -eq $$inode2 ]; then \
	   echo "Everything already installed..."; \
	else \
	   echo "Installing binaries in $(DESTDIR)$(BINDIR)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(BINDIR); \
	   $(INSTALL) $(CINT)                   $(DESTDIR)$(BINDIR); \
	   $(INSTALL) $(MAKECINT)               $(DESTDIR)$(BINDIR); \
	   $(INSTALL) $(ROOTCINT)               $(DESTDIR)$(BINDIR); \
	   $(INSTALL) $(RMKDEP)                 $(DESTDIR)$(BINDIR); \
	   if [ "x$(BINDEXP)" != "x" ] ; then \
	      $(INSTALL) $(BINDEXP)             $(DESTDIR)$(BINDIR); \
           fi; \
	   $(INSTALL) bin/root-config           $(DESTDIR)$(BINDIR); \
	   $(INSTALL) bin/memprobe              $(DESTDIR)$(BINDIR); \
	   $(INSTALL) $(ALLEXECS)               $(DESTDIR)$(BINDIR); \
	   echo "Installing libraries in $(DESTDIR)$(LIBDIR)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(LIBDIR); \
	   vers=`sed 's|\(.*\)\..*/.*|\1|' < build/version_number` ; \
	   for lib in $(ALLLIBS) $(CINTLIB); do \
	      rm -f $(DESTDIR)$(LIBDIR)/`basename $$lib` ; \
	      rm -f $(DESTDIR)$(LIBDIR)/`basename $$lib`.$$vers ; \
	      $(INSTALL) $$lib*                 $(DESTDIR)$(LIBDIR); \
	   done ; \
	   echo "Installing headers in $(DESTDIR)$(INCDIR)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(INCDIR); \
	   $(INSTALLDATA) include/*.h           $(DESTDIR)$(INCDIR); \
	   echo "Installing main/src/rmain.cxx in $(DESTDIR)$(INCDIR)"; \
	   $(INSTALLDATA) main/src/rmain.cxx    $(DESTDIR)$(INCDIR); \
	   echo "Installing $(MAKEINFO) in $(DESTDIR)$(CINTINCDIR)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(CINTINCDIR); \
	   $(INSTALLDATA) $(MAKEINFO)           $(DESTDIR)$(CINTINCDIR); \
	   echo "Installing cint/include cint/lib and cint/stl in $(DESTDIR)$(CINTINCDIR)"; \
	   $(INSTALLDATA) cint/include          $(DESTDIR)$(CINTINCDIR); \
	   $(INSTALLDATA) cint/lib              $(DESTDIR)$(CINTINCDIR); \
	   $(INSTALLDATA) cint/stl              $(DESTDIR)$(CINTINCDIR); \
	   rm -rf $(DESTDIR)$(CINTINCDIR)/include/CVS; \
	   rm -rf $(DESTDIR)$(CINTINCDIR)/lib/CVS; \
	   rm -rf $(DESTDIR)$(CINTINCDIR)/stl/CVS; \
	   echo "Installing PROOF files in $(DESTDIR)$(PROOFDATADIR)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(PROOFDATADIR); \
	   $(INSTALLDATA) proof/etc             $(DESTDIR)$(PROOFDATADIR); \
	   $(INSTALLDATA) proof/utils           $(DESTDIR)$(PROOFDATADIR); \
	   rm -rf $(DESTDIR)$(PROOFDATADIR)/etc/CVS; \
	   rm -rf $(DESTDIR)$(PROOFDATADIR)/utils/CVS; \
	   echo "Installing icons in $(DESTDIR)$(ICONPATH)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(ICONPATH); \
	   $(INSTALLDATA) icons/*.xpm           $(DESTDIR)$(ICONPATH); \
	   echo "Installing misc docs in  $(DESTDIR)$(DOCDIR)" ; \
	   $(INSTALLDIR)                        $(DESTDIR)$(DOCDIR); \
	   $(INSTALLDATA) LICENSE               $(DESTDIR)$(DOCDIR); \
	   $(INSTALLDATA) README/README         $(DESTDIR)$(DOCDIR); \
	   $(INSTALLDATA) README/README.PROOF   $(DESTDIR)$(DOCDIR); \
	   $(INSTALLDATA) README/ChangeLog-2-24 $(DESTDIR)$(DOCDIR); \
	   $(INSTALLDATA) README/CREDITS        $(DESTDIR)$(DOCDIR); \
	   echo "Installing tutorials in $(DESTDIR)$(TUTDIR)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(TUTDIR); \
	   $(INSTALLDATA) tutorials/*           $(DESTDIR)$(TUTDIR); \
	   rm -rf $(DESTDIR)$(TUTDIR)/CVS; \
	   echo "Installing tests in $(DESTDIR)$(TESTDIR)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(TESTDIR); \
	   $(INSTALLDATA) test/*                $(DESTDIR)$(TESTDIR); \
	   rm -rf $(DESTDIR)$(TESTDIR)/CVS; \
	   echo "Installing macros in $(DESTDIR)$(MACRODIR)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(MACRODIR); \
	   $(INSTALLDATA) macros/*              $(DESTDIR)$(MACRODIR); \
	   rm -rf $(DESTDIR)$(MACRODIR)/CVS; \
	   echo "Installing man(1) pages in $(DESTDIR)$(MANDIR)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(MANDIR); \
	   $(INSTALLDATA) man/man1/*            $(DESTDIR)$(MANDIR); \
	   rm -rf $(DESTDIR)$(MANDIR)/CVS; \
	   echo "Installing config files in $(DESTDIR)$(ETCDIR)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(ETCDIR); \
	   $(INSTALLDATA) etc/*                 $(DESTDIR)$(ETCDIR); \
	   rm -rf $(DESTDIR)$(ETCDIR)/CVS; \
	   echo "Installing Autoconf macro in $(DESTDIR)$(ACLOCALDIR)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(ACLOCALDIR); \
	   $(INSTALLDATA) build/misc/root.m4    $(DESTDIR)$(ACLOCALDIR); \
	   rm -rf $(DESTDIR)$(DATADIR)/CVS; \
	fi

uninstall:
	@if [ -d $(BINDIR) ]; then \
	   inode1=`ls -id $(BINDIR) | awk '{ print $$1 }'`; \
	fi; \
	inode2=`ls -id $$PWD/bin | awk '{ print $$1 }'`; \
	if [ -d $(BINDIR) ] && [ $$inode1 -eq $$inode2 ]; then \
	   $(MAKE) distclean ; \
	else \
	   rm -f $(DESTDIR)$(BINDIR)/`basename $(CINT)`; \
	   rm -f $(DESTDIR)$(BINDIR)/`basename $(MAKECINT)`; \
	   rm -f $(DESTDIR)$(BINDIR)/`basename $(ROOTCINT)`; \
	   rm -f $(DESTDIR)$(BINDIR)/`basename $(RMKDEP)`; \
	   if [ "x$(BINDEXP)" != "x" ] ; then \
	      rm -f $(DESTDIR)$(BINDIR)/`basename $(BINDEXP)`; \
	   fi; \
	   rm -f $(DESTDIR)$(BINDIR)/root-config; \
	   rm -f $(DESTDIR)$(BINDIR)/memprobe; \
	   for i in $(ALLEXECS) ; do \
	      rm -f $(DESTDIR)$(BINDIR)/`basename $$i`; \
	   done; \
	   if test -d $(DESTDIR)$(BINDIR) && \
	      test "x`ls $(DESTDIR)$(BINDIR)`" = "x" ; then \
	      rm -rf $(DESTDIR)$(BINDIR); \
	   fi ; \
	   for lib in $(ALLLIBS) $(CINTLIB); do \
	      rm -f $(DESTDIR)$(LIBDIR)/`basename $$lib`* ; \
	   done ; \
	   if test -d $(DESTDIR)$(LIBDIR) && \
	      test "x`ls $(DESTDIR)$(LIBDIR)`" = "x" ; then \
	      rm -rf $(DESTDIR)$(LIBDIR); \
	   fi ; \
	   for i in include/*.h ; do \
	      rm -f $(DESTDIR)$(INCDIR)/`basename $$i`; \
	   done ; \
	   if test -d $(DESTDIR)$(INCDIR) && \
	      test "x`ls $(DESTDIR)$(INCDIR)`" = "x" ; then \
	      rm -rf $(DESTDIR)$(INCDIR); \
	   fi ; \
	   rm -f $(DESTDIR)$(INCDIR)/rmain.cxx; \
	   rm -rf $(DESTDIR)$(CINTINCDIR); \
	   rm -rf $(DESTDIR)$(PROOFDATADIR); \
	   for i in icons/*.xpm ; do \
	      rm -fr $(DESTDIR)$(ICONPATH)/`basename $$i`; \
	   done; \
	   if test -d $(DESTDIR)$(ICONPATH) && \
	      test "x`ls $(DESTDIR)$(ICONPATH)`" = "x" ; then \
	      rm -rf $(DESTDIR)$(ICONPATH); \
	   fi ; \
	   rm -rf $(DESTDIR)$(TUTDIR); \
	   rm -rf $(DESTDIR)$(TESTDIR); \
	   rm -rf $(DESTDIR)$(DOCDIR); \
	   rm -rf $(DESTDIR)$(MACRODIR); \
	   for i in man/man1/* ; do \
	      rm -fr $(DESTDIR)$(MANDIR)/`basename $$i`; \
	   done; \
	   if test -d $(DESTDIR)$(MANDIR) && \
	      test "x`ls $(DESTDIR)$(MANDIR)`" = "x" ; then \
	      rm -rf $(DESTDIR)$(MANDIR); \
	   fi ; \
	   for i in etc/* ; do \
	      rm -fr $(DESTDIR)$(ETCDIR)/`basename $$i`; \
	   done; \
	   if test -d $(DESTDIR)$(ETCDIR) && \
	      test "x`ls $(DESTDIR)$(ETCDIR)`" = "x" ; then \
	      rm -rf $(DESTDIR)$(ETCDIR); \
	   fi ; \
	   for i in build/misc/* ; do \
	      rm -fr $(DESTDIR)$(DATADIR)/`basename $$i`; \
	   done; \
	   if test -d $(DESTDIR)$(DATADIR) && \
	      test "x`ls $(DESTDIR)$(DATADIR)`" = "x" ; then \
	      rm -rf $(DESTDIR)$(DATADIR); \
	   fi ; \
	fi

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
	@echo "GCCVERS            = $(GCCVERS)"
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
	@echo "FPYTHIALIBDIR      = $(FPYTHIALIBDIR)"
	@echo "FPYTHIA6LIBDIR     = $(FPYTHIA6LIBDIR)"
	@echo "FVENUSLIBDIR       = $(FVENUSLIBDIR)"
	@echo "STAR               = $(STAR)"
	@echo "XPMLIBDIR          = $(XPMLIBDIR)"
	@echo "XPMLIB             = $(XPMLIB)"
	@echo "TTFLIBDIR          = $(TTFLIBDIR)"
	@echo "TTFLIB             = $(TTFLIB)"
	@echo "TTFINCDIR          = $(TTFINCDIR)"
	@echo "TTFFONTDIR         = $(TTFFONTDIR)"
	@echo "OPENGLLIBDIR       = $(OPENGLLIBDIR)"
	@echo "OPENGLULIB         = $(OPENGLULIB)"
	@echo "OPENGLLIB          = $(OPENGLLIB)"
	@echo "OPENGLINCDIR       = $(OPENGLINCDIR)"
	@echo "CERNLIBDIR         = $(CERNLIBDIR)"
	@echo "CERNLIBS           = $(CERNLIBS)"
	@echo "OSTHREADLIB        = $(OSTHREADLIB)"
	@echo "SHIFTLIB           = $(SHIFTLIB)"
	@echo "DCAPLIB            = $(DCAPLIB)"
	@echo "MYSQLINCDIR        = $(MYSQLINCDIR)"
	@echo "PGSQLINCDIR        = $(PGSQLINCDIR)"
	@echo "SAPDBINCDIR        = $(SAPDBINCDIR)"
	@echo "SRPLIBDIR          = $(SRPLIBDIR)"
	@echo "SRPINCDIR          = $(SRPINCDIR)"
	@echo "SRPUTILLIB         = $(SRPUTILLIB)"
	@echo "AFSDIR             = $(AFSDIR)"
	@echo ""
	@echo "INSTALL            = $(INSTALL)"
	@echo "MAKEDEP            = $(MAKEDEP)"
	@echo "MAKELIB            = $(MAKELIB)"
	@echo "MAKEDIST           = $(MAKEDIST)"
	@echo "MAKEDISTSRC        = $(MAKEDISTSRC)"
	@echo "MAKEVERSION        = $(MAKEVERSION)"
	@echo "IMPORTCINT         = $(IMPORTCINT)"
