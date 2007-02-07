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
ifeq ($(MAKECMDGOALS),clean)
include config/Makefile.config
endif

MAKE_VERSION_MAJOR := $(word 1,$(subst ., ,$(MAKE_VERSION)))
MAKE_VERSION_MINOR := $(shell echo $(word 2,$(subst ., ,$(MAKE_VERSION))) | \
                              sed 's/\([0-9][0-9]*\).*/\1/')
MAKE_VERSION_MAJOR ?= 0
MAKE_VERSION_MINOR ?= 0
ORDER_ := $(shell test $(MAKE_VERSION_MAJOR) -gt 3 || \
                  test $(MAKE_VERSION_MAJOR) -eq 3 && \
                  test $(MAKE_VERSION_MINOR) -ge 80 && echo '|')

##### Include machine dependent macros                     #####
##### However, if we are building packages or cleaning, we #####
##### don't include this file since it may screw up things #####

ifeq ($(findstring $(MAKECMDGOALS), maintainer-clean debian redhat),)
include config/Makefile.$(ARCH)
endif
ifeq ($(MAKECMDGOALS),clean)
include config/Makefile.$(ARCH)
endif

##### Include library dependencies for explicit linking #####

ifeq ($(EXPLICITLINK),yes)
include config/Makefile.depend
endif
ifneq ($(findstring map, $(MAKECMDGOALS)),)
include config/Makefile.depend
endif

##### Allow local macros #####

-include MyConfig.mk

##### Modules to build #####

MODULES       = build cint metautils pcre utils base cont meta net auth zip \
                clib matrix newdelete hist tree freetype graf g3d gpad gui \
                minuit histpainter treeplayer treeviewer physics postscript \
                rint html eg geom geompainter vmc fumili mlp ged quadp \
                guibuilder xml foam splot smatrix sql tmva geombuilder spectrum \
                spectrumpainter fitpanel math io

ifeq ($(ARCH),win32)
MODULES      += winnt win32gdk
SYSTEML       = $(WINNTL)
SYSTEMO       = $(WINNTO)
SYSTEMDO      = $(WINNTDO)
else
ifeq ($(ARCH),win32gcc)
MODULES      += unix x11 x11ttf x3d rootx
SYSTEML       = $(UNIXL)
SYSTEMO       = $(UNIXO)
SYSTEMDO      = $(UNIXDO)
else
MODULES      += unix x11 x11ttf x3d rootx
SYSTEML       = $(UNIXL)
SYSTEMO       = $(UNIXO)
SYSTEMDO      = $(UNIXDO)
endif
endif
ifeq ($(BUILDGL),yes)
MODULES      += gl
endif
ifeq ($(BUILDMYSQL),yes)
MODULES      += mysql
endif
ifeq ($(BUILDORACLE),yes)
MODULES      += oracle
endif
ifeq ($(BUILDPGSQL),yes)
MODULES      += pgsql
endif
ifeq ($(BUILDSAPDB),yes)
MODULES      += sapdb
endif
ifeq ($(BUILDODBC),yes)
MODULES      += odbc
endif
ifeq ($(BUILDRFIO),yes)
MODULES      += rfio
endif
ifeq ($(BUILDCASTOR),yes)
MODULES      += castor
endif
ifeq ($(BUILDDCAP),yes)
MODULES      += dcache
endif
ifeq ($(BUILDGFAL),yes)
MODULES      += gfal
endif
ifeq ($(BUILDG4ROOT),yes)
MODULES      += g4root
endif
ifeq ($(BUILDCHIRP),yes)
MODULES      += chirp
endif
ifeq ($(BUILDASIMAGE),yes)
MODULES      += asimage
endif
ifeq ($(ENABLETHREAD),yes)
MODULES      += thread
MODULES      += proof
endif
ifeq ($(BUILDFPYTHIA6),yes)
MODULES      += pythia6
endif
ifeq ($(BUILDFFTW3),yes)
MODULES      += fftw
endif
ifeq ($(BUILDPYTHON),yes)
MODULES      += pyroot
endif
ifeq ($(BUILDRUBY),yes)
MODULES      += ruby
endif
ifeq ($(BUILDXML),yes)
MODULES      += xmlparser
endif
ifeq ($(BUILDQT),yes)
MODULES      += qt qtroot
endif
ifeq ($(BUILDQTGSI),yes)
MODULES      += qtgsi
endif
ifeq ($(BUILDMATHCORE),yes)
MODULES      += mathcore
endif
ifeq ($(BUILDMATHMORE),yes)
MODULES      += mathmore
endif
ifeq ($(BUILDREFLEX),yes)
MODULES      += reflex
endif
ifeq ($(BUILDMINUIT2),yes)
MODULES      += minuit2
endif
ifeq ($(BUILDUNURAN),yes)
MODULES      += unuran
endif
ifeq ($(BUILDCINT7),yes)
MODULES      += cint7
endif
ifeq ($(BUILDCINTEX),yes)
MODULES      += cintex
endif
ifeq ($(BUILDROOFIT),yes)
MODULES      += roofit
endif
ifeq ($(BUILDGDML),yes)
MODULES      += gdml
endif
ifeq ($(BUILDTABLE),yes)
MODULES      += table
endif
ifeq ($(BUILDSRPUTIL),yes)
MODULES      += srputils
endif
ifeq ($(BUILDKRB5),yes)
MODULES      += krb5auth
endif
ifeq ($(BUILDLDAP),yes)
MODULES      += ldap
endif
ifeq ($(BUILDMONALISA),yes)
MODULES      += monalisa
endif
ifeq ($(BUILDGLOBUS),yes)
MODULES      += globusauth
endif
ifeq ($(BUILDHBOOK),yes)
MODULES      += hbook
endif
ifeq ($(BUILDXRD),yes)
ifneq ($(XROOTDDIR),)
MODULES      += netx
else
MODULES      += xrootd netx
endif
endif
ifeq ($(BUILDALIEN),yes)
MODULES      += alien
endif
ifeq ($(BUILDCLARENS),yes)
MODULES      += clarens
endif
ifeq ($(BUILDPEAC),yes)
MODULES      += peac
endif
ifneq ($(ARCH),win32)
MODULES      += rpdutils rootd proofd
endif
ifeq ($(BUILDXRD),yes)
ifeq ($(ARCH),win32)
MODULES      += proofd
endif
MODULES      += proofx
endif

-include MyModules.mk   # allow local modules

ifneq ($(findstring $(MAKECMDGOALS),distclean maintainer-clean),)
MODULES      += unix winnt x11 x11ttf win32gdk gl rfio castor thread \
                pythia6 table mysql pgsql sapdb srputils x3d \
                rootx rootd proofd proof dcache chirp hbook asimage \
                ldap mlp krb5auth rpdutils globusauth pyroot ruby gfal \
                qt qtroot qtgsi xrootd netx proofx alien clarens peac oracle \
                xmlparser mathcore mathmore reflex cintex roofit minuit2 \
                monalisa fftw odbc unuran gdml g4root cint7
MODULES      := $(sort $(MODULES))   # removes duplicates
endif

MODULES      += main   # must be last, $(ALLLIBS) must be fully formed

##### ROOT libraries #####

LPATH         = lib

ifneq ($(PLATFORM),win32)
RPATH        := -L$(LPATH)
CINTLIBS     := -lCint
CINT7LIBS    := -lCint7 -lReflex
NEWLIBS      := -lNew
ROOTLIBS     := -lCore -lCint -lHist -lGraf -lGraf3d -lGpad -lTree -lMatrix
ifneq ($(ROOTDICTTYPE),cint)
ROOTLIBS     += -lCintex -lReflex
endif
RINTLIBS     := -lRint
else
CINTLIBS     := $(LPATH)/libCint.lib
CINT7LIBS    := $(LPATH)/libCint7.lib $(LPATH)/libReflex.lib
NEWLIBS      := $(LPATH)/libNew.lib
ROOTLIBS     := $(LPATH)/libCore.lib $(LPATH)/libCint.lib \
                $(LPATH)/libHist.lib $(LPATH)/libGraf.lib \
                $(LPATH)/libGraf3d.lib $(LPATH)/libGpad.lib \
                $(LPATH)/libTree.lib $(LPATH)/libMatrix.lib
ifneq ($(ROOTDICTTYPE),cint)
ROOTLIBS     += $(LPATH)/libCintex.lib $(LPATH)/libReflex.lib
endif
RINTLIBS     := $(LPATH)/libRint.lib
endif

# ROOTLIBSDEP is intended to match the content of ROOTLIBS
ROOTLIBSDEP   = $(ORDER_) $(CORELIB) $(CINTLIB) $(HISTLIB) \
                $(GRAFLIB) $(G3DLIB) $(GPADLIB) $(TREELIB) $(MATRIXLIB)
ifneq ($(ROOTDICTTYPE),cint)
ROOTLIBSDEP  += $(CINTEXLIB) $(REFLEXLIB)
endif

# Force linking of not referenced libraries
ifeq ($(FORCELINK),yes)
ifeq ($(PLATFORM),aix5)
ROOTULIBS    := -Wl,-u,.G__cpp_setupG__Hist     \
                -Wl,-u,.G__cpp_setupG__Graf1    \
                -Wl,-u,.G__cpp_setupG__G3D      \
                -Wl,-u,.G__cpp_setupG__GPad     \
                -Wl,-u,.G__cpp_setupG__Tree     \
                -Wl,-u,.G__cpp_setupG__Matrix
else
ROOTULIBS    := -Wl,-u,_G__cpp_setupG__Hist    \
                -Wl,-u,_G__cpp_setupG__Graf1   \
                -Wl,-u,_G__cpp_setupG__G3D     \
                -Wl,-u,_G__cpp_setupG__GPad    \
                -Wl,-u,_G__cpp_setupG__Tree    \
                -Wl,-u,_G__cpp_setupG__Matrix
endif
endif
ifeq ($(PLATFORM),win32)
ROOTULIBS    := -include:_G__cpp_setupG__Hist    \
                -include:_G__cpp_setupG__Graf1   \
                -include:_G__cpp_setupG__G3D     \
                -include:_G__cpp_setupG__GPad    \
                -include:_G__cpp_setupG__Tree    \
                -include:_G__cpp_setupG__Matrix
endif

##### Compiler output option #####

CXXOUT ?= -o # keep whitespace after "-o"

##### gcc version #####

ifneq ($(findstring gnu,$(COMPILER)),)
GCC_MAJOR     := $(shell $(CXX) -dumpversion 2>&1 | cut -d'.' -f1)
GCC_MINOR     := $(shell $(CXX) -dumpversion 2>&1 | cut -d'.' -f2)
GCC_PATCH     := $(shell $(CXX) -dumpversion 2>&1 | cut -d'.' -f3)
GCC_VERS      := gcc-$(GCC_MAJOR).$(GCC_MINOR)
GCC_VERS_FULL := gcc-$(GCC_MAJOR).$(GCC_MINOR).$(GCC_PATCH)

# Precompiled headers for gcc
ifeq ($(GCC_MAJOR),4)
PCHSUPPORTED  := $(ENABLEPCH)
endif
ifeq ($(PCHSUPPORTED),yes)
PCHFILE        = include/precompile.h.gch
PCHCXXFLAGS    = -DUSEPCH -include precompile.h
PCHEXTRAOBJBUILD = $(CXX) $(CXXFLAGS) -DUSEPCH $(OPT) -x c++-header \
                   -c include/precompile.h $(CXXOUT)$(PCHFILE) \
                   && touch $(PCHEXTRAOBJ)
endif
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

ifeq ($(GCC_MAJOR),3)
ifneq ($(GCC_MINOR),0)
LIBFRTBEGIN  := $(shell $(F77) -print-file-name=libfrtbegin.a)
F77LIBS      := $(LIBFRTBEGIN) $(F77LIBS)
endif
endif
ifeq ($(GCC_MAJOR),4)
ifeq ($(F77),g77)
LIBFRTBEGIN  := $(shell $(F77) -print-file-name=libfrtbegin.a)
F77LIBS      := $(LIBFRTBEGIN) $(F77LIBS)
endif
endif

##### Utilities #####

ROOTCINTTMP   = $(ROOTCINTTMPEXE) $(addprefix -,$(ROOTDICTTYPE))
MAKEDEP       = $(RMKDEP)
MAKELIB       = build/unix/makelib.sh $(MKLIBOPTIONS)
MAKEDIST      = build/unix/makedist.sh
MAKEDISTSRC   = build/unix/makedistsrc.sh
MAKEVERSION   = build/unix/makeversion.sh
IMPORTCINT    = build/unix/importcint.sh
MAKECOMPDATA  = build/unix/compiledata.sh
MAKECHANGELOG = build/unix/makechangelog.sh
MAKEHTML      = build/unix/makehtml.sh
MAKELOGHTML   = build/unix/makeloghtml.sh
MAKECINTDLL   = build/unix/makecintdll.sh
MAKESTATIC    = build/unix/makestatic.sh
RECONFIGURE   = build/unix/reconfigure.sh
ifeq ($(PLATFORM),win32)
MAKELIB       = build/win/makelib.sh
MAKECOMPDATA  = build/win/compiledata.sh
endif

##### Compiler directives and run-control file #####

COMPILEDATA   = include/compiledata.h
ROOTRC        = etc/system.rootrc
ROOTMAP       = etc/system.rootmap

##### libCore #####

COREL         = $(BASEL1) $(BASEL2) $(BASEL3) $(CONTL) $(METAL) $(NETL) \
                $(SYSTEML) $(CLIBL) $(METAUTILSL)
COREO         = $(BASEO) $(CONTO) $(METAO) $(NETO) $(SYSTEMO) $(ZIPO) $(CLIBO) \
                $(METAUTILSO)
COREDO        = $(BASEDO) $(CONTDO) $(METADO) $(NETDO) $(SYSTEMDO) $(CLIBDO) \
                $(METAUTILSDO)

CORELIB      := $(LPATH)/libCore.$(SOEXT)
ifneq ($(BUILTINZLIB),yes)
CORELIBEXTRA += $(ZLIBCLILIB)
endif

##### In case shared libs need to resolve all symbols (e.g.: aix, win32) #####

ifeq ($(EXPLICITLINK),yes)
MAINLIBS     := $(CORELIB) $(CINTLIB)
ifneq ($(ROOTDICTTYPE),cint)
MAINLIBS     += $(CINTEXLIB) $(REFLEXLIB)
endif
else
MAINLIBS      =
endif

##### pre-compiled header support #####

ifeq ($(PCHSUPPORTED),yes)
include config/Makefile.precomp
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
G__%.o: G__%.cxx
	$(MAKEDEP) -R -f$(patsubst %.o,%.d,$@) -Y -w 1000 -- \
	   $(CXXFLAGS) -D__cplusplus -I$(CINTDIR)/lib/prec_stl \
	   -I$(CINTDIR)/stl -- $<
	$(CXX) $(NOOPT) $(CXXFLAGS) -I. $(CXXOUT)$@ -c $<

cint/%.o: cint/%.cxx
	$(MAKEDEP) -R -fcint/$*.d -Y -w 1000 -- $(CINTCXXFLAGS) -D__cplusplus -- $<
	$(CXX) $(OPT) $(CINTCXXFLAGS) $(CXXOUT)$@ -c $<

cint/%.o: cint/%.c
	$(MAKEDEP) -R -fcint/$*.d -Y -w 1000 -- $(CINTCFLAGS) -- $<
	$(CC) $(OPT) $(CINTCFLAGS) $(CXXOUT)$@ -c $<

cint7/%.o: cint7/%.cxx
	$(MAKEDEP) -R -fcint7/$*.d -Y -w 1000 -- $(CINT7CXXFLAGS) -D__cplusplus -- $<
	$(CXX) $(OPT) $(CINT7CXXFLAGS) $(CXXOUT)$@ -c $<

cint7/%.o: cint7/%.c
	$(MAKEDEP) -R -fcint7/$*.d -Y -w 1000 -- $(CINT7CFLAGS) -- $<
	$(CC) $(OPT) $(CINT7CFLAGS) $(CXXOUT)$@ -c $<

build/%.o: build/%.cxx
	$(CXX) $(OPT) $(CXXFLAGS) $(CXXOUT)$@ -c $<

build/%.o: build/%.c
	$(CC) $(OPT) $(CFLAGS) $(CXXOUT)$@ -c $<

%.o: %.cxx
	$(MAKEDEP) -R -f$*.d -Y -w 1000 -- $(CXXFLAGS) -D__cplusplus -- $<
	$(CXX) $(OPT) $(CXXFLAGS) $(PCHCXXFLAGS) $(CXXOUT)$@ -c $<

%.o: %.c
	$(MAKEDEP) -R -f$*.d -Y -w 1000 -- $(CFLAGS) -- $<
	$(CC) $(OPT) $(CFLAGS) $(CXXOUT)$@ -c $<

%.o: %.f
ifeq ($(F77),f2c)
	f2c -a -A $<
	$(CC) $(F77OPT) $(CFLAGS) $(CXXOUT)$@ -c $*.c
else
	$(F77) $(F77OPT) $(F77FLAGS) $(CXXOUT)$@ -c $<
endif

##### TARGETS #####
.PHONY:         all fast config rootcint rootlibs rootexecs dist distsrc \
                clean distclean maintainer-clean compiledata importcint \
                version html changelog install uninstall showbuild \
                static map debian redhat skip $(POSTBIN) \
                $(patsubst %,all-%,$(MODULES)) \
                $(patsubst %,map-%,$(MODULES)) \
                $(patsubst %,clean-%,$(MODULES)) \
                $(patsubst %,distclean-%,$(MODULES))

ifneq ($(findstring map, $(MAKECMDGOALS)),)
.NOTPARALLEL:
endif

all:            rootexecs $(POSTBIN)

fast:           rootexecs

skip:
		@true;

include $(patsubst %,%/Module.mk,$(MODULES))
include cint/cintdlls.mk

-include MyRules.mk            # allow local rules

ifeq ($(findstring $(MAKECMDGOALS),clean distclean maintainer-clean dist \
      distsrc version importcint importcint7 install uninstall showbuild changelog html \
      debian redhat),)
ifeq ($(findstring clean-,$(MAKECMDGOALS)),)
ifeq ($(findstring skip,$(MAKECMDGOALS))$(findstring fast,$(MAKECMDGOALS)),)
-include $(INCLUDEFILES)
endif
ifeq ($(PCHSUPPORTED),yes)
INCLUDEPCHRULES = yes
include config/Makefile.precomp
endif
-include build/dummy.d          # must be last include
endif
endif

rootcint:       all-cint all-utils

rootlibs:       rootcint compiledata $(ALLLIBS)

rootexecs:      rootlibs $(ALLEXECS)

compiledata:    $(COMPILEDATA)

config config/Makefile.:
ifeq ($(BUILDING_WITHIN_IDE),)
	@(if [ ! -f config/Makefile.config ] ; then \
	   echo ""; echo "Please, run ./configure first"; echo ""; \
	   exit 1; \
	fi)
else
# Building from within an IDE, running configure
	@(if [ ! -f config/Makefile.config ] ; then \
	   ./configure --build=debug `cat config.status 2>/dev/null`; \
	fi)
endif

# Target Makefile is synonym for "run (re-)configure"
# Makefile is target as we need to re-parse dependencies after
# configure is run (as RConfigure.h changed etc)
config/Makefile.config include/RConfigure.h etc/system.rootauthrc \
  etc/system.rootdaemonrc etc/root.mimes $(ROOTRC) bin/root-config: Makefile

ifeq ($(findstring $(MAKECMDGOALS),distclean maintainer-clean debian redhat),)
Makefile: configure config/rootrc.in config/RConfigure.in config/Makefile.in \
  config/root-config.in config/rootauthrc.in config/rootdaemonrc.in \
  config/mimes.unix.in config/mimes.win32.in config.status
	@(if [ ! -x $(RECONFIGURE) ] || ! $(RECONFIGURE) "$?"; then \
	   echo ""; echo "Please, run ./configure again as config option files ($?) have changed."; \
	   echo ""; exit 1; \
	 fi)
endif

$(COMPILEDATA): config/Makefile.$(ARCH) $(MAKECOMPDATA)
	@$(MAKECOMPDATA) $(COMPILEDATA) "$(CXX)" "$(OPTFLAGS)" "$(DEBUGFLAGS)" \
	   "$(CXXFLAGS)" "$(SOFLAGS)" "$(LDFLAGS)" "$(SOEXT)" "$(SYSLIBS)" \
	   "$(LIBDIR)" "$(ROOTLIBS)" "$(RINTLIBS)" "$(INCDIR)" \
	   "$(MAKESHAREDLIB)" "$(MAKEEXE)" "$(ARCH)" "$(ROOTBUILD)"

build/dummy.d: config Makefile $(ALLHDRS) $(RMKDEP) $(BINDEXP) $(PCHDEP)
	@(if [ ! -f $@ ] ; then \
	   touch $@; \
	fi)

$(CORELIB): $(COREO) $(COREDO) $(CINTLIB) $(PCREDEP) $(CORELIBDEP)
ifneq ($(ARCH),alphacxx6)
	@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
	   "$(SOFLAGS)" libCore.$(SOEXT) $@ "$(COREO) $(COREDO)" \
	   "$(CORELIBEXTRA) $(PCRELDFLAGS) $(PCRELIB) $(CRYPTLIBS)"
else
	@$(MAKELIB) $(PLATFORM) $(LD) "$(CORELDFLAGS)" \
	   "$(SOFLAGS)" libCore.$(SOEXT) $@ "$(COREO) $(COREDO)" \
	   "$(CORELIBEXTRA) $(PCRELDFLAGS) $(PCRELIB) $(CRYPTLIBS)"
endif

map::   $(RLIBMAP)
	$(RLIBMAP) -r $(ROOTMAP) -l $(CORELIB) -d $(CORELIBDEP) -c $(COREL)

dist:
	@rm -f $(ROOTMAP)
	@$(MAKE) map
	@$(MAKEDIST) $(GCC_VERS)

distsrc:
	@$(MAKEDISTSRC)

distmsi: build/package/msi/makemsi$(EXEEXT)
	@rm -f $(ROOTMAP)
	@$(MAKE) map
	$(MAKEDIST) -msi

build/package/msi/makemsi$(EXEEXT): build/package/msi/makemsi.cxx build/version_number
	@vers=`sed 's|\(.*\)/\(.*\)|\1.\2|' < build/version_number` && \
	$(CXX) -DPRODUCT=\"ROOT\" -DVERSION=\"$$vers\" $(CXXFLAGS) Rpcrt4.lib build/package/msi/makemsi.cxx -Fe$@

rebase: $(ALLLIBS) $(ALLEXECS)
	@echo -n "Rebasing binaries... "
	@rebase -b 0x71000000 bin/*.exe bin/*.dll
	@echo done.

debian:
	@if [ ! -x `which dpkg-buildpackage` ] || [ ! -x `which dh_testdir` ]; then \
	   echo "You must have debhelper installed to make the "; \
	   echo "Debian GNU/Linux packages"; exit 1; fi
	@echo "OK, you're on a Debian GNU/Linux system - cool"
	@vers=`sed 's|\(.*\)/\(.*\)|\1.\2|' < build/version_number` ; \
	  dirvers=`basename $$PWD | sed 's|root-\(.*\)|\1|'` ; \
	  if [ "$$vers" != "$$dirvers" ] ; then \
	    echo "Must have ROOT source tree in root-$$vers" ; \
	    echo "Please rename this directory to `basename $$PWD` to"; \
	    echo "root-$$vers and try again"; exit 1 ; fi
	rm -rf debian
	build/package/lib/makedebdir.sh
	fakeroot debian/rules debian/control
	dpkg-buildpackage -rfakeroot -us -uc -i"G__|^debian|root-bin.png|\.d$$"
	@echo "Debian GNU/Linux packages done. They are put in '../'"

redhat:
	@if [ ! -x `which rpm` ]; then \
	   echo "You must have rpm installed to make the Redhat package"; \
	   exit 1; fi
	@echo "OK, you have RPM on your system - good"
	build/package/lib/makerpmspec.sh
	@echo "To build the packages, make a gzipped tar ball of the sources"
	@vers=`sed 's|\(.*\)/\(.*\)|\1.\2|' < build/version_number` ; \
	  echo "called root_v$$vers.source.tar.gz and put it in you RPM "
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

rootdrpm:
	@if [ ! -x `which rpm` ]; then \
	   echo "You must have rpm installed to make the root-rootd package"; \
	   exit 1; fi
	@echo "OK, you have RPM on your system - good"
	@if [ "x$(ARCOMP)" != "x" ]; then \
	    rm -f rootd-$(ARCOMP)-*-$(ROOTDRPMREL).spec ; \
	else  \
	    rm -f rootd-*-$(ROOTDRPMREL).spec ; \
	fi
	build/package/lib/makerpmspecs.sh rpm build/package/common \
	        build/package/rpm root-rootd >> root-rootd.spec.tmp
	@if [ "x$(ARCOMP)" != "x" ]; then \
	    echo "Architecture+compiler flag: $(ARCOMP)" ; \
	fi
	@if [ "x$(ROOTDRPMREL)" != "x" ]; then \
	    echo "RPM release set to: $(ROOTDRPMREL)" ; \
	fi
	@if [ ! -d "/tmp/rootdrpm" ]; then \
	   echo "Creating build directory /tmp/rootdrpm ..."; \
	   mkdir -p /tmp/rootdrpm; \
	   chmod 0777 /tmp/rootdrpm; \
	fi
	@echo "Make the substitutions ..."
	@vers=`sed 's|\(.*\)/\(.*\)|\1.\2|' < build/version_number` ; \
	        echo "Version is $$vers ... " ; \
	   rootdir=`echo $(PWD)` ; \
	        echo "Rootdir: $$rootdir" ; \
	   prefix=`dirname $(DESTDIR)$(BINDIR)` ; \
	        echo "Prefix: $$prefix" ; \
	   etcdir=`echo $(DESTDIR)$(ETCDIR)` ; \
	        echo "Etcdir: $$etcdir" ; \
	   arcomp="" ; \
	   if [ "x$(ARCOMP)" != "x" ]; then \
	      arcomp=`echo -$(ARCOMP)` ; \
	   fi ; \
	   sed -e "s|@version@|$$vers|" \
	       -e "s|@rootdir@|$$rootdir|" \
	       -e "s|@prefix@|$$prefix|" \
	       -e "s|@etcdir@|$$etcdir|" \
	       -e "s|@arcomp@|$$arcomp|" \
	       -e "s|@release@|$(ROOTDRPMREL)|" \
	           < root-rootd.spec.tmp \
	           > rootd$$arcomp-$$vers-$(ROOTDRPMREL).spec
	@echo " "
	@echo "To build the RPM package run:"
	@specfile=`ls -1 rootd*$(ARCOMP)*-$(ROOTDRPMREL).spec` ; \
	        echo "   rpmbuild -ba $$specfile "
	@rm -f root-rootd.spec.tmp root-rootd.spec.tmp.*
	@if [ "x$(ARCOMP)" == "x" ]; then \
	    echo " " ; \
	    echo "To add a flag to the package name re-run with" ; \
	    echo " " ; \
	    echo "  make rootdrpm ARCOMP=<flag> " ; \
	    echo " " ; \
	    echo "The RPM will then be called rootd-<flag> " ; \
	    echo " " ; \
	fi
	@if [ "x$(ROOTDRPMREL)" == "x1" ]; then \
	    echo " " ; \
	    echo "To change the release version number re-run with" ; \
	    echo " " ; \
	    echo "  make rootdrpm ROOTDRPMREL=<new_release_version_number> " ; \
	    echo " " ; \
	fi

clean::
	@rm -f __compiledata *~ core $(PCHFILE)

ifeq ($(CXX),KCC)
clean::
	@(find . -name "ti_files" -exec rm -rf {} \; >/dev/null 2>&1;true)
endif
ifeq ($(SUNCC5),true)
clean::
	@(find . -name "SunWS_cache" -exec rm -rf {} \; >/dev/null 2>&1;true)
endif

distclean:: clean
	-@mv -f include/RConfigure.h include/RConfigure.h-
	@rm -f include/*.h $(ROOTMAP) $(CORELIB)
	-@mv -f include/RConfigure.h- include/RConfigure.h
	@rm -f bin/*.dll bin/*.exp bin/*.lib bin/*.pdb \
               lib/*.def lib/*.exp lib/*.lib lib/*.dll.a \
               *.def .def
ifeq ($(PLATFORM),macosx)
	@rm -f lib/*.so
endif
	-@mv -f tutorials/gallery.root tutorials/gallery.root-
	-@mv -f tutorials/mlp/mlpHiggs.root tutorials/mlp/mlpHiggs.root-
	-@mv -f tutorials/quadp/stock.root tutorials/quadp/stock.root-
	@(find tutorials -name "*.root" -exec rm -rf {} \; >/dev/null 2>&1;true)
	@(find tutorials -name "*.ps" -exec rm -rf {} \; >/dev/null 2>&1;true)
	@(find tutorials -name "*.gif" -exec rm -rf {} \; >/dev/null 2>&1;true)
	@(find tutorials -name "so_locations" -exec rm -rf {} \; >/dev/null 2>&1;true)
	@(find tutorials -name "pca.C" -exec rm -rf {} \; >/dev/null 2>&1;true)
	@(find tutorials -name "*.so" -exec rm -rf {} \; >/dev/null 2>&1;true)
	@(find tutorials -name "work.pc" -exec rm -rf {} \; >/dev/null 2>&1;true)
	@(find tutorials -name "work.pcl" -exec rm -rf {} \; >/dev/null 2>&1;true)
	-@mv -f tutorials/gallery.root- tutorials/gallery.root
	-@mv -f tutorials/mlp/mlpHiggs.root- tutorials/mlp/mlpHiggs.root
	-@mv -f tutorials/quadp/stock.root- tutorials/quadp/stock.root
	@rm -f bin/roota bin/proofserva lib/libRoot.a
	@rm -f $(CINTDIR)/include/*.dll $(CINTDIR)/include/sys/*.dll
	@rm -f $(CINTDIR)/stl/*.dll README/ChangeLog build/dummy.d
	@rm -f $(CINTDIR)/lib/posix/a.out $(CINTDIR)/include/*.so*
	@rm -f etc/daemons/rootd.rc.d etc/daemons/rootd.xinetd
	@rm -f etc/daemons/proofd.rc.d etc/daemons/proofd.xinetd
	@rm -f etc/daemons/olbd.rc.d etc/daemons/xrootd.rc.d
	@(find . -path '*/daemons' -prune -o -name *.d -exec rm -rf {} \; >/dev/null 2>&1;true)
	@(find . -name *.o -exec rm -rf {} \; >/dev/null 2>&1;true)
	-@cd test && $(MAKE) distclean

maintainer-clean:: distclean
	@rm -rf bin lib include htmldoc system.rootrc config/Makefile.config \
	   $(ROOTRC) etc/system.rootauthrc etc/system.rootdaemonrc \
	   etc/root.mimes build/misc/root-help.el \
	   rootd/misc/rootd.rc.d build-arch-stamp build-indep-stamp \
	   configure-stamp build-arch-cint-stamp config.status config.log

version: $(CINTTMP)
	@$(MAKEVERSION)

static: rootlibs
	@$(MAKESTATIC) $(PLATFORM) "$(CXX)" "$(CC)" "$(LD)" "$(LDFLAGS)" \
	   "$(XLIBS)" "$(SYSLIBS)"

importcint: distclean-cint
	@$(IMPORTCINT)

importcint7: distclean-cint7
	@$(IMPORTCINT) cint7

changelog:
	@$(MAKECHANGELOG)

html: $(ROOTEXE) changelog
	@$(MAKELOGHTML)
	@$(MAKEHTML)

install: all
	@rm -f $(ROOTMAP)
	@$(MAKE) map
	@if [ -d $(BINDIR) ]; then \
	   inode1=`ls -id $(BINDIR) | awk '{ print $$1 }'`; \
	fi; \
	inode2=`ls -id $$PWD/bin | awk '{ print $$1 }'`; \
	if [ -d $(BINDIR) ] && [ "x$$inode1" = "x$$inode2" ]; then \
	   echo "Everything already installed..."; \
	else \
	   echo "Installing binaries in $(DESTDIR)$(BINDIR)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(BINDIR); \
	   $(INSTALL) $(CINT)                   $(DESTDIR)$(BINDIR); \
	   $(INSTALL) $(MAKECINT)               $(DESTDIR)$(BINDIR); \
	   $(INSTALL) $(ROOTCINTEXE)            $(DESTDIR)$(BINDIR); \
	   $(INSTALL) $(RLIBMAP)                $(DESTDIR)$(BINDIR); \
	   $(INSTALL) $(RMKDEP)                 $(DESTDIR)$(BINDIR); \
	   if [ "x$(BINDEXP)" != "x" ] ; then \
	      $(INSTALL) $(BINDEXP)             $(DESTDIR)$(BINDIR); \
           fi; \
	   $(INSTALL) bin/root-config           $(DESTDIR)$(BINDIR); \
	   $(INSTALL) bin/memprobe              $(DESTDIR)$(BINDIR); \
	   $(INSTALL) bin/thisroot.sh           $(DESTDIR)$(BINDIR); \
	   $(INSTALL) bin/thisroot.csh          $(DESTDIR)$(BINDIR); \
	   $(INSTALL) $(ALLEXECS)               $(DESTDIR)$(BINDIR); \
	   echo "Installing libraries in $(DESTDIR)$(LIBDIR)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(LIBDIR); \
	   $(INSTALLDATA) lib/*                 $(DESTDIR)$(LIBDIR); \
	   if [ x"$(ARCH)" = x"win32gcc" ]; then \
	      $(INSTALLDATA) bin/*.dll             $(DESTDIR)$(BINDIR); \
	      for f in $(DESTDIR)$(LIBDIR)/*.dll; do \
	         bindll=$$(basename $$f | sed 's,\..*$$,,'); \
	         bindll=$$(ls $(DESTDIR)$(BINDIR)/$${bindll}.*dll); \
	         ln -sf $${bindll} $$f; \
	      done; \
           elif [ x"$(PLATFORM)" = x"win32" ]; then \
	      $(INSTALLDATA) $(GDKDLL)             $(DESTDIR)$(BINDIR); \
	      $(INSTALLDATA) $(GDKDLLS)            $(DESTDIR)$(BINDIR); \
	   fi; \
	   echo "Installing headers in $(DESTDIR)$(INCDIR)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(INCDIR); \
	   $(INSTALLDATA) include/*             $(DESTDIR)$(INCDIR); \
	   echo "Installing main/src/rmain.cxx in $(DESTDIR)$(INCDIR)"; \
	   $(INSTALLDATA) main/src/rmain.cxx    $(DESTDIR)$(INCDIR); \
	   echo "Installing cint/include cint/lib and cint/stl in $(DESTDIR)$(CINTINCDIR)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(CINTINCDIR); \
	   $(INSTALLDATA) cint/include          $(DESTDIR)$(CINTINCDIR); \
	   $(INSTALLDATA) cint/lib              $(DESTDIR)$(CINTINCDIR); \
	   $(INSTALLDATA) cint/stl              $(DESTDIR)$(CINTINCDIR); \
	   find $(DESTDIR)$(CINTINCDIR) -name CVS -exec rm -rf {} \; >/dev/null 2>&1; \
	   echo "Installing icons in $(DESTDIR)$(ICONPATH)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(ICONPATH); \
	   $(INSTALLDATA) icons/*.xpm           $(DESTDIR)$(ICONPATH); \
	   $(INSTALLDATA) icons/*.png           $(DESTDIR)$(ICONPATH); \
	   $(INSTALLDATA) icons/*.gif           $(DESTDIR)$(ICONPATH); \
	   echo "Installing fonts in $(DESTDIR)$(TTFFONTDIR)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(TTFFONTDIR); \
	   $(INSTALLDATA) fonts/*               $(DESTDIR)$(TTFFONTDIR); \
	   find $(DESTDIR)$(TTFFONTDIR) -name CVS -exec rm -rf {} \; >/dev/null 2>&1; \
	   echo "Installing misc docs in $(DESTDIR)$(DOCDIR)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(DOCDIR); \
	   $(INSTALLDATA) LICENSE               $(DESTDIR)$(DOCDIR); \
	   $(INSTALLDATA) README/*              $(DESTDIR)$(DOCDIR); \
	   find $(DESTDIR)$(DOCDIR) -name CVS -exec rm -rf {} \; >/dev/null 2>&1; \
	   echo "Installing tutorials in $(DESTDIR)$(TUTDIR)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(TUTDIR); \
	   $(INSTALLDATA) tutorials/*           $(DESTDIR)$(TUTDIR); \
	   find $(DESTDIR)$(TUTDIR) -name CVS -exec rm -rf {} \; >/dev/null 2>&1; \
	   echo "Installing tests in $(DESTDIR)$(TESTDIR)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(TESTDIR); \
	   $(INSTALLDATA) test/*                $(DESTDIR)$(TESTDIR); \
	   find $(DESTDIR)$(TESTDIR) -name CVS -exec rm -rf {} \; >/dev/null 2>&1; \
	   echo "Installing macros in $(DESTDIR)$(MACRODIR)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(MACRODIR); \
	   $(INSTALLDATA) macros/*              $(DESTDIR)$(MACRODIR); \
	   find $(DESTDIR)$(MACRODIR) -name CVS -exec rm -rf {} \; >/dev/null 2>&1; \
	   echo "Installing man(1) pages in $(DESTDIR)$(MANDIR)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(MANDIR); \
	   $(INSTALLDATA) man/man1/*            $(DESTDIR)$(MANDIR); \
	   find $(DESTDIR)$(MANDIR) -name CVS -exec rm -rf {} \; >/dev/null 2>&1; \
	   echo "Installing config files in $(DESTDIR)$(ETCDIR)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(ETCDIR); \
	   $(INSTALLDATA) etc/*                 $(DESTDIR)$(ETCDIR); \
	   find $(DESTDIR)$(ETCDIR) -name CVS -exec rm -rf {} \; >/dev/null 2>&1; \
	   echo "Installing Autoconf macro in $(DESTDIR)$(ACLOCALDIR)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(ACLOCALDIR); \
	   $(INSTALLDATA) build/misc/root.m4    $(DESTDIR)$(ACLOCALDIR); \
	   echo "Installing Emacs Lisp library in $(DESTDIR)$(ELISPDIR)"; \
	   $(INSTALLDIR)                          $(DESTDIR)$(ELISPDIR); \
	   $(INSTALLDATA) build/misc/root-help.el $(DESTDIR)$(ELISPDIR); \
	   echo "Installing GDML conversion scripts in $(DESTDIR)$(LIBDIR)"; \
	   $(INSTALLDATA) gdml/*.py               $(DESTDIR)$(LIBDIR); \
	   find $(DESTDIR)$(DATADIR) -name CVS -exec rm -rf {} \; >/dev/null 2>&1; \
	fi

uninstall:
	@if [ -d $(BINDIR) ]; then \
	   inode1=`ls -id $(BINDIR) | awk '{ print $$1 }'`; \
	fi; \
	inode2=`ls -id $$PWD/bin | awk '{ print $$1 }'`; \
	if [ -d $(BINDIR) ] && [ "x$$inode1" = "x$$inode2" ]; then \
	   $(MAKE) distclean ; \
	else \
	   rm -f $(DESTDIR)$(BINDIR)/`basename $(CINT)`; \
	   rm -f $(DESTDIR)$(BINDIR)/`basename $(MAKECINT)`; \
	   rm -f $(DESTDIR)$(BINDIR)/`basename $(ROOTCINTEXE)`; \
	   rm -f $(DESTDIR)$(BINDIR)/`basename $(RMKDEP)`; \
	   if [ "x$(BINDEXP)" != "x" ] ; then \
	      rm -f $(DESTDIR)$(BINDIR)/`basename $(BINDEXP)`; \
	   fi; \
	   rm -f $(DESTDIR)$(BINDIR)/root-config; \
	   rm -f $(DESTDIR)$(BINDIR)/memprobe; \
	   rm -f $(DESTDIR)$(BINDIR)/thisroot.sh; \
	   rm -f $(DESTDIR)$(BINDIR)/thisroot.csh; \
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
	   for i in icons/*.xpm ; do \
	      rm -fr $(DESTDIR)$(ICONPATH)/`basename $$i`; \
	   done; \
	   for i in icons/*.png ; do \
	      rm -fr $(DESTDIR)$(ICONPATH)/`basename $$i`; \
	   done; \
	   if test -d $(DESTDIR)$(ICONPATH) && \
	      test "x`ls $(DESTDIR)$(ICONPATH)`" = "x" ; then \
	      rm -rf $(DESTDIR)$(ICONPATH); \
	   fi ; \
	   rm -rf $(DESTDIR)$(TTFFONTDIR); \
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
	      rm -rf $(DESTDIR)$(ETCDIR)/`basename $$i`; \
	   done; \
	   if test -d $(DESTDIR)$(ETCDIR) && \
	      test "x`ls $(DESTDIR)$(ETCDIR)`" = "x" ; then \
	      rm -rf $(DESTDIR)$(ETCDIR); \
	   fi ; \
	   for i in build/misc/* ; do \
	      rm -rf $(DESTDIR)$(DATADIR)/`basename $$i`; \
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
	@echo "GCC_MAJOR          = $(GCC_MAJOR)"
	@echo "GCC_MINOR          = $(GCC_MINOR)"
	@echo ""
	@echo "CXXFLAGS           = $(CXXFLAGS)"
	@echo "CINTCXXFLAGS       = $(CINTCXXFLAGS)"
	@echo "EXTRA_CXXFLAGS     = $(EXTRA_CXXFLAGS)"
	@echo "CFLAGS             = $(CFLAGS)"
	@echo "CINTCFLAGS         = $(CINTCFLAGS)"
	@echo "EXTRA_CFLAGS       = $(EXTRA_CFLAGS)"
	@echo "F77FLAGS           = $(F77FLAGS)"
	@echo "LDFLAGS            = $(LDFLAGS)"
	@echo "F77LDFLAGS         = $(F77LDFLAGS)"
	@echo "EXTRA_LDFLAGS      = $(EXTRA_LDFLAGS)"
	@echo "SOFLAGS            = $(SOFLAGS)"
	@echo "SOEXT              = $(SOEXT)"
	@echo ""
	@echo "SYSLIBS            = $(SYSLIBS)"
	@echo "XLIBS              = $(XLIBS)"
	@echo "CILIBS             = $(CILIBS)"
	@echo "F77LIBS            = $(F77LIBS)"
	@echo ""
	@echo "FPYTHIA6LIBDIR     = $(FPYTHIA6LIBDIR)"
	@echo "TABLE              = $(TABLE)"
	@echo "XPMLIBDIR          = $(XPMLIBDIR)"
	@echo "XPMLIB             = $(XPMLIB)"
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
	@echo "GFALLIB            = $(GFALLIB)"
	@echo "G4INCDIR           = $(G4INCDIR)"
	@echo "G4LIBDIR           = $(G4LIBDIR)"
	@echo "MYSQLINCDIR        = $(MYSQLINCDIR)"
	@echo "ORACLEINCDIR       = $(ORACLEINCDIR)"
	@echo "PGSQLINCDIR        = $(PGSQLINCDIR)"
	@echo "PYTHONLIBDIR       = $(PYTHONLIBDIR)"
	@echo "PYTHONLIB          = $(PYTHONLIB)"
	@echo "PYTHONINCDIR       = $(PYTHONINCDIR)"
	@echo "RUBYLIBDIR         = $(RUBYLIBDIR)"
	@echo "RUBYLIB            = $(RUBYLIB)"
	@echo "RUBYINCDIR         = $(RUBYINCDIR)"
	@echo "FFTW3LIBDIR        = $(FFTW3LIBDIR)"
	@echo "FFTW3LIB           = $(FFTW3LIB)"
	@echo "FFTW3INCDIR        = $(FFTW3INCDIR)"
	@echo "SAPDBINCDIR        = $(SAPDBINCDIR)"
	@echo "SRPLIBDIR          = $(SRPLIBDIR)"
	@echo "SRPINCDIR          = $(SRPINCDIR)"
	@echo "SRPUTILLIB         = $(SRPUTILLIB)"
	@echo "LDAPINCDIR         = $(LDAPINCDIR)"
	@echo "LDAPCLILIB         = $(LDAPCLILIB)"
	@echo "MONALISAINCDIR     = $(MONALISAINCDIR)"
	@echo "MONALISAWSCLILIB   = $(MONALISAWSCLILIB)"
	@echo "MONALISACLILIB     = $(MONALISACLILIB)"
	@echo "QTLIBDIR           = $(QTLIBDIR)"
	@echo "QTLIB              = $(QTLIB)"
	@echo "QTINCDIR           = $(QTINCDIR)"
	@echo "AFSDIR             = $(AFSDIR)"
	@echo "SHADOWFLAGS        = $(SHADOWFLAGS)"
	@echo ""
	@echo "INSTALL            = $(INSTALL)"
	@echo "INSTALLDATA        = $(INSTALLDATA)"
	@echo "INSTALLDIR         = $(INSTALLDIR)"
	@echo "MAKEDEP            = $(MAKEDEP)"
	@echo "MAKELIB            = $(MAKELIB)"
	@echo "MAKEDIST           = $(MAKEDIST)"
	@echo "MAKEDISTSRC        = $(MAKEDISTSRC)"
	@echo "MAKEVERSION        = $(MAKEVERSION)"
	@echo "IMPORTCINT         = $(IMPORTCINT)"
	@echo ""
	@echo "The list of modules to be built:"
	@echo "--------------------------------"
	@echo "$(MODULES)"
