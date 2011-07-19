# Top level Makefile for ROOT System
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000


##### Check version of GNU make #####

MAKE_VERSION_MAJOR := $(word 1,$(subst ., ,$(MAKE_VERSION)))
MAKE_VERSION_MINOR := $(shell echo $(word 2,$(subst ., ,$(MAKE_VERSION))) | \
                              sed 's/\([0-9][0-9]*\).*/\1/')
MAKE_VERSION_MAJOR ?= 0
MAKE_VERSION_MINOR ?= 0
ORDER_ := $(shell test $(MAKE_VERSION_MAJOR) -gt 3 || \
                  test $(MAKE_VERSION_MAJOR) -eq 3 && \
                  test $(MAKE_VERSION_MINOR) -ge 80 && echo '|')

##### Include path/location macros (result of ./configure) #####

include config/Makefile.config

##### Include compiler overrides specified via ./configure #####
##### However, if we are building packages or cleaning, we #####
##### don't include this file since it may screw up things #####
##### Included before Makefile.$ARCH only because of f77   #####
##### if case has to be processed                          #####

ifeq ($(findstring $(MAKECMDGOALS), maintainer-clean debian redhat),)
include config/Makefile.comp
endif
ifeq ($(MAKECMDGOALS),clean)
include config/Makefile.comp
endif

##### Include machine dependent macros                     #####
##### However, if we are building packages or cleaning, we #####
##### don't include this file since it may screw up things #####

ifndef ROOT_SRCDIR
$(error Please run ./configure first)
endif

ifeq ($(findstring $(MAKECMDGOALS), maintainer-clean debian redhat),)
include $(ROOT_SRCDIR)/config/Makefile.$(ARCH)
endif
ifeq ($(MAKECMDGOALS),clean)
include $(ROOT_SRCDIR)/config/Makefile.$(ARCH)
endif

##### Include compiler overrides specified via ./configure #####
##### However, if we are building packages or cleaning, we #####
##### don't include this file since it may screw up things #####

ifeq ($(findstring $(MAKECMDGOALS), maintainer-clean debian redhat),)
include config/Makefile.comp
endif
ifeq ($(MAKECMDGOALS),clean)
include config/Makefile.comp
endif

##### Include library dependencies for explicit linking #####

MAKEFILEDEP = $(ROOT_SRCDIR)/config/Makefile.depend
include $(MAKEFILEDEP)

##### Allow local macros #####

-include MyConfig.mk

##### Modules to build #####

MODULES       = build cint/cint core/metautils core/pcre core/clib core/utils \
                core/textinput core/base core/cont core/meta core/thread \
                io/io math/mathcore net/net core/zip core/lzma math/matrix \
                core/newdelete hist/hist tree/tree graf2d/freetype \
                graf2d/graf graf2d/gpad graf3d/g3d \
                gui/gui math/minuit hist/histpainter tree/treeplayer \
                gui/ged tree/treeviewer math/physics graf2d/postscript \
                core/rint html montecarlo/eg \
                geom/geom geom/geompainter montecarlo/vmc \
                math/fumili math/mlp math/quadp net/auth gui/guibuilder io/xml \
                math/foam math/splot math/smatrix io/sql \
                geom/geombuilder hist/spectrum hist/spectrumpainter \
                gui/fitpanel proof/proof proof/proofplayer \
                gui/sessionviewer gui/guihtml gui/recorder

ifeq ($(ARCH),win32)
MODULES      += core/winnt graf2d/win32gdk
MODULES      := $(filter-out core/newdelete,$(MODULES))
SYSTEML       = $(WINNTL)
SYSTEMO       = $(WINNTO)
SYSTEMDO      = $(WINNTDO)
else
ifeq ($(ARCH),win32gcc)
MODULES      += core/unix
SYSTEML       = $(UNIXL)
SYSTEMO       = $(UNIXO)
SYSTEMDO      = $(UNIXDO)
else
MODULES      += core/unix
SYSTEML       = $(UNIXL)
SYSTEMO       = $(UNIXO)
SYSTEMDO      = $(UNIXDO)
endif
endif
ifeq ($(BUILDX11),yes)
MODULES      += graf2d/x11 graf2d/x11ttf graf3d/x3d rootx
endif
ifeq ($(BUILDGL),yes)
ifeq ($(BUILDFTGL),yes)
MODULES      += graf3d/ftgl
endif
ifeq ($(BUILDGLEW),yes)
MODULES      += graf3d/glew
endif
MODULES      += graf3d/gl graf3d/eve graf3d/gviz3d
endif
ifeq ($(BUILDMYSQL),yes)
MODULES      += sql/mysql
endif
ifeq ($(BUILDORACLE),yes)
MODULES      += sql/oracle
endif
ifeq ($(BUILDPGSQL),yes)
MODULES      += sql/pgsql
endif
ifeq ($(BUILDSAPDB),yes)
MODULES      += sql/sapdb
endif
ifeq ($(BUILDODBC),yes)
MODULES      += sql/odbc
endif
ifeq ($(BUILDRFIO),yes)
MODULES      += io/rfio
endif
ifeq ($(BUILDCASTOR),yes)
MODULES      += io/castor
endif
ifeq ($(BUILDDCAP),yes)
MODULES      += io/dcache
endif
ifeq ($(BUILDGFAL),yes)
MODULES      += io/gfal
endif
ifeq ($(BUILDGLITE),yes)
MODULES      += net/glite
endif
ifeq ($(BUILDBONJOUR),yes)
MODULES      += net/bonjour
endif
ifeq ($(BUILDCHIRP),yes)
MODULES      += io/chirp
endif
ifeq ($(BUILDHDFS),yes)
MODULES      += io/hdfs
endif
ifeq ($(BUILDMEMSTAT),yes)
MODULES      += misc/memstat
endif
ifeq ($(BUILDASIMAGE),yes)
MODULES      += graf2d/asimage
ifeq ($(BUILDFITSIO),yes)
MODULES      += graf2d/fitsio
endif
endif
ifeq ($(BUILDFPYTHIA6),yes)
MODULES      += montecarlo/pythia6
endif
ifeq ($(BUILDFPYTHIA8),yes)
MODULES      += montecarlo/pythia8
endif
ifeq ($(BUILDFFTW3),yes)
MODULES      += math/fftw
endif
ifeq ($(BUILDGVIZ),yes)
MODULES      += graf2d/gviz
endif
ifeq ($(BUILDPYTHON),yes)
MODULES      += bindings/pyroot
endif
ifeq ($(BUILDRUBY),yes)
MODULES      += bindings/ruby
endif
ifeq ($(BUILDXML),yes)
MODULES      += io/xmlparser
endif
ifeq ($(BUILDQT),yes)
MODULES      += graf2d/qt gui/qtroot
endif
ifeq ($(BUILDQTGSI),yes)
MODULES      += gui/qtgsi
endif
ifeq ($(BUILDGENVECTOR),yes)
MODULES      += math/genvector
endif
ifeq ($(BUILDMATHMORE),yes)
MODULES      += math/mathmore
endif
ifeq ($(BUILDREFLEX),yes)
# put reflex right in front of CINT; CINT needs it
MODULES      := $(subst cint/cint,cint/reflex cint/cint,$(MODULES))
endif
ifeq ($(BUILDMINUIT2),yes)
MODULES      += math/minuit2
endif
ifeq ($(BUILDUNURAN),yes)
MODULES      += math/unuran
endif
ifeq ($(BUILDCINTEX),yes)
MODULES      += cint/cintex
endif
ifeq ($(BUILDCLING),yes)
# to be added to the unconditional MODULES list above once cling is in trunk
MODULES      += cint/cling
endif
ifeq ($(BUILDROOFIT),yes)
MODULES      += roofit/roofitcore roofit/roofit roofit/roostats
ifeq ($(BUILDXML),yes)
MODULES      += roofit/histfactory
endif
endif
ifeq ($(BUILDGDML),yes)
MODULES      += geom/gdml
endif
ifeq ($(BUILDTABLE),yes)
MODULES      += misc/table
endif
ifeq ($(BUILDSRPUTIL),yes)
MODULES      += net/srputils
endif
ifeq ($(BUILDKRB5),yes)
MODULES      += net/krb5auth
endif
ifeq ($(BUILDLDAP),yes)
MODULES      += net/ldap
endif
ifeq ($(BUILDMONALISA),yes)
MODULES      += net/monalisa
endif
ifeq ($(BUILDGLOBUS),yes)
MODULES      += net/globusauth
endif
ifneq ($(F77),)
MODULES      += misc/minicern hist/hbook
endif
ifeq ($(BUILDXRD),yes)
MODULES      += net/xrootd
endif
ifeq ($(HASXRD),yes)
MODULES      += net/netx
ifeq ($(BUILDALIEN),yes)
MODULES      += net/alien
endif
endif
ifeq ($(BUILDCLARENS),yes)
MODULES      += proof/clarens
endif
ifeq ($(BUILDPEAC),yes)
MODULES      += proof/peac
endif
ifneq ($(ARCH),win32)
MODULES      += net/rpdutils net/rootd proof/proofd proof/pq2 proof/proofbench
endif
ifeq ($(BUILDTMVA),yes)
MODULES      += tmva math/genetic
endif
ifeq ($(HASXRD),yes)
ifeq ($(ARCH),win32)
MODULES      += proof/proofd
endif
MODULES      += proof/proofx
endif
ifeq ($(BUILDAFDSMGRD),yes)
MODULES      += proof/afdsmgrd
endif

-include MyModules.mk   # allow local modules

ifneq ($(findstring $(MAKECMDGOALS),distclean maintainer-clean),)
MODULES      += core/unix core/winnt graf2d/x11 graf2d/x11ttf \
                graf3d/gl graf3d/ftgl graf3d/glew io/rfio io/castor \
                montecarlo/pythia6 montecarlo/pythia8 misc/table \
                sql/mysql sql/pgsql sql/sapdb net/srputils graf3d/x3d \
                rootx net/rootd io/dcache io/chirp hist/hbook graf2d/asimage \
                net/ldap net/krb5auth net/rpdutils net/globusauth \
                bindings/pyroot bindings/ruby io/gfal misc/minicern \
                graf2d/qt gui/qtroot gui/qtgsi net/xrootd net/netx net/alien \
                proof/proofd proof/proofx proof/clarens proof/peac proof/pq2 \
                sql/oracle io/xmlparser math/mathmore cint/reflex cint/cintex \
                tmva math/genetic io/hdfs graf2d/fitsio roofit/roofitcore \
                roofit/roofit roofit/roostats roofit/histfactory \
                math/minuit2 net/monalisa math/fftw sql/odbc math/unuran \
                geom/gdml graf3d/eve net/glite misc/memstat \
                math/genvector net/bonjour graf3d/gviz3d graf2d/gviz \
                proof/proofbench proof/afdsmgrd
MODULES      := $(sort $(MODULES))   # removes duplicates
endif

MODULES      += main   # must be last, $(ALLLIBS) must be fully formed

ifeq ($(BUILDTOOLS),yes)
MODULES       = build cint/cint core/metautils core/clib core/base core/meta \
                core/utils
endif

##### ROOT libraries #####

LPATH         = lib

ifneq ($(PLATFORM),win32)
RPATH        := -L$(LPATH)
CINTLIBS     := -lCint
NEWLIBS      := -lNew
BOOTLIBS     := -lCore -lCint -lMathCore
ifneq ($(ROOTDICTTYPE),cint)
BOOTLIBS     += -lCintex -lReflex
endif
ROOTLIBS     := $(BOOTLIBS) -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad \
                -lTree -lMatrix -lThread
RINTLIBS     := -lRint
else
CINTLIBS     := $(LPATH)/libCint.lib
NEWLIBS      := $(LPATH)/libNew.lib
BOOTLIBS     := $(LPATH)/libCore.lib $(LPATH)/libCint.lib \
                $(LPATH)/libMathcore.lib
ifneq ($(ROOTDICTTYPE),cint)
BOOTLIBS     += $(LPATH)/libCintex.lib $(LPATH)/libReflex.lib
endif
ROOTLIBS     := $(BOOTLIBS) $(LPATH)/libRIO.lib $(LPATH)/libNet.lib \
                $(LPATH)/libHist.lib $(LPATH)/libGraf.lib \
                $(LPATH)/libGraf3d.lib $(LPATH)/libGpad.lib \
                $(LPATH)/libTree.lib $(LPATH)/libMatrix.lib \
                $(LPATH)/libThread.lib
RINTLIBS     := $(LPATH)/libRint.lib
endif

ROOTALIB     := $(LPATH)/libRoot.a
ROOTA        := bin/roota
PROOFSERVA   := bin/proofserva

# ROOTLIBSDEP is intended to match the content of ROOTLIBS
BOOTLIBSDEP   = $(ORDER_) $(CORELIB) $(CINTLIB) $(MATHCORELIB)
ifneq ($(ROOTDICTTYPE),cint)
BOOTLIBSDEP  += $(CINTEXLIB) $(REFLEXLIB)
endif
ROOTLIBSDEP   = $(BOOTLIBSDEP) $(IOLIB) $(NETLIB) $(HISTLIB) \
                $(GRAFLIB) $(G3DLIB) $(GPADLIB) $(TREELIB) $(MATRIXLIB)

# Force linking of not referenced libraries
ifeq ($(FORCELINK),yes)
ifeq ($(PLATFORM),aix5)
ROOTULIBS    := -Wl,-u,.G__cpp_setupG__Net      \
                -Wl,-u,.G__cpp_setupG__IO       \
                -Wl,-u,.G__cpp_setupG__Hist     \
                -Wl,-u,.G__cpp_setupG__Graf     \
                -Wl,-u,.G__cpp_setupG__G3D      \
                -Wl,-u,.G__cpp_setupG__GPad     \
                -Wl,-u,.G__cpp_setupG__Tree     \
                -Wl,-u,.G__cpp_setupG__Thread   \
                -Wl,-u,.G__cpp_setupG__Matrix
BOOTULIBS    := -Wl,-u,.G__cpp_setupG__MathCore
else
ROOTULIBS    := -Wl,-u,_G__cpp_setupG__Net      \
                -Wl,-u,_G__cpp_setupG__IO       \
                -Wl,-u,_G__cpp_setupG__Hist     \
                -Wl,-u,_G__cpp_setupG__Graf     \
                -Wl,-u,_G__cpp_setupG__G3D      \
                -Wl,-u,_G__cpp_setupG__GPad     \
                -Wl,-u,_G__cpp_setupG__Tree     \
                -Wl,-u,_G__cpp_setupG__Thread   \
                -Wl,-u,_G__cpp_setupG__Matrix
BOOTULIBS    := -Wl,-u,_G__cpp_setupG__MathCore
endif
endif
ifeq ($(PLATFORM),win32)
ROOTULIBS    := -include:_G__cpp_setupG__Net    \
                -include:_G__cpp_setupG__IO     \
                -include:_G__cpp_setupG__Hist   \
                -include:_G__cpp_setupG__Graf   \
                -include:_G__cpp_setupG__G3D    \
                -include:_G__cpp_setupG__GPad   \
                -include:_G__cpp_setupG__Tree   \
                -include:_G__cpp_setupG__Thread \
                -include:_G__cpp_setupG__Matrix
BOOTULIBS    := -include:_G__cpp_setupG__MathCore
endif

##### Compiler output option #####

CXXOUT ?= -o # keep whitespace after "-o"

##### clang or gcc version #####

ifneq ($(findstring clang,$(CXX)),)
CLANG_MAJOR  := $(shell $(CXX) -v 2>&1 | awk '{if (NR==1) print $$3}' | cut -d'.' -f1)
CLANG_MINOR  := $(shell $(CXX) -v 2>&1 | awk '{if (NR==1) print $$3}' | cut -d'.' -f2)
ifeq ($(CLANG_MAJOR),version)
   # Apple version of clang has different -v layout
   CLANG_MAJOR  := $(shell $(CXX) -v 2>&1 | awk '{if (NR==1) print $$4}' | cut -d'.' -f1)
   CLANG_MINOR  := $(shell $(CXX) -v 2>&1 | awk '{if (NR==1) print $$4}' | cut -d'.' -f2)
endif
else
ifneq ($(findstring gnu,$(COMPILER)),)
GCC_MAJOR     := $(shell $(CXX) -dumpversion 2>&1 | cut -d'.' -f1)
GCC_MINOR     := $(shell $(CXX) -dumpversion 2>&1 | cut -d'.' -f2)
GCC_PATCH     := $(shell $(CXX) -dumpversion 2>&1 | cut -d'.' -f3)
GCC_VERS      := gcc-$(GCC_MAJOR).$(GCC_MINOR)
GCC_VERS_FULL := gcc-$(GCC_MAJOR).$(GCC_MINOR).$(GCC_PATCH)
endif
endif

##### f77 options #####

ifneq ($(F77),)
F77LD        := $(F77)
endif
ifeq ($(F77OPT),)
F77OPT       := $(OPT)
endif
ifeq ($(F77LDFLAGS),)
F77LDFLAGS   := $(LDFLAGS)
endif

ifeq ($(GCC_MAJOR),3)
ifneq ($(GCC_MINOR),0)
ifeq ($(F77),g77)
LIBFRTBEGIN  := $(shell $(F77) -print-file-name=libfrtbegin.a)
F77LIBS      := $(LIBFRTBEGIN) $(F77LIBS)
endif
endif
endif
ifeq ($(GCC_MAJOR),4)
ifeq ($(F77),g77)
LIBFRTBEGIN  := $(shell $(F77) -print-file-name=libfrtbegin.a)
F77LIBS      := $(LIBFRTBEGIN) $(F77LIBS)
endif
endif

##### Store SVN revision number #####

ifeq ($(findstring $(MAKECMDGOALS),clean distclean maintainer-clean dist),)
ifeq ($(findstring clean-,$(MAKECMDGOALS)),)
ifeq ($(shell which svn 2>&1 | sed -ne "s@.*/svn@svn@p"),svn)
SVNREV  := $(shell bash $(ROOT_SRCDIR)/build/unix/svninfo.sh $(ROOT_SRCDIR))
endif
endif
endif

##### Utilities #####

ifneq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
   MAKEDIR      = +@[ -d $(dir $@) ] || mkdir -p $(dir $@)
   RUNTIMEDIRS := etc macros icons fonts README tutorials test man
   POSTBIN     += runtimedirs
endif

ifneq ($(HOST),)
   BUILDTOOLSDIR := buildtools
endif

ifeq ($(PLATFORM),ios)
   POSTBIN       += staticlib
endif

MAKEDEP        = $(RMKDEP)
MAKELIB        = $(ROOT_SRCDIR)/build/unix/makelib.sh $(MKLIBOPTIONS)
MAKEDIST      := $(ROOT_SRCDIR)/build/unix/makedist.sh
MAKEDISTSRC   := $(ROOT_SRCDIR)/build/unix/makedistsrc.sh
MAKEVERSION   := $(ROOT_SRCDIR)/build/unix/makeversion.sh
MAKECOMPDATA  := $(ROOT_SRCDIR)/build/unix/compiledata.sh
MAKECINTDLL   := $(ROOT_SRCDIR)/build/unix/makecintdll.sh
MAKECHANGELOG := $(ROOT_SRCDIR)/build/unix/makechangelog.sh
MAKEHTML      := $(ROOT_SRCDIR)/build/unix/makehtml.sh
MAKELOGHTML   := $(ROOT_SRCDIR)/build/unix/makeloghtml.sh
MAKEPLUGINS   := $(ROOT_SRCDIR)/build/unix/makeplugins-ios.sh
MAKERELNOTES  := $(ROOT_SRCDIR)/build/unix/makereleasenotes.sh
STATICOBJLIST := $(ROOT_SRCDIR)/build/unix/staticobjectlist.sh
MAKESTATICLIB := $(ROOT_SRCDIR)/build/unix/makestaticlib.sh
MAKESTATIC    := $(ROOT_SRCDIR)/build/unix/makestatic.sh
RECONFIGURE   := $(ROOT_SRCDIR)/build/unix/reconfigure.sh
ifeq ($(PLATFORM),win32)
MAKELIB       := $(ROOT_SRCDIR)/build/win/makelib.sh
MAKECOMPDATA  := $(ROOT_SRCDIR)/build/win/compiledata.sh
endif

##### Compiler directives and run-control file #####

COMPILEDATA   = include/compiledata.h
ROOTRC        = etc/system.rootrc
ROOTMAP       = etc/system.rootmap

##### Extra libs needed for "static" target #####

STATICEXTRALIBS = $(PCRELDFLAGS) $(PCRELIB) \
                  $(FREETYPELDFLAGS) $(FREETYPELIB)

##### libCore #####

COREL         = $(BASEL1) $(BASEL2) $(BASEL3) $(CONTL) $(METAL) $(ZIPL) \
                $(SYSTEML) $(CLIBL) $(METAUTILSL) $(TEXTINPUTL)
COREO         = $(BASEO) $(CONTO) $(METAO) $(SYSTEMO) $(ZIPO) $(LZMAO) \
                $(CLIBO) $(METAUTILSO) $(TEXTINPUTO) $(CLINGO)
COREDO        = $(BASEDO) $(CONTDO) $(METADO) $(METACDO) $(SYSTEMDO) $(ZIPDO) \
                $(CLIBDO) $(METAUTILSDO) $(TEXTINPUTDO) $(CLINGDO)

CORELIB      := $(LPATH)/libCore.$(SOEXT)
COREMAP      := $(CORELIB:.$(SOEXT)=.rootmap)

ifneq ($(BUILTINZLIB),yes)
CORELIBEXTRA    += $(ZLIBLIBDIR) $(ZLIBCLILIB)
STATICEXTRALIBS += $(ZLIBLIBDIR) $(ZLIBCLILIB)
endif

ifneq ($(BUILTINLZMA),yes)
CORELIBEXTRA    += $(LZMALIBDIR) $(LZMACLILIB)
STATICEXTRALIBS += $(LZMALIBDIR) $(LZMACLILIB)
else
CORELIBEXTRA    += $(LZMALIB)
STATICEXTRALIBS += $(LZMALIB)
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

##### all #####

ALLHDRS      :=
ALLLIBS      := $(CORELIB)
ALLMAPS      := $(COREMAP)
ALLEXECS     :=
INCLUDEFILES :=

##### RULES #####

.SUFFIXES: .cxx .d
.PRECIOUS: include/%.h

# special rules (need to be defined before generic ones)
cint/cint/lib/dll_stl/G__%.o: cint/cint/lib/dll_stl/G__%.cxx
	$(MAKEDEP) -R -f$(patsubst %.o,%.d,$@) -Y -w 1000 -- \
	   $(CXXFLAGS) -D__cplusplus -I$(CINTDIRL)/prec_stl \
	   -I$(CINTDIRSTL) -I$(CINTDIR)/inc -- $<
	$(CXX) $(NOOPT) $(CXXFLAGS) -I. -I$(CINTDIR)/inc  $(CXXOUT)$@ -c $<

cint/cint/lib/dll_stl/G__c_%.o: cint/cint/lib/dll_stl/G__c_%.c
	$(MAKEDEP) -R -f$(patsubst %.o,%.d,$@) -Y -w 1000 -- \
	   $(CFLAGS) -I$(CINTDIRL)/prec_stl \
	   -I$(CINTDIRSTL) -I$(CINTDIR)/inc -- $<
	$(CC) $(NOOPT) $(CFLAGS) -I. -I$(CINTDIR)/inc  $(CXXOUT)$@ -c $<

cint/cint/lib/G__%.o: cint/cint/lib/G__%.cxx
	$(MAKEDEP) -R -f$(patsubst %.o,%.d,$@) -Y -w 1000 -- \
	   $(CXXFLAGS) -D__cplusplus -I$(CINTDIRL)/prec_stl \
	   -I$(CINTDIRSTL) -I$(CINTDIR)/inc -- $<
	$(CXX) $(NOOPT) $(CXXFLAGS) -I. -I$(CINTDIR)/inc  $(CXXOUT)$@ -c $<

cint/cint/lib/G__c_%.o: cint/cint/lib/G__c_%.c
	$(MAKEDEP) -R -f$(patsubst %.o,%.d,$@) -Y -w 1000 -- \
	   $(CFLAGS) -I$(CINTDIRL)/prec_stl \
	   -I$(CINTDIRSTL) -I$(CINTDIR)/inc -- $<
	$(CC) $(NOOPT) $(CFLAGS) -I. -I$(CINTDIR)/inc  $(CXXOUT)$@ -c $<

G__%.o: G__%.cxx
	$(MAKEDEP) -R -f$(patsubst %.o,%.d,$@) -Y -w 1000 -- \
	   $(CXXFLAGS) -D__cplusplus -I$(CINTDIRL)/prec_stl \
	   -I$(CINTDIRSTL) -I$(CINTDIR)/inc -- $<
	$(CXX) $(NOOPT) $(CXXFLAGS) -I. -I$(CINTDIR)/inc  $(CXXOUT)$@ -c $<

G__c_%.o: G__c_%.c
	$(MAKEDEP) -R -f$(patsubst %.o,%.d,$@) -Y -w 1000 -- \
	   $(CFLAGS) -I$(CINTDIRL)/prec_stl \
	   -I$(CINTDIRSTL) -I$(CINTDIR)/inc -- $<
	$(CC) $(NOOPT) $(CFLAGS) -I. -I$(CINTDIR)/inc  $(CXXOUT)$@ -c $<

cint/cint/%.o: cint/cint/%.cxx
	$(MAKEDEP) -R -fcint/cint/$*.d -Y -w 1000 -- $(CINTCXXFLAGS) -I. -D__cplusplus -- $<
	$(CXX) $(OPT) $(CINTCXXFLAGS) -I. $(CXXOUT)$@ -c $<

cint/cint/%.o: $(ROOT_SRCDIR)/cint/cint/%.cxx
	$(MAKEDIR)
	$(MAKEDEP) -R -fcint/cint/$*.d -Y -w 1000 -- $(CINTCXXFLAGS) -I. -D__cplusplus -- $<
	$(CXX) $(OPT) $(CINTCXXFLAGS) -I. $(CXXOUT)$@ -c $<

cint/cint/%.o: $(ROOT_SRCDIR)/cint/cint/%.c
	$(MAKEDIR)
	$(MAKEDEP) -R -fcint/cint/$*.d -Y -w 1000 -- $(CINTCFLAGS) -I. -- $<
	$(CC) $(OPT) $(CINTCFLAGS) -I. $(CXXOUT)$@ -c $<

build/rmkdepend/%.o: $(ROOT_SRCDIR)/build/rmkdepend/%.cxx
	$(MAKEDIR)
	$(CXX) $(OPT) $(CXXFLAGS) $(CXXOUT)$@ -c $<

build/rmkdepend/%.o: $(ROOT_SRCDIR)/build/rmkdepend/%.c
	$(MAKEDIR)
	$(CC) $(OPT) $(CFLAGS) $(CXXOUT)$@ -c $<

define SRCTOOBJ_template
$(1)/%_tmp.o: $(1)/%_tmp.cxx
	$$(MAKEDIR)
	$$(MAKEDEP) -R -f$$(@:.o=.d) -Y -w 1000 -- $$(CXXFLAGS) -D__cplusplus -- $$<
	$$(CXX) $$(OPT) $$(CXXFLAGS) $$(CXXOUT)$$@ -c $$<

$(1)/%.o: $(ROOT_SRCDIR)/$(1)/%.cxx
	$$(MAKEDIR)
	$$(MAKEDEP) -R -f$$(@:.o=.d) -Y -w 1000 -- $$(CXXFLAGS) -D__cplusplus -- $$<
	$$(CXX) $$(OPT) $$(CXXFLAGS) $$(CXXOUT)$$@ -c $$<

$(1)/%.o: $(ROOT_SRCDIR)/$(1)/%.c
	$$(MAKEDIR)
	$$(MAKEDEP) -R -f$$(@:.o=.d) -Y -w 1000 -- $$(CFLAGS) -- $$<
	$$(CC) $$(OPT) $$(CFLAGS) $$(CXXOUT)$$@ -c $$<

$(1)/%.o: $(ROOT_SRCDIR)/$(1)/%.f
	$$(MAKEDIR)
ifeq ($$(F77),f2c)
	f2c -a -A $$<
	$$(CC) $$(F77OPT) $$(CFLAGS) $$(CXXOUT)$$@ -c $$(@:.o=.c)
else
	$$(F77) $$(F77OPT) $$(F77FLAGS) $$(CXXOUT)$$@ -c $$<
endif
endef

MODULESGENERIC := build $(filter-out build,$(MODULES))
MODULESGENERIC := $(filter-out cint/cint,$(MODULES))
$(foreach module,$(MODULESGENERIC),$(eval $(call SRCTOOBJ_template,$(module))))

%.o: %.cxx
	$(MAKEDEP) -R -f$*.d -Y -w 1000 -- $(CXXFLAGS) -D__cplusplus -- $<
	$(CXX) $(OPT) $(CXXFLAGS) $(CXXOUT)$@ -c $<

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
                clean distclean maintainer-clean compiledata \
                version html changelog install uninstall showbuild \
                releasenotes staticlib static map debian redhat skip postbin \
                showit help runtimedirs plugins-ios

ifneq ($(findstring map, $(MAKECMDGOALS)),)
.NOTPARALLEL:
endif

all:            rootexecs postbin

fast:           rootexecs

skip:
		@true;

-include $(patsubst %,$(ROOT_SRCDIR)/%/ModuleVars.mk,$(MODULES))
include $(patsubst %,$(ROOT_SRCDIR)/%/Module.mk,$(MODULES))

-include MyRules.mk            # allow local rules

ifeq ($(findstring $(MAKECMDGOALS),clean distclean maintainer-clean dist \
      distsrc version showbuild \
      changelog debian redhat),)
ifeq ($(findstring clean-,$(MAKECMDGOALS)),)
ifeq ($(findstring skip,$(MAKECMDGOALS))$(findstring fast,$(MAKECMDGOALS)),)
-include $(INCLUDEFILES)
endif
-include build/dummy.d          # must be last include
endif
endif

rootcint:       all-cint all-utils

rootlibs:       rootcint compiledata $(ALLLIBS) $(ALLMAPS)

rootexecs:      rootlibs $(ALLEXECS)

ifneq ($(HOST),)
.PHONY:         buildtools

buildtools:
		@if [ ! -f $(BUILDTOOLSDIR)/Makefile ]; then \
		   echo "*** Building build tools in $(BUILDTOOLSDIR)..."; \
		   mkdir -p $(BUILDTOOLSDIR); \
		   cd $(BUILDTOOLSDIR); \
		   $(ROOT_SRCDIR)/configure $(HOST) --minimal; \
		else \
		   echo "*** Running make in $(BUILDTOOLSDIR)..."; \
		   cd $(BUILDTOOLSDIR); \
		fi; \
		($(MAKE) BUILDTOOLS=yes \
		   TARGETFLAGS=-DR__$(shell echo $(ARCH) | tr 'a-z' 'A-Z') \
		   rootcint cint/cint/lib/posix/mktypes \
		) || exit 1;

distclean::
		@rm -rf $(BUILDTOOLSDIR)
endif

postbin:        $(POSTBIN)

compiledata:    $(COMPILEDATA)

config config/Makefile.:
ifeq ($(BUILDING_WITHIN_IDE),)
	@(if [ ! -f config/Makefile.config ] || \
	     [ ! -f config/Makefile.comp ]; then \
	   echo ""; echo "Please, run ./configure first"; echo ""; \
	   exit 1; \
	fi)
else
# Building from within an IDE, running configure
	@(if [ ! -f config/Makefile.config ] || \
	     [ ! -f config/Makefile.comp ]; then \
	   ./configure --build=debug `cat config.status 2>/dev/null`; \
	fi)
endif

# Target Makefile is synonym for "run (re-)configure"
# Makefile is target as we need to re-parse dependencies after
# configure is run (as RConfigure.h changed etc)
config/Makefile.config config/Makefile.comp include/RConfigure.h \
  include/RConfigOptions.h etc/system.rootauthrc etc/system.rootdaemonrc \
  etc/root.mimes $(ROOTRC) \
  bin/root-config: Makefile

ifeq ($(findstring $(MAKECMDGOALS),distclean maintainer-clean debian redhat),)
Makefile: $(addprefix $(ROOT_SRCDIR)/,configure config/rootrc.in \
  config/RConfigure.in config/Makefile.in config/Makefile.$(ARCH) \
  config/Makefile-comp.in config/root-config.in config/rootauthrc.in \
  config/rootdaemonrc.in config/mimes.unix.in config/mimes.win32.in \
  config/proofserv.in config/roots.in) config.status
	+@( $(RECONFIGURE) "$?" "$(ROOT_SRCDIR)" || ( \
	   echo ""; echo "Please, run $(ROOT_SRCDIR)/configure again as config option files ($?) have changed."; \
	   echo ""; exit 1; \
	 ) )
endif

$(COMPILEDATA): $(ROOT_SRCDIR)/config/Makefile.$(ARCH) config/Makefile.comp \
                $(MAKECOMPDATA)
	@$(MAKECOMPDATA) $(COMPILEDATA) "$(CXX)" "$(OPTFLAGS)" "$(DEBUGFLAGS)" \
	   "$(CXXFLAGS)" "$(SOFLAGS)" "$(LDFLAGS)" "$(SOEXT)" "$(SYSLIBS)" \
	   "$(LIBDIR)" "$(BOOTLIBS)" "$(RINTLIBS)" "$(INCDIR)" \
	   "$(MAKESHAREDLIB)" "$(MAKEEXE)" "$(ARCH)" "$(ROOTBUILD)" \
	   "$(EXPLICITLINK)"

ifeq ($(HOST),)
build/dummy.d: config Makefile $(ALLHDRS) $(RMKDEP) $(BINDEXP)
else
build/dummy.d: config Makefile buildtools $(ALLHDRS) $(RMKDEP) $(BINDEXP)
endif
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

$(COREMAP): $(RLIBMAP) $(MAKEFILEDEP) $(COREL)
	$(RLIBMAP) -o $@ -l $(CORELIB) -d $(CORELIBDEPM) -c $(COREL)

map::   $(ALLMAPS)

dist:
	@$(MAKEDIST) $(GCC_VERS)

distsrc:
	@$(MAKEDISTSRC)

distmsi: build/package/msi/makemsi$(EXEEXT)
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
	$(ROOT_SRCDIR)/build/package/lib/makedebdir.sh
	fakeroot debian/rules debian/control
	dpkg-buildpackage -rfakeroot -us -uc -i"G__|^debian|root-bin.png|\.d$$"
	@echo "Debian GNU/Linux packages done. They are put in '../'"

redhat:
	@if [ ! -x `which rpm` ]; then \
	   echo "You must have rpm installed to make the Redhat package"; \
	   exit 1; fi
	@echo "OK, you have RPM on your system - good"
	$(ROOT_SRCDIR)/build/package/lib/makerpmspec.sh
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

redhat-tar:
	@if [ ! -x `which rpm` ]; then \
	   echo "You must have rpm installed to make the Redhat package"; \
	   exit 1; fi
	@echo "OK, you have RPM on your system - good"
	$(ROOT_SRCDIR)/build/package/lib/makerpmspec.sh
	-@$(MAKE) distclean
	-@$(MAKE) maintainer-clean
	@vers=`sed 's|\(.*\)/\(.*\)|\1.\2|' < build/version_number` && \
	  rm -f root_v$$vers.source.tar.gz && \
	  (cd ../ && tar 		\
		--exclude=\\.svn 	\
		--exclude=root/debian 	\
		--exclude=root/bin	\
		--exclude=root/lib	\
		--exclude=root/include	\
	     -czf root_v$$vers.source.tar.gz root) && \
	  mv ../root_v$$vers.source.tar.gz .
	@echo "To build the packages, run "
	@echo ""
	@vers=`sed 's|\(.*\)/\(.*\)|\1.\2|' < build/version_number` ; \
	  echo "  rpmbuild -ta root_v$$vers.source.tar.gz"
	@echo ""
	@echo "as user root (or similar). If you want to build outside"
	@echo "the regular tree (as a normal user), please refer to the"
	@echo "RPM documentation."

redhat-rpm: redhat-tar
	@rm -rf rpm
	@mkdir -p rpm/SOURCES rpm/SPECS rpm/BUILD rpm/RPMS rpm/SRPMS
	@vers=`sed 's|\(.*\)/\(.*\)|\1.\2|' < build/version_number` && \
	  rpmbuild --define "_topdir `pwd`/rpm" -ta root_v$$vers.source.tar.gz
	@rm -rf rpm/SOURCES rpm/SPECS
	@echo "Packages build in rpm/RPMS and rpm/SPRMS"

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
	$(ROOT_SRCDIR)/build/package/lib/makerpmspecs.sh rpm \
		$(ROOT_SRCDIR)/build/package/common \
	        $(ROOT_SRCDIR)/build/package/rpm root-rootd >> \
		root-rootd.spec.tmp
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
	@rm -f __compiledata *~ core.*

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
	-@mv -f include/RConfigOptions.h include/RConfigOptions.h-
	@rm -f include/*.h $(ROOTMAP) $(CORELIB) $(COREMAP)
	-@mv -f include/RConfigure.h- include/RConfigure.h
	-@mv -f include/RConfigOptions.h- include/RConfigOptions.h
	@rm -f bin/*.dll bin/*.exp bin/*.lib bin/*.pdb \
               lib/*.def lib/*.exp lib/*.lib lib/*.dll.a \
               *.def .def
ifeq ($(PLATFORM),macosx)
	@rm -f lib/*.dylib
	@rm -f lib/*.so
	@(find . -name "*.dSYM" -exec rm -rf {} \; >/dev/null 2>&1;true)
endif
	-@(mv -f tutorials/gallery.root tutorials/gallery.root- >/dev/null 2>&1;true)
	-@(mv -f tutorials/mlp/mlpHiggs.root tutorials/mlp/mlpHiggs.root- >/dev/null 2>&1;true)
	-@(mv -f tutorials/quadp/stock.root tutorials/quadp/stock.root- >/dev/null 2>&1;true)
	@(find tutorials -name "files" -exec rm -rf {} \; >/dev/null 2>&1;true)
	@(find tutorials -name "*.root" -exec rm -rf {} \; >/dev/null 2>&1;true)
	@(find tutorials -name "*.ps" -exec rm -rf {} \; >/dev/null 2>&1;true)
	@(find tutorials -path '*/doc' -prune -o -name "*.gif" -exec rm -rf {} \; >/dev/null 2>&1;true)
	@(find tutorials -name "so_locations" -exec rm -rf {} \; >/dev/null 2>&1;true)
	@(find tutorials -name "pca.C" -exec rm -rf {} \; >/dev/null 2>&1;true)
	@(find tutorials -name "*.so" -exec rm -rf {} \; >/dev/null 2>&1;true)
	@(find tutorials -name "work.pc" -exec rm -rf {} \; >/dev/null 2>&1;true)
	@(find tutorials -name "work.pcl" -exec rm -rf {} \; >/dev/null 2>&1;true)
	@rm -rf tutorials/eve/aliesd
	-@(mv -f tutorials/gallery.root- tutorials/gallery.root >/dev/null 2>&1;true)
	-@(mv -f tutorials/mlp/mlpHiggs.root- tutorials/mlp/mlpHiggs.root >/dev/null 2>&1;true)
	-@(mv -f tutorials/quadp/stock.root- tutorials/quadp/stock.root >/dev/null 2>&1;true)
	@rm -f $(ROOTA) $(PROOFSERVA) $(ROOTALIB)
	@rm -f $(CINTDIR)/include/*.dll $(CINTDIR)/include/*.so*
	@rm -f $(CINTDIR)/stl/*.dll $(CINTDIR)/stl/*.so*
	@rm -f $(CINTDIR)/include/sys/*.dll $(CINTDIR)/include/sys/*.so.*
	@rm -f $(CINTDIR)/lib/posix/a.out $(CINTDIR)/lib/posix/mktypes
	@rm -f README/ChangeLog build/dummy.d
	@rm -rf README/ReleaseNotes
	@rm -f etc/svninfo.txt
	@(find . -path '*/daemons' -prune -o -name *.d -exec rm -rf {} \; >/dev/null 2>&1;true)
	@(find . -name *.o -exec rm -rf {} \; >/dev/null 2>&1;true)
	-@([ -d test ] && (cd test && $(MAKE) distclean); true)

maintainer-clean:: distclean
	@rm -rf bin lib include htmldoc system.rootrc config/Makefile.config \
	   config/Makefile.comp $(ROOTRC) etc/system.rootauthrc \
	   etc/system.rootdaemonrc etc/root.mimes etc/daemons/rootd.rc.d \
	   etc/daemons/rootd.xinetd etc/daemons/proofd.rc.d \
	   etc/daemons/proofd.xinetd main/src/proofserv.sh main/src/roots.sh \
	   etc/daemons/olbd.rc.d etc/daemons/xrootd.rc.d \
	   etc/daemons/cmsd.rc.d macros/html.C \
	   build/misc/root-help.el build-arch-stamp build-indep-stamp \
	   configure-stamp build-arch-cint-stamp config.status config.log

version: $(CINTTMP)
	@$(MAKEVERSION)

staticlib: $(ROOTALIB)

static: $(ROOTA)

$(ROOTA) $(PROOFSERVA): $(ROOTALIB) $(MAKESTATIC) $(STATICOBJLIST)
	@$(MAKESTATIC) $(PLATFORM) "$(CXX)" "$(CC)" "$(LD)" "$(LDFLAGS)" \
	   "$(XLIBS)" "$(SYSLIBS)" "$(STATICEXTRALIBS)" $(STATICOBJLIST)

$(ROOTALIB): $(ALLLIBS) $(MAKESTATICLIB) $(STATICOBJLIST)
	@$(MAKESTATICLIB) $(STATICOBJLIST)

plugins-ios: $(ROOTEXE)
	@$(MAKEPLUGINS)

changelog:
	@$(MAKECHANGELOG)

releasenotes:
	@$(MAKERELNOTES)

html: $(ROOTEXE) changelog releasenotes
	@$(MAKELOGHTML)
	@$(MAKEHTML)

# Use DESTDIR to set a sandbox prior to calling "make install", e.g.:
#   ./configure --prefix=/usr/
#   make
#   DESTDIR=/tmp/root_install/ make install
#   cd /tmp/root_install
#   tar czf ~/root-vxxxx.tar.gz usr
# Needed to create e.g. rpms.
install: all
	@if [ -d $(DESTDIR)$(BINDIR) ]; then \
	   inode1=`ls -id $(DESTDIR)$(BINDIR) | awk '{ print $$1 }'`; \
	fi; \
	inode2=`ls -id $$PWD/bin | awk '{ print $$1 }'`; \
	if ([ -d $(DESTDIR)$(BINDIR) ] && [ "x$$inode1" = "x$$inode2" ]); then \
	   echo "Everything already installed..."; \
	else \
           if [ "$(USECONFIG)" = "FALSE" ] && [ -z "$(ROOTSYS)" ]; then \
              echo "ROOTSYS not set, set it to a destination directory"; \
              exit 1; \
           fi; \
	   echo "Installing binaries in $(DESTDIR)$(BINDIR)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(BINDIR); \
	   $(INSTALLDATA) bin/*                 $(DESTDIR)$(BINDIR); \
	   echo "Installing libraries in $(DESTDIR)$(LIBDIR)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(LIBDIR); \
	   $(INSTALLDATA) lib/*                 $(DESTDIR)$(LIBDIR); \
	   if [ x"$(ARCH)" = x"win32gcc" ]; then \
	      $(INSTALLDATA) bin/*.dll             $(DESTDIR)$(BINDIR); \
	      for f in $(DESTDIR)$(LIBDIR)/*.dll; do \
	         bindll=`basename $$f | sed 's,\..*$$,,'`; \
	         bindll=`ls $(DESTDIR)$(BINDIR)/$${bindll}.*dll`; \
	         ln -sf $${bindll} $$f; \
	      done; \
           elif [ x"$(PLATFORM)" = x"win32" ]; then \
	      $(INSTALLDATA) $(GDKDLL)             $(DESTDIR)$(BINDIR); \
	      $(INSTALLDATA) $(GDKDLLS)            $(DESTDIR)$(BINDIR); \
	   fi; \
	   echo "Installing headers in $(DESTDIR)$(INCDIR)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(INCDIR); \
	   $(INSTALLDATA) include/*             $(DESTDIR)$(INCDIR); \
	   echo "Installing $(ROOT_SRCDIR)/main/src/rmain.cxx in $(DESTDIR)$(INCDIR)"; \
	   $(INSTALLDATA) $(ROOT_SRCDIR)/main/src/rmain.cxx $(DESTDIR)$(INCDIR); \
	   echo "Installing cint/cint/include cint/cint/lib and cint/cint/stl in $(DESTDIR)$(CINTINCDIR)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(CINTINCDIR)/cint; \
	   $(INSTALLDATA) cint/cint/include     $(DESTDIR)$(CINTINCDIR)/cint; \
	   $(INSTALLDATA) cint/cint/lib         $(DESTDIR)$(CINTINCDIR)/cint; \
	   $(INSTALLDATA) cint/cint/stl         $(DESTDIR)$(CINTINCDIR)/cint; \
	   find $(DESTDIR)$(CINTINCDIR) -name CVS -exec rm -rf {} \; >/dev/null 2>&1; \
	   find $(DESTDIR)$(CINTINCDIR) -name .svn -exec rm -rf {} \; >/dev/null 2>&1; \
	   echo "Installing icons in $(DESTDIR)$(ICONPATH)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(ICONPATH); \
	   $(INSTALLDATA) icons/*.xpm           $(DESTDIR)$(ICONPATH); \
	   $(INSTALLDATA) icons/*.png           $(DESTDIR)$(ICONPATH); \
	   $(INSTALLDATA) icons/*.gif           $(DESTDIR)$(ICONPATH); \
	   echo "Installing fonts in $(DESTDIR)$(TTFFONTDIR)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(TTFFONTDIR); \
	   $(INSTALLDATA) fonts/*               $(DESTDIR)$(TTFFONTDIR); \
	   find $(DESTDIR)$(TTFFONTDIR) -name CVS -exec rm -rf {} \; >/dev/null 2>&1; \
	   find $(DESTDIR)$(TTFFONTDIR) -name .svn -exec rm -rf {} \; >/dev/null 2>&1; \
	   echo "Installing misc docs in $(DESTDIR)$(DOCDIR)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(DOCDIR); \
	   $(INSTALLDATA) LICENSE               $(DESTDIR)$(DOCDIR); \
	   $(INSTALLDATA) README/*              $(DESTDIR)$(DOCDIR); \
	   find $(DESTDIR)$(DOCDIR) -name CVS -exec rm -rf {} \; >/dev/null 2>&1; \
	   find $(DESTDIR)$(DOCDIR) -name .svn -exec rm -rf {} \; >/dev/null 2>&1; \
	   echo "Installing tutorials in $(DESTDIR)$(TUTDIR)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(TUTDIR); \
	   $(INSTALLDATA) tutorials/*           $(DESTDIR)$(TUTDIR); \
	   find $(DESTDIR)$(TUTDIR) -name CVS -exec rm -rf {} \; >/dev/null 2>&1; \
	   find $(DESTDIR)$(TUTDIR) -name .svn -exec rm -rf {} \; >/dev/null 2>&1; \
	   echo "Installing tests in $(DESTDIR)$(TESTDIR)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(TESTDIR); \
	   $(INSTALLDATA) test/*                $(DESTDIR)$(TESTDIR); \
	   find $(DESTDIR)$(TESTDIR) -name CVS -exec rm -rf {} \; >/dev/null 2>&1; \
	   find $(DESTDIR)$(TESTDIR) -name .svn -exec rm -rf {} \; >/dev/null 2>&1; \
	   echo "Installing macros in $(DESTDIR)$(MACRODIR)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(MACRODIR); \
	   $(INSTALLDATA) macros/*              $(DESTDIR)$(MACRODIR); \
	   find $(DESTDIR)$(MACRODIR) -name CVS -exec rm -rf {} \; >/dev/null 2>&1; \
	   find $(DESTDIR)$(MACRODIR) -name .svn -exec rm -rf {} \; >/dev/null 2>&1; \
	   echo "Installing man(1) pages in $(DESTDIR)$(MANDIR)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(MANDIR); \
	   $(INSTALLDATA) man/man1/*            $(DESTDIR)$(MANDIR); \
	   find $(DESTDIR)$(MANDIR) -name CVS -exec rm -rf {} \; >/dev/null 2>&1; \
	   find $(DESTDIR)$(MANDIR) -name .svn -exec rm -rf {} \; >/dev/null 2>&1; \
	   echo "Installing config files in $(DESTDIR)$(ETCDIR)"; \
	   rm -f                                $(DESTDIR)$(ETCDIR)/system.rootmap; \
	   $(INSTALLDIR)                        $(DESTDIR)$(ETCDIR); \
	   $(INSTALLDATA) etc/*                 $(DESTDIR)$(ETCDIR); \
	   find $(DESTDIR)$(ETCDIR) -name CVS -exec rm -rf {} \; >/dev/null 2>&1; \
	   find $(DESTDIR)$(ETCDIR) -name .svn -exec rm -rf {} \; >/dev/null 2>&1; \
	   echo "Installing Autoconf macro in $(DESTDIR)$(ACLOCALDIR)"; \
	   $(INSTALLDIR)                        $(DESTDIR)$(ACLOCALDIR); \
	   $(INSTALLDATA) build/misc/root.m4    $(DESTDIR)$(ACLOCALDIR); \
	   echo "Installing Emacs Lisp library in $(DESTDIR)$(ELISPDIR)"; \
	   $(INSTALLDIR)                          $(DESTDIR)$(ELISPDIR); \
	   $(INSTALLDATA) build/misc/root-help.el $(DESTDIR)$(ELISPDIR); \
	   echo "Installing GDML conversion scripts in $(DESTDIR)$(LIBDIR)"; \
	   $(INSTALLDATA) geom/gdml/*.py          $(DESTDIR)$(LIBDIR); \
	   find $(DESTDIR)$(DATADIR) -name CVS -exec rm -rf {} \; >/dev/null 2>&1; \
	   find $(DESTDIR)$(DATADIR) -name .svn -exec rm -rf {} \; >/dev/null 2>&1; \
	fi

uninstall:
	@if [ -d $(DESTDIR)$(BINDIR) ]; then \
	   inode1=`ls -id $(DESTDIR)$(BINDIR) | awk '{ print $$1 }'`; \
	fi; \
	inode2=`ls -id $$PWD/bin | awk '{ print $$1 }'`; \
	if [ -d $(DESTDIR)$(BINDIR) ] && [ "x$$inode1" = "x$$inode2" ]; then \
	   $(MAKE) distclean ; \
	elif [ "$(USECONFIG)" = "FALSE" ]; then \
	   echo "To uninstall ROOT just delete directory $$PWD"; \
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
	   fi; \
	   for lib in $(ALLLIBS) $(CINTLIB) $(ALLMAPS); do \
	      rm -f $(DESTDIR)$(LIBDIR)/`basename $$lib`* ; \
	   done; \
	   if test "x$(RFLX_GRFLXPY)" != "x"; then \
	      rm -f $(DESTDIR)$(LIBDIR)/$(RFLX_GRFLXPY); \
	   fi; \
	   if test "x$(RFLX_GRFLXPYC)" != "x"; then \
	      rm -f $(DESTDIR)$(LIBDIR)/$(RFLX_GRFLXPYC); \
	   fi; \
	   if test "x$(RFLX_GRFLXPY)$(RFLX_GRFLXPYC)" != "x"; then \
	      dir=$(RFLX_GRFLXDD:lib/=); \
	      while test "x$${dir}" != "x" && \
	         test -d $(DESTDIR)$(LIBDIR)/$${dir} && \
	         test "x`ls $(DESTDIR)$(INCDIR)/$${dir}`" = "x"; do \
	         rm -rf $(DESTDIR)$(INCDIR)/$${dir}; \
	         dir=$(dirname $${dir}); \
	      done; \
	   fi; \
	   if test -d $(DESTDIR)$(LIBDIR) && \
	      test "x`ls $(DESTDIR)$(LIBDIR)`" = "x"; then \
	      rm -rf $(DESTDIR)$(LIBDIR); \
	   fi ; \
	   for subdir in \
              . \
              Math/GenVector Math \
              Reflex/internal Reflex/Builder Reflex \
              Cintex GL TMVA Minuit2; do \
              if test -d include/$${subdir}; then \
	         for i in include/$${subdir}/*.h ; do \
	            rm -f $(DESTDIR)$(INCDIR)/$${subdir}/`basename $$i`; \
	         done ; \
	         if test -d $(DESTDIR)$(INCDIR)/$${subdir} && \
	            test "x`ls $(DESTDIR)$(INCDIR)/$${subdir}`" = "x" ; then \
	            rm -rf $(DESTDIR)$(INCDIR)/$${subdir}; \
	         fi; \
              fi; \
	   done; \
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

ifneq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
# install directrories needed at run-time
runtimedirs:
	@echo "Rsync'ing $(ROOT_SRCDIR)/etc..."; \
	$(RSYNC) \
		--exclude '.svn' \
		--exclude root.mimes \
		--exclude system.rootauthrc \
		--exclude system.rootdaemonrc \
		--exclude system.rootrc \
		--exclude cmsd.rc.d \
		--exclude olbd.rc.d \
		--exclude proofd.rc.d \
		--exclude proofd.xinetd \
		--exclude rootd.rc.d \
		--exclude rootd.xinetd \
		--exclude xrootd.rc.d \
		--exclude svninfo.txt \
		$(ROOT_SRCDIR)/etc . ; \
	echo "Rsync'ing $(ROOT_SRCDIR)/macros..."; \
	$(RSYNC) \
		--exclude '.svn' \
		--exclude html.C \
		$(ROOT_SRCDIR)/macros . ; \
	for d in icons fonts README tutorials test man; do \
		echo "Rsync'ing $(ROOT_SRCDIR)/$$d..."; \
		$(RSYNC) \
			--exclude '.svn' \
			--exclude '*.o' \
			--exclude '*.so' \
			--exclude '*.lib' \
			--exclude '*.dll' \
			$(ROOT_SRCDIR)/$$d . ; \
	done;
endif

showbuild:
	@echo "ROOTSYS            = $(ROOTSYS)"
	@echo "SVNREV             = $(SVNREV)"
	@echo "PLATFORM           = $(PLATFORM)"
	@echo "OPT                = $(OPT)"
	@echo ""
	@echo "ROOT_SRCDIR        = $(ROOT_SRCDIR)"
	@echo "ROOT_OBJDIR        = $(ROOT_OBJDIR)"
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
	@echo "GVIZLIBDIR         = $(GVIZLIBDIR)"
	@echo "GVIZLIB            = $(GVIZLIB)"
	@echo "GVIZINCDIR         = $(GVIZINCDIR)"
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
	@echo ""
	@echo "The list of modules to be built:"
	@echo "--------------------------------"
	@echo "$(MODULES)"

showit:
	@echo "Modules:$(word 1, $(MODULES))"
	@$(foreach m, $(filter-out $(word 1, $(MODULES)), $(MODULES)), \
	  echo -e "\t$(m)" ;)
	@echo "Libraries:$(word 1, $(ALLLIBS))"
	@$(foreach l, $(filter-out $(word 1, $(ALLLIBS)), $(ALLLIBS)), \
	  echo -e "\t$(l)" ;)

help:
	@$(MAKE) --print-data-base --question |               \
	awk '/^[^.%][-A-Za-z0-9_]*:/                          \
		{ print substr($$1, 1, length($$1)-1) }' |    \
	sort | uniq |                                         \
	pr -t -w 80 -4
