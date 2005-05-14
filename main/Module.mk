# Module.mk for main module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := main
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

MAINDIR      := $(MODDIR)
MAINDIRS     := $(MAINDIR)/src
MAINDIRI     := $(MAINDIR)/inc
MAINDIRW     := $(MAINDIR)/win32

##### root.exe #####
ROOTEXES     := $(MODDIRS)/rmain.cxx
ROOTEXEO     := $(ROOTEXES:.cxx=.o)
ROOTEXEDEP   := $(ROOTEXEO:.o=.d)
ifeq ($(ARCH),win32gcc)
ROOTEXE      := bin/root_exe.exe
else
ROOTEXE      := bin/root.exe
endif
ROOTNEXE     := bin/rootn.exe
ifeq ($(PLATFORM),win32)
ROOTICON     := icons/RootIcon.obj
endif

##### proofserv #####
PROOFSERVS   := $(MODDIRS)/pmain.cxx
PROOFSERVO   := $(PROOFSERVS:.cxx=.o)
PROOFSERVDEP := $(PROOFSERVO:.o=.d)
PROOFSERV    := bin/proofserv$(EXEEXT)
ifeq ($(PROOFLIB),)
PROOFSERV    :=
endif

##### hadd #####
HADDS        := $(MODDIRS)/hadd.cxx
HADDO        := $(HADDS:.cxx=.o)
HADDDEP      := $(HADDO:.o=.d)
HADD         := bin/hadd$(EXEEXT)

##### h2root #####
H2ROOTS1     := $(MODDIRS)/h2root.cxx
H2ROOTS2     := $(HBOOKS2)
# Symbols in cfopei.obj is already provided in packmd.lib,
#H2ROOTS3    := $(wildcard $(MAINDIRW)/*.c)
H2ROOTS3     := $(filter-out $(MAINDIRW)/cfopei.c, $(wildcard $(MAINDIRW)/*.c))
H2ROOTS4     := $(MAINDIRW)/tzvers.f
H2ROOTO      := $(H2ROOTS1:.cxx=.o) $(H2ROOTS2:.f=.o)
ifeq ($(PLATFORM),win32)
H2ROOTO      += $(H2ROOTS3:.c=.o) $(H2ROOTS4:.f=.o)
endif
H2ROOTDEP    := $(H2ROOTS1:.cxx=.d)
H2ROOT       := bin/h2root$(EXEEXT)

##### g2root #####
G2ROOTS      := $(MODDIRS)/g2root.f
G2ROOTO      := $(G2ROOTS:.f=.o)
ifeq ($(PLATFORM),win32)
G2ROOTO      += $(H2ROOTS3:.c=.o) $(H2ROOTS4:.f=.o)
endif
G2ROOT       := bin/g2root$(EXEEXT)
ifeq ($(PLATFORM),win32)
G2ROOT       :=
endif

##### g2rootold #####
G2ROOTOLDS   := $(MODDIRS)/g2rootold.f
G2ROOTOLDO   := $(G2ROOTOLDS:.f=.o)
ifeq ($(PLATFORM),win32)
G2ROOTOLDO   += $(H2ROOTS3:.c=.o) $(H2ROOTS4:.f=.o)
endif
G2ROOTOLD    := bin/g2rootold$(EXEEXT)
ifeq ($(PLATFORM),win32)
G2ROOTOLD    :=
endif

##### ssh2rpd #####
SSH2RPDS        := $(MODDIRS)/ssh2rpd.cxx
SSH2RPDO        := $(SSH2RPDS:.cxx=.o)
SSH2RPDDEP      := $(SSH2RPDO:.o=.d)
SSH2RPD         := bin/ssh2rpd$(EXEEXT)
ifeq ($(PLATFORM),win32)
SSH2RPD         :=
endif

# used in the main Makefile
ALLEXECS     += $(ROOTEXE) $(ROOTNEXE) $(PROOFSERV) $(HADD) $(SSH2RPD)
ifneq ($(CERNLIBS),)
ALLEXECS     += $(H2ROOT) $(G2ROOT) $(G2ROOTOLD)
endif

# include all dependency files
INCLUDEFILES += $(ROOTEXEDEP) $(PROOFSERVDEP) $(HADDDEP) $(H2ROOTDEP) \
                $(SSH2RPDDEP)

##### local rules #####
$(ROOTEXE):     $(ROOTEXEO) $(ROOTLIBSDEP) $(RINTLIB)
		$(LD) $(LDFLAGS) -o $@ $(ROOTEXEO) $(ROOTICON) $(ROOTULIBS) \
		   $(RPATH) $(ROOTLIBS) $(RINTLIBS) $(SYSLIBS)

$(ROOTNEXE):    $(ROOTEXEO) $(NEWLIB) $(ROOTLIBSDEP) $(RINTLIB)
		$(LD) $(LDFLAGS) -o $@ $(ROOTEXEO) $(ROOTICON) $(ROOTULIBS) \
		   $(RPATH) $(NEWLIBS) $(ROOTLIBS) $(RINTLIBS) $(SYSLIBS)

$(PROOFSERV):   $(PROOFSERVO) $(ROOTLIBSDEP) $(PROOFLIB) \
                $(TREEPLAYERLIB) $(THREADLIB)
		$(LD) $(LDFLAGS) -o $@ $(PROOFSERVO) $(ROOTULIBS) \
		   $(RPATH) $(ROOTLIBS) $(PROOFLIBS) $(SYSLIBS)

$(HADD):        $(HADDO) $(ROOTLIBSDEP) $(MATRIXLIB)
		$(LD) $(LDFLAGS) -o $@ $(HADDO) $(ROOTULIBS) \
		   $(RPATH) $(ROOTLIBS) $(SYSLIBS)

$(SSH2RPD):     $(SSH2RPDO) $(SNPRINTFO)
		$(LD) $(LDFLAGS) -o $@ $(SSH2RPDO) $(SNPRINTFO) $(SYSLIBS)

$(H2ROOT):      $(H2ROOTO) $(ROOTLIBSDEP)
		$(LD) $(LDFLAGS) -o $@ $(H2ROOTO) \
		   $(RPATH) $(ROOTLIBS) \
		   $(CERNLIBDIR) $(CERNLIBS) $(RFIOLIBEXTRA) $(SHIFTLIBDIR) \
		   $(SHIFTLIB) $(F77LIBS) $(SYSLIBS)

$(G2ROOT):      $(G2ROOTO)
		$(F77LD) $(F77LDFLAGS) -o $@ $(G2ROOTO) \
		   $(CERNLIBDIR) $(CERNLIBS) $(RFIOLIBEXTRA) $(SHIFTLIBDIR) \
		   $(SHIFTLIB) $(F77LIBS) $(SYSLIBS)

$(G2ROOTOLD):   $(G2ROOTOLDO)
		$(F77LD) $(F77LDFLAGS) -o $@ $(G2ROOTOLDO) \
		   $(CERNLIBDIR) $(CERNLIBS) $(RFIOLIBEXTRA) $(SHIFTLIBDIR) \
		   $(SHIFTLIB) $(F77LIBS) $(SYSLIBS)

ifneq ($(CERNLIBS),)
all-main:      $(ROOTEXE) $(ROOTNEXE) $(PROOFSERV) $(HADD) $(SSH2RPD) \
               $(H2ROOT) $(G2ROOT) $(G2ROOTOLD)
else
all-main:      $(ROOTEXE) $(ROOTNEXE) $(PROOFSERV) $(HADD) $(SSH2RPD)
endif

clean-main:
		@rm -f $(ROOTEXEO) $(PROOFSERVO) $(HADDO) $(H2ROOTO) \
		   $(G2ROOTO) $(G2ROOTOLDO) $(SSH2RPDO)

clean::         clean-main

distclean-main: clean-main
		@rm -f $(ROOTEXEDEP) $(ROOTEXE) $(ROOTNEXE) $(PROOFSERVDEP) \
		   $(PROOFSERV) $(HADDDEP) $(HADD) $(H2ROOTDEP) $(H2ROOT) \
		   $(G2ROOT) $(G2ROOTOLD) $(SSH2RPDDEP) $(SSH2RPD)

distclean::     distclean-main
