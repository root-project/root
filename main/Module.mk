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
ROOTEXE      := bin/root.exe
ROOTNEXE     := bin/rootn.exe
ifeq ($(PLATFORM),win32)
ROOTICON     := icons/RootIcon.obj
endif

##### proofserv #####
PROOFSERVS   := $(MODDIRS)/pmain.cxx
PROOFSERVO   := $(PROOFSERVS:.cxx=.o)
PROOFSERVDEP := $(PROOFSERVO:.o=.d)
PROOFSERV    := bin/proofserv$(EXEEXT)

##### h2root #####
H2ROOTS1     := $(MODDIRS)/h2root.cxx
H2ROOTS2     := $(HBOOKS2)
H2ROOTS3     := $(wildcard $(MAINDIRW)/*.c)
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

# used in the main Makefile
ALLEXECS     += $(ROOTEXE) $(ROOTNEXE) $(PROOFSERV)
ifneq ($(CERNLIBS),)
ALLEXECS     += $(H2ROOT) $(G2ROOT)
endif

# include all dependency files
INCLUDEFILES += $(ROOTEXEDEP) $(PROOFSERVDEP) $(H2ROOTDEP)

##### local rules #####
$(ROOTEXE):     $(ROOTEXEO) $(CORELIB) $(CINTLIB) $(HISTLIB) \
                $(GRAFLIB) $(G3DLIB) $(TREELIB) $(MATRIXLIB) $(RINTLIB)
		$(LD) $(LDFLAGS) -o $@ $(ROOTEXEO) $(ROOTICON) \
		   $(RPATH) $(ROOTLIBS) $(RINTLIBS) $(SYSLIBS)

$(ROOTNEXE):    $(ROOTEXEO) $(NEWLIB) $(CORELIB) $(CINTLIB) $(HISTLIB) \
                $(GRAFLIB) $(G3DLIB) $(TREELIB) $(MATRIXLIB) $(RINTLIB)
		$(LD) $(LDFLAGS) -o $@ $(ROOTEXEO) $(ROOTICON) \
		   $(RPATH) $(NEWLIBS) $(ROOTLIBS) $(RINTLIBS) $(SYSLIBS)

$(PROOFSERV):   $(PROOFSERVO) $(CORELIB) $(CINTLIB) $(HISTLIB) \
                $(GRAFLIB) $(G3DLIB) $(TREELIB) $(MATRIXLIB) $(GPADLIB) \
                $(PROOFLIB) $(TREEPLAYERLIB)
		$(LD) $(LDFLAGS) -o $@ $(PROOFSERVO) \
		   $(RPATH) $(ROOTLIBS) $(PROOFLIBS) $(SYSLIBS)

$(H2ROOT):      $(H2ROOTO) $(CORELIB) $(CINTLIB) $(HISTLIB) \
                $(GRAFLIB) $(G3DLIB) $(TREELIB) $(MATRIXLIB)
		$(LD) $(LDFLAGS) -o $@ $(H2ROOTO) \
		   $(RPATH) $(ROOTLIBS) \
		   $(CERNLIBDIR) $(CERNLIBS) $(F77LIBS) $(SYSLIBS)

$(G2ROOT):      $(G2ROOTO)
		$(F77LD) $(F77LDFLAGS) -o $@ $(G2ROOTO) \
		   $(CERNLIBDIR) $(CERNLIBS) $(F77LIBS) $(SYSLIBS)

ifneq ($(CERNLIBS),)
all-main:      $(ROOTEXE) $(ROOTNEXE) $(PROOFSERV) $(H2ROOT) $(G2ROOT)
else
all-main:      $(ROOTEXE) $(ROOTNEXE) $(PROOFSERV)
endif

clean-main:
		@rm -f $(ROOTEXEO) $(PROOFSERVO) $(H2ROOTO) $(G2ROOTO)

clean::         clean-main

distclean-main: clean-main
		@rm -f $(ROOTEXEDEP) $(ROOTEXE) $(ROOTNEXE) $(PROOFSERVDEP) \
		   $(PROOFSERV) $(H2ROOTDEP) $(H2ROOT) $(G2ROOT)

distclean::     distclean-main
