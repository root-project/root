# Module.mk for main module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := main
MODDIR       := $(MODNAME)
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
ifneq ($(BUILDBOTHCINT),)
ROOT7EXE     := $(subst root,rootc7,$(ROOTEXE))
endif
ifneq ($(PLATFORM),win32)
ROOTNEXE     := bin/rootn.exe
endif

##### proofserv #####
PROOFSERVS   := $(MODDIRS)/pmain.cxx
PROOFSERVO   := $(PROOFSERVS:.cxx=.o)
PROOFSERVDEP := $(PROOFSERVO:.o=.d)
ifeq ($(ARCH),win32gcc)
PROOFSERVEXE := bin/proofserv_exe.exe
else
PROOFSERVEXE := bin/proofserv.exe
PROOFSERVSH  := bin/proofserv
#ifeq ($(PLATFORM),win32)
#PROOFSERVEXE :=
#PROOFSERVSH  :=
#endif
endif
ifeq ($(PROOFLIB),)
PROOFSERVEXE :=
PROOFSERVSH  :=
endif

##### roots.exe #####
ROOTSEXES   := $(MODDIRS)/roots.cxx
ROOTSEXEO   := $(ROOTSEXES:.cxx=.o)
ROOTSEXEDEP := $(ROOTSEXEO:.o=.d)
ifeq ($(ARCH),win32gcc)
ROOTSEXE    := bin/roots_exe.exe
else
ROOTSEXE    := bin/roots.exe
ROOTSSH     := bin/roots
endif
ifeq ($(PLATFORM),win32)
ROOTSEXE    :=
ROOTSSH     :=
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
#H2ROOTS4     := $(MAINDIRW)/tzvers.f
H2ROOTO      := $(H2ROOTS1:.cxx=.o) $(H2ROOTS2:.f=.o)
ifeq ($(PLATFORM),win32)
H2ROOTO      += $(H2ROOTS3:.c=.o)
endif
H2ROOTDEP    := $(H2ROOTS1:.cxx=.d)
H2ROOT       := bin/h2root$(EXEEXT)

##### g2root #####
G2ROOTS      := $(MODDIRS)/g2root.f
G2ROOTO      := $(G2ROOTS:.f=.o)
ifeq ($(PLATFORM),win32)
G2ROOTO      += $(H2ROOTS3:.c=.o)
endif
G2ROOT       := bin/g2root$(EXEEXT)
ifeq ($(PLATFORM),win32)
G2ROOT       :=
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
ALLEXECS     += $(ROOTEXE) $(ROOT7EXE) $(ROOTNEXE) $(PROOFSERVEXE) $(PROOFSERVSH) \
                $(HADD) $(SSH2RPD) $(ROOTSEXE) $(ROOTSSH)
ifneq ($(F77),)
ALLEXECS     += $(H2ROOT) $(G2ROOT)
endif

# include all dependency files
INCLUDEFILES += $(ROOTEXEDEP) $(PROOFSERVDEP) $(HADDDEP) $(H2ROOTDEP) \
                $(SSH2RPDDEP) $(ROOTSEXEDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

$(ROOTEXE):     $(ROOTEXEO) $(BOOTLIBSDEP) $(RINTLIB)
		$(LD) $(LDFLAGS) -o $@ $(ROOTEXEO) $(ROOTICON) $(BOOTULIBS) \
		   $(RPATH) $(BOOTLIBS) $(RINTLIBS) $(SYSLIBS)

ifneq ($(BUILDBOTHCINT),)
$(ROOT7EXE):    $(ROOTEXEO) $(subst Cint,Cint7,$(BOOTLIBSDEP)) $(RINTLIB) $(REFLEXLIB)
		$(LD) $(LDFLAGS) -o $@ $(ROOTEXEO) $(ROOTICON) $(BOOTULIBS) \
		   $(RPATH) $(subst Cint,Cint7,$(BOOTLIBS)) $(RINTLIBS) $(RFLX_REFLEXLL) $(SYSLIBS)
endif

ifneq ($(PLATFORM),win32)
$(ROOTNEXE):    $(ROOTEXEO) $(NEWLIB) $(BOOTLIBSDEP) $(RINTLIB)
		$(LD) $(LDFLAGS) -o $@ $(ROOTEXEO) $(ROOTICON) $(BOOTULIBS) \
		   $(RPATH) $(NEWLIBS) $(BOOTLIBS) $(RINTLIBS) $(SYSLIBS)
endif

$(PROOFSERVEXE): $(PROOFSERVO) $(BOOTLIBSDEP)
		$(LD) $(LDFLAGS) -o $@ $(PROOFSERVO) $(BOOTULIBS) \
		   $(RPATH) $(BOOTLIBS) $(SYSLIBS)

$(PROOFSERVSH): $(MAINDIRS)/proofserv.sh
		@echo "Install proofserv wrapper."
		@cp $< $@
		@chmod 0755 $@

$(ROOTSEXE):    $(ROOTSEXEO) $(BOOTLIBSDEP)
		$(LD) $(LDFLAGS) -o $@ $(ROOTSEXEO) $(BOOTULIBS) \
		   $(RPATH) $(BOOTLIBS) $(SYSLIBS)

$(ROOTSSH):     $(MAINDIRS)/roots.sh
		@echo "Install roots wrapper."
		@cp $< $@
		@chmod 0755 $@

$(HADD):        $(HADDO) $(ROOTLIBSDEP)
		$(LD) $(LDFLAGS) -o $@ $(HADDO) $(ROOTULIBS) \
		   $(RPATH) $(ROOTLIBS) $(SYSLIBS)

$(SSH2RPD):     $(SSH2RPDO) $(SNPRINTFO)
		$(LD) $(LDFLAGS) -o $@ $(SSH2RPDO) $(SNPRINTFO) $(SYSLIBS)

$(H2ROOT):      $(H2ROOTO) $(ROOTLIBSDEP) $(MINICERNLIB)
		$(LD) $(LDFLAGS) -o $@ $(H2ROOTO) \
		   $(RPATH) $(ROOTLIBS) $(MINICERNLIB) \
		   $(F77LIBS) $(SYSLIBS)

$(G2ROOT):      $(G2ROOTO) $(ORDER_) $(MINICERNLIB)
		$(F77LD) $(F77LDFLAGS) -o $@ $(G2ROOTO) \
		   $(RPATH) $(MINICERNLIB) \
		   $(F77LIBS) $(SYSLIBS)

ifneq ($(F77),)
all-$(MODNAME): $(ROOTEXE) $(ROOTNEXE) $(PROOFSERVEXE) $(PROOFSERVSH) \
                $(HADD) $(SSH2RPD) $(H2ROOT) $(G2ROOT) \
                $(ROOTSEXE) $(ROOTSSH)
else
all-$(MODNAME): $(ROOTEXE) $(ROOTNEXE) $(PROOFSERVEXE) $(PROOFSERVSH) \
                $(HADD) $(SSH2RPD) $(ROOTSEXE) $(ROOTSSH)
endif

clean-$(MODNAME):
		@rm -f $(ROOTEXEO) $(PROOFSERVO) $(HADDO) $(H2ROOTO) \
		   $(G2ROOTO) $(SSH2RPDO) $(ROOTSEXEO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(ROOTEXEDEP) $(ROOTEXE) $(ROOTNEXE) $(PROOFSERVDEP) \
		   $(PROOFSERVEXE) $(PROOFSERVSH) $(HADDDEP) $(HADD) \
		   $(H2ROOTDEP) $(H2ROOT) $(G2ROOT) \
		   $(SSH2RPDDEP) $(SSH2RPD) $(ROOTSEXEDEP) $(ROOTSEXE) \
		   $(ROOTSSH) $(subst root,rootc7,$(ROOTEXE))

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(SSH2RPDO): PCHCXXFLAGS =
$(PROOFSERVO): CXXFLAGS += $(AFSEXTRACFLAGS)
