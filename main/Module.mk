# Module.mk for main module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := main
MODDIR       := $(ROOT_SRCDIR)/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

MAINDIR      := $(MODDIR)
MAINDIRS     := $(MAINDIR)/src
MAINDIRI     := $(MAINDIR)/inc

##### root.exe #####
ROOTEXES     := $(MODDIRS)/rmain.cxx
ROOTEXEO     := $(call stripsrc,$(ROOTEXES:.cxx=.o))
ROOTEXEDEP   := $(ROOTEXEO:.o=.d)
ifeq ($(ARCH),win32gcc)
ROOTEXE      := bin/root_exe.exe
else
ROOTEXE      := bin/root.exe
endif
ifneq ($(PLATFORM),win32)
ROOTNEXE     := bin/rootn.exe
endif

##### proofserv #####
PROOFSERVS   := $(MODDIRS)/pmain.cxx
PROOFSERVO   := $(call stripsrc,$(PROOFSERVS:.cxx=.o))
PROOFSERVDEP := $(PROOFSERVO:.o=.d)
ifneq ($(findstring win32,$(ARCH)),)
PROOFSERVEXE := bin/proofserv_exe.exe
PROOFSERVSH  := bin/proofserv
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

##### xpdtest #####
XPDTESTS   := $(MODDIRS)/xpdtest.cxx
XPDTESTO   := $(call stripsrc,$(XPDTESTS:.cxx=.o))
XPDTESTDEP := $(XPDTESTO:.o=.d)
ifneq ($(PLATFORM),win32)
XPDTESTEXE := bin/xpdtest
endif
ifeq ($(PROOFLIB),)
XPDTESTEXE :=
endif
XPDTESTLIBS := -lProof -lTree -lHist -lRIO -lNet -lThread -lMatrix -lMathCore 
XPDTESTLIBSDEP = $(IOLIB) $(TREELIB) $(NETLIB) $(HISTLIB) $(PROOFLIB) \
                 $(THREADLIB) $(MATRIXLIB) $(MATHCORELIB)

##### roots.exe #####
ROOTSEXES   := $(MODDIRS)/roots.cxx
ROOTSEXEO   := $(call stripsrc,$(ROOTSEXES:.cxx=.o))
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
HADDO        := $(call stripsrc,$(HADDS:.cxx=.o))
HADDDEP      := $(HADDO:.o=.d)
HADD         := bin/hadd$(EXEEXT)

##### h2root #####
H2ROOTS1     := $(MODDIRS)/h2root.cxx
H2ROOTS2     := $(wildcard $(MODDIRS)/*.c)
H2ROOTO      := $(call stripsrc,$(H2ROOTS1:.cxx=.o))
ifeq ($(PLATFORM),win32)
H2ROOTO      += $(call stripsrc,$(H2ROOTS2:.c=.o))
endif
H2ROOTDEP    := $(H2ROOTO:.o=.d)
H2ROOT       := bin/h2root$(EXEEXT)

##### g2root #####
G2ROOTS      := $(MODDIRS)/g2root.f
G2ROOTO      := $(call stripsrc,$(G2ROOTS:.f=.o))
ifeq ($(PLATFORM),win32)
G2ROOTO      += $(call stripsrc,$(H2ROOTS2:.c=.o))
endif
G2ROOT       := bin/g2root$(EXEEXT)
ifeq ($(PLATFORM),win32)
G2ROOT       :=
endif

##### ssh2rpd #####
SSH2RPDS        := $(MODDIRS)/ssh2rpd.cxx
SSH2RPDO        := $(call stripsrc,$(SSH2RPDS:.cxx=.o))
SSH2RPDDEP      := $(SSH2RPDO:.o=.d)
SSH2RPD         := bin/ssh2rpd$(EXEEXT)
ifeq ($(PLATFORM),win32)
SSH2RPD         :=
endif

# used in the main Makefile
ALLEXECS     += $(ROOTEXE) $(ROOTNEXE) $(PROOFSERVEXE) $(PROOFSERVSH) \
                $(XPDTESTEXE) $(HADD) $(SSH2RPD) $(ROOTSEXE) $(ROOTSSH)
ifneq ($(F77),)
ALLEXECS     += $(H2ROOT) $(G2ROOT)
endif

# include all dependency files
INCLUDEFILES += $(ROOTEXEDEP) $(PROOFSERVDEP) $(XPDTESTDEP) $(HADDDEP) \
                $(H2ROOTDEP) $(SSH2RPDDEP) $(ROOTSEXEDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

$(ROOTEXE):     $(ROOTEXEO) $(BOOTLIBSDEP) $(RINTLIB)
		$(LD) $(LDFLAGS) -o $@ $(ROOTEXEO) $(ROOTICON) \
		   $(RPATH) $(BOOTLIBS) $(RINTLIBS) $(SYSLIBS)

ifneq ($(PLATFORM),win32)
$(ROOTNEXE):    $(ROOTEXEO) $(NEWLIB) $(BOOTLIBSDEP) $(RINTLIB)
		$(LD) $(LDFLAGS) -o $@ $(ROOTEXEO) $(ROOTICON) \
		   $(RPATH) $(NEWLIBS) $(BOOTLIBS) $(RINTLIBS) $(SYSLIBS)
endif

$(PROOFSERVEXE): $(PROOFSERVO) $(BOOTLIBSDEP)
		$(LD) $(LDFLAGS) -o $@ $(PROOFSERVO) \
		   $(RPATH) $(BOOTLIBS) $(SYSLIBS)

$(PROOFSERVSH): $(call stripsrc,$(MAINDIRS)/proofserv.sh)
		@echo "Install proofserv wrapper."
		@cp $< $@
		@chmod 0755 $@

$(XPDTESTEXE): $(XPDTESTO) $(BOOTLIBSDEP) $(XPDTESTLIBSDEP)
		$(LD) $(LDFLAGS) -o $@ $(XPDTESTO) \
		$(RPATH) $(BOOTLIBS) $(XPDTESTLIBS) $(SYSLIBS)

$(ROOTSEXE):    $(ROOTSEXEO) $(BOOTLIBSDEP)
		$(LD) $(LDFLAGS) -o $@ $(ROOTSEXEO) \
		   $(RPATH) $(BOOTLIBS) $(SYSLIBS)

$(ROOTSSH):     $(call stripsrc,$(MAINDIRS)/roots.sh)
		@echo "Install roots wrapper."
		@cp $< $@
		@chmod 0755 $@

$(HADD):        $(HADDO) $(ROOTLIBSDEP)
		$(LD) $(LDFLAGS) -o $@ $(HADDO) $(ROOTULIBS) \
		   $(RPATH) $(ROOTLIBS) $(SYSLIBS)

$(SSH2RPD):     $(SSH2RPDO) $(SNPRINTFO) $(STRLCPYO)
		$(LD) $(LDFLAGS) -o $@ $(SSH2RPDO) $(SNPRINTFO) $(STRLCPYO) \
		   $(SYSLIBS)

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
                $(XPDTESTEXE) $(HADD) $(SSH2RPD) $(H2ROOT) $(G2ROOT) \
                $(ROOTSEXE) $(ROOTSSH)
else
all-$(MODNAME): $(ROOTEXE) $(ROOTNEXE) $(PROOFSERVEXE) $(PROOFSERVSH) \
                $(XPDTESTEXE) $(HADD) $(SSH2RPD) $(ROOTSEXE) $(ROOTSSH)
endif

clean-$(MODNAME):
		@rm -f $(ROOTEXEO) $(PROOFSERVO) $(XPDTESTO) $(HADDO) \
		   $(H2ROOTO) $(G2ROOTO) $(SSH2RPDO) $(ROOTSEXEO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(ROOTEXEDEP) $(ROOTEXE) $(ROOTNEXE) $(PROOFSERVDEP) \
		   $(PROOFSERVEXE) $(PROOFSERVSH)  $(XPDTESTDEP) $(XPDTESTEXE) \
		   $(HADDDEP) $(HADD) $(H2ROOTDEP) $(H2ROOT) $(G2ROOT) \
		   $(SSH2RPDDEP) $(SSH2RPD) $(ROOTSEXEDEP) $(ROOTSEXE) \
		   $(ROOTSSH)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(PROOFSERVO): CXXFLAGS += $(AFSEXTRACFLAGS)
