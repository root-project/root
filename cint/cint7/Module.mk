# Module.mk for cint7 module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME       := cint7
MODDIRBASE    := cint
MODDIR        := $(MODDIRBASE)/$(MODNAME)
MODDIRS       := $(MODDIR)/src
MODDIRI       := $(MODDIR)/inc

CINT7DIR      := $(MODDIR)
CINT7DIRS     := $(CINT7DIR)/src
CINT7DIRI     := $(CINT7DIR)/inc
CINT7DIRM     := $(CINT7DIR)/main
CINT7DIRT     := $(MODDIRBASE)/tool
CINT7DIRL     := $(CINT7DIR)/lib
CINT7DIRDLLS  := $(CINT7DIR)/include
CINT7DIRSTL   := $(CINT7DIR)/stl
CINT7DIRDLLSTL:= $(CINT7DIRL)/dll_stl
CINT7DIRSD    := $(CINT7DIRS)/dict
CINT7DIRIOSEN := $(MODDIRBASE)/iosenum

##### libCint7 #####
CINT7CONF     := $(CINT7DIRI)/configcint.h
CINT7H        := $(filter-out $(CINT7CONF),$(wildcard $(CINT7DIRI)/*.h))
ifeq ($(BUILDBOTHCINT),)
CINT7HT       := $(sort $(patsubst $(CINT7DIRI)/%.h,include/%.h,$(CINT7H) $(CINT7CONF)))
else
CINT7HT       := $(sort $(patsubst $(CINT7DIRI)/%.h,include/cint7/%.h,$(CINT7H) $(CINT7CONF)))
endif
CINT7S1       := $(wildcard $(MODDIRS)/*.c)
CINT7S2       := $(wildcard $(CINT7DIRS)/*.cxx) $(CINT7DIRSD)/longif.cxx $(CINT7DIRSD)/Apiif.cxx $(CINT7DIRSD)/stdstrct.o

CINT7S1       += $(CINT7DIRM)/G__setup.c

CINT7ALLO     := $(CINT7S1:.c=.o) $(CINT7S2:.cxx=.o)
CINT7ALLDEP   := $(CINT7ALLO:.o=.d)

CINT7CONFMK   := $(MODDIRBASE)/ROOT/configcint.mk

CINT7S1       := $(filter-out $(MODDIRS)/dlfcn.%,$(CINT7S1))

CINT7S2       := $(filter-out $(MODDIRS)/sunos.%,$(CINT7S2))
CINT7S2       := $(filter-out $(MODDIRS)/macos.%,$(CINT7S2))
CINT7S2       := $(filter-out $(MODDIRS)/winnt.%,$(CINT7S2))
CINT7S2       := $(filter-out $(MODDIRS)/newsos.%,$(CINT7S2))
CINT7S2       := $(filter-out $(MODDIRS)/loadfile_tmp.%,$(CINT7S2))
CINT7S2       := $(filter-out $(MODDIRS)/pragma_tmp.%,$(CINT7S2))

# strip off possible leading path from compiler command name
CXXCMD       := $(shell echo $(CXX) | sed s/".*\/"//)

ifeq ($(CXXCMD),KCC)
CINT7S2       += $(CINT7DIRSD)/kccstrm.cxx
CINT7S2       := $(filter-out $(CINT7DIRSD)/longif.%,$(CINT7S2))
CINT7S2       += $(CINT7DIRSD)/longif3.cxx
else
ifeq ($(PLATFORM),linux)
CINT7S2       += $(CINT7DIRSD)/libstrm.cxx
endif
ifeq ($(PLATFORM),hurd)
CINT7S2       += $(CINT7DIRSD)/libstrm.cxx
endif
ifeq ($(PLATFORM),fbsd)
CINT7S2       += $(CINT7DIRSD)/libstrm.cxx
endif
ifeq ($(PLATFORM),obsd)
CINT7S2       += $(CINT7DIRSD)/libstrm.cxx
endif
ifeq ($(PLATFORM),hpux)
ifeq ($(ARCH),hpuxia64acc)
CINT7S2       += $(CINT7DIRSD)/accstrm.cxx
CINT7S2       := $(filter-out $(CINT7DIRSD)/longif.%,$(CINT7S2))
CINT7S2       += $(CINT7DIRSD)/longif3.cxx
else
CINT7S2       += $(CINT7DIRSD)/libstrm.cxx
endif
endif
ifeq ($(PLATFORM),solaris)
 ifeq ($(SUNCC5),true)
CINT7S2       := $(filter-out $(CINT7DIRSD)/longif.%,$(CINT7S2))
CINT7S2       += $(CINT7DIRSD)/longif3.cxx
  ifeq ($(findstring -library=stlport4,$(CXXFLAGS)),)
CINT7S2       += $(CINT7DIRSD)/sunstrm.cxx
  else
CINT7S2       += $(CINT7DIRSD)/stlport4strm.cxx
  endif
 else
CINT7S2       += $(CINT7DIRSD)/libstrm.cxx
 endif
endif
ifeq ($(PLATFORM),aix3)
CINT7S1       += $(CINT7DIRS)/dlfcn.c
CINT7S2       += $(CINT7DIRSD)/libstrm.cxx
endif
ifeq ($(PLATFORM),aix)
CINT7S2       += $(CINT7DIRSD)/libstrm.cxx
endif
ifeq ($(PLATFORM),aix5)
CINT7S2       += $(CINT7DIRSD)/libstrm.cxx
endif
ifeq ($(PLATFORM),sgi)
CINT7S2       += $(CINT7DIRSD)/libstrm.cxx
endif
ifeq ($(PLATFORM),alpha)
CINT7S2       += $(CINT7DIRSD)/alphastrm.cxx
endif
ifeq ($(PLATFORM),alphagcc)
CINT7S2       += $(CINT7DIRSD)/libstrm.cxx
endif
endif
ifeq ($(PLATFORM),sunos)
CINT7S1       += $(CINT7DIRS)/sunos.c
endif
ifeq ($(PLATFORM),macos)
CINT7S2       += $(CINT7DIRS)/macos.cxx
CINT7S2       += $(CINT7DIRSD)/fakestrm.cxx
endif
ifeq ($(PLATFORM),macosx)
CINT7S2       += $(CINT7DIRSD)/libstrm.cxx
endif
ifeq ($(PLATFORM),lynxos)
CINT7S2       += $(CINT7DIRSD)/fakestrm.cxx
endif
ifeq ($(PLATFORM),win32)
CINT7S2       += $(CINT7DIRS)/winnt.cxx
CINT7S2       := $(filter-out $(CINT7DIRSD)/longif.%,$(CINT7S2))
CINT7S2       += $(CINT7DIRSD)/longif3.cxx
ifeq ($(VC_MAJOR),13)
 ifeq ($(VC_MINOR),10)
  CINT7S2       += $(CINT7DIRSD)/vc7strm.cxx
 else
  CINT7S2       += $(CINT7DIRSD)/iccstrm.cxx
 endif
else
 ifeq ($(VC_MAJOR),14)
  CINT7S2       += $(CINT7DIRSD)/vc7strm.cxx
 else
  CINT7S2       += $(CINT7DIRSD)/iccstrm.cxx
 endif
endif
endif
ifeq ($(PLATFORM),vms)
CINT7S2       += $(CINT7DIRSD)/fakestrm.cxx
endif
ifeq ($(CXXCMD),icc)
CINT7S2       := $(filter-out $(CINT7DIRSD)/libstrm.%,$(CINT7S2))
CINT7S2       := $(filter-out $(CINT7DIRSD)/longif.%,$(CINT7S2))
ifneq ($(ICC_GE_9),)
CINT7S2       += $(CINT7DIRSD)/gcc3strm.cxx
else
CINT7S2       += $(CINT7DIRSD)/iccstrm.cxx
endif
CINT7S2       += $(CINT7DIRSD)/longif3.cxx
endif
ifeq ($(GCC_MAJOR),3)
CINT7S2       := $(filter-out $(CINT7DIRSD)/libstrm.%,$(CINT7S2))
CINT7S2       := $(filter-out $(CINT7DIRSD)/longif.%,$(CINT7S2))
CINT7S2       += $(CINT7DIRSD)/gcc3strm.cxx
CINT7S2       += $(CINT7DIRSD)/longif3.cxx
endif
ifeq ($(GCC_MAJOR),4)
CINT7S2       := $(filter-out $(CINT7DIRSD)/libstrm.%,$(CINT7S2))
CINT7S2       := $(filter-out $(CINT7DIRSD)/longif.%,$(CINT7S2))
CINT7S2       += $(CINT7DIRSD)/gcc4strm.cxx
CINT7S2       += $(CINT7DIRSD)/longif3.cxx
endif
ifeq ($(CXXCMD),xlC)
ifeq ($(PLATFORM),macosx)
CINT7S2       := $(filter-out $(CINT7DIRSD)/libstrm.%,$(CINT7S2))
CINT7S2       := $(filter-out $(CINT7DIRSD)/longif.%,$(CINT7S2))
CINT7S2       += $(CINT7DIRSD)/gcc3strm.cxx
CINT7S2       += $(CINT7DIRSD)/longif3.cxx
endif
endif

CINT7S        := $(CINT7S1) $(CINT7S2)
CINT7O        := $(CINT7S1:.c=.o) $(CINT7S2:.cxx=.o)
CINT7TMPO     := $(subst loadfile.o,loadfile_tmp.o,$(CINT7O))
CINT7TMPO     := $(subst pragma.o,pragma_tmp.o,$(CINT7TMPO))
CINT7TMPINC   := -I$(MODDIR)/inc -I$(MODDIR)/include -I$(MODDIR)/stl -I$(MODDIR)/lib -Iinclude
CINT7DEP      := $(CINT7O:.o=.d)
CINT7DEP      += $(CINT7DIRS)/loadfile_tmp.d
CINT7DEP      += $(CINT7DIRS)/pragma_tmp.d
CINT7ALLDEP   += $(CINT7DIRS)/loadfile_tmp.d
CINT7ALLDEP   += $(CINT7DIRS)/pragma_tmp.d

ifeq ($(BUILDBOTHCINT),)
CINT7LIB      := $(LPATH)/libCint.$(SOEXT)
else
CINT7LIB      := $(LPATH)/libCint7.$(SOEXT)
endif

##### cint #####
CINT7EXES     := $(CINT7DIRM)/cppmain.cxx
CINT7EXEO     := $(CINT7EXES:.cxx=.o)
CINT7EXEDEP   := $(CINT7EXEO:.o=.d)
CINT7TMP      := $(CINT7DIRM)/cint_tmp$(EXEEXT)
ifeq ($(BUILDBOTHCINT),)
CINT7         := bin/cint$(EXEEXT)
else
CINT7         := bin/cint7$(EXEEXT)
endif

##### makecint #####
MAKECINT7S    := $(CINT7DIRT)/makecint.cxx
MAKECINT7O    := $(MAKECINT7S:.cxx=.o)
ifeq ($(BUILDBOTHCINT),)
MAKECINT7     := bin/makecint$(EXEEXT)
else
MAKECINT7     := bin/makecint7$(EXEEXT)
endif

##### iosenum.h #####
IOSENUM7      := $(MODDIR)/include/iosenum.h
IOSENUM7C     := $(MODDIR)/include/iosenum.cxx
ifeq ($(GCC_MAJOR),4)
IOSENUM7A     := $(CINT7DIRIOSEN)/iosenum.$(ARCH)3
else
ifeq ($(GCC_MAJOR),3)
IOSENUM7A     := $(CINT7DIRIOSEN)/iosenum.$(ARCH)3
else
IOSENUM7A     := $(CINT7DIRIOSEN)/iosenum.$(ARCH)
endif
endif

CINT7_STDIOH   := $(MODDIR)/include/stdio.h
CINT7_MKINCLD  := $(MODDIR)/include/mkincld
CINT7_MKINCLDS := $(MODDIR)/include/mkincld.c
CINT7_MKINCLDO := $(MODDIR)/include/mkincld.o

# used in the main Makefile
ALLHDRS     += $(CINT7HT)

ALLLIBS      += $(CINT7LIB)
ALLEXECS     += $(CINT7TMP)

# include all dependency files
INCLUDEFILES += $(CINT7DEP) $(CINT7EXEDEP)

# Make sure -Iinclude/ is _after_ CINT7 -I
CINT7CXXFLAGS = $(subst -Iinclude ,-I$(CINT7DIRI) -I$(CINT7DIRS) -I$(CINT7DIRSD) ,\
                   $(patsubst -Icint/cint/%,,$(CINTCXXFLAGS))) \
                -Iinclude
CINT7CFLAGS   = $(subst -Iinclude ,-I$(CINT7DIRI) -I$(CINT7DIRS) -I$(CINT7DIRSD) ,\
                   $(patsubst -Icint/cint/%,,$(CINTCFLAGS))) \
                -Iinclude

CINT7CXXFLAGS += -DG__CINTBODY -DG__HAVE_CONFIG -DG__NOMAKEINFO
CINT7CFLAGS   += -DG__CINTBODY -DG__HAVE_CONFIG -DG__NOMAKEINFO

##### used by configcint.mk #####
G__CFG_CXXFLAGS := $(CINT7CXXFLAGS)
G__CFG_CFLAGS   := $(CINT7CFLAGS)
G__CFG_DIR      := $(CINT7DIR)
G__CFG_CONF     := $(CINT7CONF)
G__CFG_CONFMK   := $(CINT7CONFMK)

##### used by cintdlls.mk #####
ifneq ($(BUILDBOTHCINT),)
CINTDLLDIRSTL_BAK := $(CINTDLLDIRSTL) 
CINTDLLDIRDLLS_BAK := $(CINTDLLDIRDLLS) 
CINTDLLDIRDLLSTL_BAK := $(CINTDLLDIRDLLSTL)
CINTDLLDIRL_BAK := $(CINTDLLDIRL) 
CINTDLLIOSENUM_BAK := $(CINTDLLIOSENUM) 
CINTDLLDICTVER_BAK := $(CINTDLLDICTVER) 
CINTDLLCINTTMP_BAK := $(CINTDLLCINTTMP) 
CINTDLLCFLAGS_BAK := $(CINTDLLCFLAGS) 
CINTDLLCXXFLAGS_BAK := $(CINTDLLCXXFLAGS) 
endif

CINTDLLDIRSTL    := $(CINT7DIRSTL)
CINTDLLDIRDLLS   := $(CINT7DIRDLLS)
CINTDLLDIRDLLSTL := $(CINT7DIRDLLSTL)
CINTDLLDIRL      := $(CINT7DIRL)
CINTDLLIOSENUM   := $(IOSENUM7)
CINTDLLDICTVER   := $(CINT7DIRI)/cintdictversion.h
CINTDLLCINTTMP   := $(CINT7TMP)
CINTDLLCFLAGS    := $(filter-out -DG__CINTBODY,$(CINT7CFLAGS))
CINTDLLCXXFLAGS  := $(filter-out -DG__CINTBODY,$(CINT7CXXFLAGS))

# the ROOT-specific cintdll dictionary part is currently built with
# CINT5's rootcint because it's protected with a ifeq(BUILDINGCINT,5).
# Nevertheless, this is what it will look like for CINT7
CINTDLLROOTCINTTMPDEP = $(ROOTCINT7TMPDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

ifeq ($(BUILDBOTHCINT),)
include/%.h: $(CINT7DIRI)/%.h
		@(if [ ! -d "include" ]; then    \
			mkdir -p include;             \
		fi)
		cp $< $@
else
include/cint7/%.h: $(CINT7DIRI)/%.h
		@(if [ ! -d "include/cint7" ]; then    \
			mkdir -p include/cint7;             \
		fi)
		cp $< $@
endif

$(CINT7LIB):    $(CINT7O) $(CINT7LIBDEP) $(REFLEXLIB)
		$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" "$(SOFLAGS)" \
		   $(notdir $(CINT7LIB)) $@ "$(CINT7O)" "$(CINT7LIBEXTRA) $(REFLEXLL)"

$(CINT7):       $(CINT7EXEO) $(CINT7LIB) $(REFLEXLIB)
		$(LD) $(LDFLAGS) -o $@ $(CINT7EXEO) $(RPATH) $(CINT7LIBS) $(CILIBS)

$(CINT7TMP):    $(CINT7EXEO) $(CINT7TMPO) $(REFLEXLIB)
		$(LD) $(LDFLAGS) -o $@ $(CINT7EXEO) $(CINT7TMPO) $(RPATH) \
		   $(REFLEXO) $(CILIBS)

$(MAKECINT7):   $(MAKECINT7O)
		$(LD) $(LDFLAGS) -o $@ $(MAKECINT7O)

$(IOSENUM7):    $(IOSENUM7A)
		cp $< $@

# Duplicate rule of the one in cint/Module.mk
#$(IOSENUM7A):   $(CINT7TMP) $(CINT7_STDIOH)
#		@(if [ ! -r $@ ]; then \
#			echo "Making $@..."; \
#			$(CINT7TMP) $(CINT7TMPINC) -Iinclude $(IOSENUM7C) > /dev/null ; \
#			mv iosenum.h $@ ; \
#		else \
#			touch $@ ; \
#		fi)

$(CINT7_STDIOH): $(CINT7_MKINCLDS)
		$(CC) $(OPT) $(CINT7CFLAGS) $(CXXOUT)$(CINT7_MKINCLDO) -c $<
		$(LD) $(LDFLAGS) -o $(CINT7_MKINCLD) $(CINT7_MKINCLDO)
		cd $(dir $(CINT7_MKINCLD)) ; ./mkincld

all-$(MODNAME): $(CINT7LIB) $(CINT7TMP) $(IOSENUM7)

clean-$(MODNAME):
		@rm -f $(CINT7TMPO) $(CINT7ALLO) $(CINT7EXEO) $(MAKECINT7O)

clean:: clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -rf $(CINT7ALLDEP) $(CINT7LIB) $(IOSENUM7) $(CINT7EXEDEP) \
		   $(CINT7) $(CINT7TMP) $(MAKECINT7) $(CINT7DIRM)/*.exp \
         $(CINT7DIRM)/*.lib $(CINT7DIRS)/loadfile_tmp.cxx \
		   $(CINT7DIRDLLS)/sys/types.h $(CINT7DIRDLLS)/systypes.h \
		   $(CINT7DIRS)/pragma_tmp.cxx \
		   $(CINT7HT) $(CINT7CONF)
		   @rm -rf include/cint7

ifeq ($(BUILDBOTHCINT),)
# Transition rules
all-cint: all-cint7
clean-cint: clean-cint7
distclean-cint: distclean-cint7

IOSENUM  = $(IOSENUM7)
CINTO    = $(CINT7O)
CINTALLO = $(CINT7ALLO)
CINTTMPO = $(CINT7TMPO) $(REFLEXO)
CINTLIB  = $(CINT7LIB)
CINTLIBS := $(subst Cint7,Cint,$(CINT7LIBS))
BOOTLIBS := $(BOOTLIBS) $(RFLX_REFLEXLL)
ROOTLIBS := $(ROOTLIBS) $(RFLX_REFLEXLL) 

CINTLIB : $(REFLEXLIB)
endif

distclean:: distclean-$(MODNAME)

##### extra rules ######
$(CINT7DIRS)/libstrm.o:  CINT7CXXFLAGS += -I$(CINT7DIRL)/stream
$(CINT7DIRS)/sunstrm.o:  CINT7CXXFLAGS += -I$(CINT7DIRL)/snstream
$(CINT7DIRS)/sun5strm.o: CINT7CXXFLAGS += -I$(CINT7DIRL)/snstream
$(CINT7DIRS)/vcstrm.o:   CINT7CXXFLAGS += -I$(CINT7DIRL)/vcstream
$(CINT7DIRS)/%strm.o:    CINT7CXXFLAGS += -I$(CINT7DIRL)/$(notdir $(basename $@))
ifeq ($(GCC_MAJOR),4)
$(CINT7DIRSD)/gcc4strm.o:  CINT7CXXFLAGS += -Wno-strict-aliasing
endif

$(MAKECINT7O) $(CINT7ALLO): $(CINT7CONF)

$(MAKECINT7O):               CXXFLAGS := $(CINT7CXXFLAGS)
$(CINT7DIRSD)/stdstrct.o:    CINT7CXXFLAGS += -I$(CINT7DIRL)/stdstrct
$(CINT7DIRS)/loadfile_tmp.o: CINT7CXXFLAGS += -UHAVE_CONFIG -DROOTBUILD -DG__BUILDING_CINTTMP
$(CINT7DIRS)/pragma_tmp.o:   CINT7CXXFLAGS += -UHAVE_CONFIG -DROOTBUILD -DG__BUILDING_CINTTMP

$(CINT7DIRS)/loadfile_tmp.cxx: $(CINT7DIRS)/loadfile.cxx
		cp -f $< $@

$(CINT7DIRS)/pragma_tmp.cxx: $(CINT7DIRS)/pragma.cxx
		cp -f $< $@

##### configcint.h
ifeq ($(CPPPREP),)
# cannot use "CPPPREP?=", as someone might set "CPPPREP="
   CPPPREP = $(CXX) -E -C
endif
include $(CINT7CONFMK)
##### configcint.h - END

##### cintdlls #####
include cint/ROOT/cintdlls.mk

ifneq ($(BUILDBOTHCINT),)
CINTDLLDIRSTL := $(CINTDLLDIRSTL_BAK) 
CINTDLLDIRDLLS := $(CINTDLLDIRDLLS_BAK) 
CINTDLLDIRDLLSTL := $(CINTDLLDIRDLLSTL_BAK)
CINTDLLDIRL := $(CINTDLLDIRL_BAK) 
CINTDLLIOSENUM := $(CINTDLLIOSENUM_BAK) 
CINTDLLDICTVER := $(CINTDLLDICTVER_BAK) 
CINTDLLCINTTMP := $(CINTDLLCINTTMP_BAK) 
CINTDLLCFLAGS := $(CINTDLLCFLAGS_BAK) 
CINTDLLCXXFLAGS := $(CINTDLLCXXFLAGS_BAK) 
endif

