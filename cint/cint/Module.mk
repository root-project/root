# Module.mk for cint module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := cint
MODDIRBASE   := cint
MODDIR       := $(ROOT_SRCDIR)/$(MODDIRBASE)/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRSD     := $(MODDIRS)/dict
MODDIRI      := $(MODDIR)/inc

CINTDIR      := $(MODDIR)
CINTDIRS     := $(CINTDIR)/src
CINTDIRSD    := $(CINTDIRS)/dict
CINTDIRI     := $(CINTDIR)/inc
CINTDIRM     := $(CINTDIR)/main
CINTDIRL     := $(call stripsrc,$(CINTDIR)/lib)
CINTDIRDLLS  := $(call stripsrc,$(CINTDIR)/include)
CINTDIRSTL   := $(call stripsrc,$(CINTDIR)/stl)
CINTDIRDLLSTL:= $(CINTDIRL)/dll_stl
CINTDIRIOSEN := $(ROOT_SRCDIR)/$(MODDIRBASE)/iosenum
CINTDIRT     := $(ROOT_SRCDIR)/$(MODDIRBASE)/tool
ifneq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
CINTINCLUDES := $(CINTDIRL) $(CINTDIRDLLS) $(CINTDIRSTL)
endif

##### libCint #####
CINTCONF     := $(call stripsrc,$(CINTDIRI)/configcint.h)
CINTH        := $(filter-out $(CINTDIRI)/configcint.h,$(wildcard $(CINTDIRI)/*.h))
CINTHT       := $(patsubst $(CINTDIRI)/%.h,include/%.h,$(CINTH))
CINTS1       := $(wildcard $(MODDIRS)/*.c) \
                $(MODDIRS)/config/strlcpy.c $(MODDIRS)/config/strlcat.c \
                $(MODDIRS)/config/snprintf.c
CINTS2       := $(wildcard $(MODDIRS)/*.cxx) \
                $(MODDIRSD)/Apiif.cxx \
                $(MODDIRSD)/stdstrct.cxx

CINTS1       += $(CINTDIRM)/G__setup.c

CINTCONFMK   := $(ROOT_SRCDIR)/cint/ROOT/configcint.mk

CINTS1       := $(filter-out $(MODDIRS)/dlfcn.%,$(CINTS1))

CINTS2       := $(filter-out $(MODDIRS)/config/sunos.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/config/macos.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/config/winnt.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/config/newsos.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/loadfile_tmp.%,$(CINTS2))

# strip off possible leading path from compiler command name
CXXCMD       := $(shell echo $(CXX) | sed s/".*\/"//)

ifeq ($(CXXCMD),KCC)
CINTS2       += $(MODDIRSD)/kccstrm.cxx
else
ifeq ($(PLATFORM),linux)
CINTS2       += $(MODDIRSD)/libstrm.cxx
endif
ifeq ($(PLATFORM),hurd)
CINTS2       += $(MODDIRSD)/libstrm.cxx
endif
ifeq ($(PLATFORM),fbsd)
CINTS2       += $(MODDIRSD)/libstrm.cxx
endif
ifeq ($(PLATFORM),obsd)
CINTS2       += $(MODDIRSD)/libstrm.cxx
endif
ifeq ($(PLATFORM),hpux)
ifeq ($(ARCH),hpuxia64acc)
CINTS2       += $(MODDIRSD)/accstrm.cxx
else
CINTS2       += $(MODDIRSD)/libstrm.cxx
endif
endif
ifeq ($(PLATFORM),solaris)
 ifeq ($(SUNCC5),true)
  ifeq ($(findstring -library=stlport4,$(CXXFLAGS)),)
CINTS2       += $(MODDIRSD)/sunstrm.cxx
  else
CINTS2       += $(MODDIRSD)/stlport4strm.cxx
  endif
 else
CINTS2       += $(MODDIRSD)/libstrm.cxx
 endif
endif
ifeq ($(PLATFORM),aix3)
CINTS1       += $(MODDIRS)/dlfcn.c
CINTS2       += $(MODDIRSD)/libstrm.cxx
endif
ifeq ($(PLATFORM),aix)
CINTS2       += $(MODDIRSD)/libstrm.cxx
endif
ifeq ($(PLATFORM),aix5)
CINTS2       += $(MODDIRSD)/stlport4strm.cxx
endif
ifeq ($(PLATFORM),sgi)
CINTS2       += $(MODDIRSD)/libstrm.cxx
endif
ifeq ($(PLATFORM),alpha)
CINTS2       += $(MODDIRSD)/alphastrm.cxx
endif
ifeq ($(PLATFORM),alphagcc)
CINTS2       += $(MODDIRSD)/libstrm.cxx
endif
endif
ifeq ($(PLATFORM),sunos)
CINTS1       += $(MODDIRS)/config/sunos.c
endif
ifeq ($(PLATFORM),macos)
CINTS2       += $(MODDIRS)/config/macos.cxx
CINTS2       += $(MODDIRSD)/fakestrm.cxx
endif
ifeq ($(PLATFORM),macosx)
CINTS2       += $(MODDIRSD)/libstrm.cxx
endif
ifeq ($(PLATFORM),lynxos)
CINTS2       += $(MODDIRSD)/fakestrm.cxx
endif
ifeq ($(PLATFORM),win32)
CINTS2       += $(MODDIRS)/config/winnt.cxx
ifeq ($(VC_MAJOR),18)
  CINTS2       += $(MODDIRSD)/vc12strm.cxx
else
  ifeq ($(VC_MAJOR),17)
    CINTS2       += $(MODDIRSD)/vc11strm.cxx
  else
    ifeq ($(VC_MAJOR),16)
      CINTS2       += $(MODDIRSD)/vc10strm.cxx
    else
      ifeq ($(VC_MAJOR).$(VC_MINOR),13.10)
        CINTS2       += $(MODDIRSD)/vc7strm.cxx
      else
        ifeq ($(find $(VC_MAJOR),13 12 11 10 9 8 7 6 5 4 3 2 1),)
          CINTS2       += $(MODDIRSD)/vc7strm.cxx
        else
          CINTS2       += $(MODDIRSD)/iccstrm.cxx
        endif
      endif
    endif
  endif
endif
endif
ifeq ($(CXXCMD),icc)
CINTS2       := $(filter-out $(MODDIRSD)/libstrm.%,$(CINTS2))
 ifneq ($(ICC_GE_9),)
  ifneq ($(ICC_GE_101),)
 CINTS2       += $(MODDIRSD)/gcc4strm.cxx
  else
 CINTS2       += $(MODDIRSD)/gcc3strm.cxx
  endif
 else
 CINTS2       += $(MODDIRSD)/iccstrm.cxx
endif
endif
ifeq ($(GCC_MAJOR),3)
CINTS2       := $(filter-out $(MODDIRSD)/libstrm.%,$(CINTS2))
CINTS2       += $(MODDIRSD)/gcc3strm.cxx
endif
ifeq ($(GCC_MAJOR),4)
CINTS2       := $(filter-out $(MODDIRSD)/libstrm.%,$(CINTS2))
CINTS2       += $(MODDIRSD)/gcc4strm.cxx
endif
ifeq ($(GCC_MAJOR),5)
CINTS2       := $(filter-out $(MODDIRSD)/libstrm.%,$(CINTS2))
CINTS2       += $(MODDIRSD)/gcc4strm.cxx
endif
ifeq ($(GCC_MAJOR),6)
CINTS2       := $(filter-out $(MODDIRSD)/libstrm.%,$(CINTS2))
CINTS2       += $(MODDIRSD)/gcc4strm.cxx
endif
ifneq ($(CLANG_MAJOR),)
CINTS2       := $(filter-out $(MODDIRSD)/libstrm.%,$(CINTS2))
CINTS2       += $(MODDIRSD)/gcc4strm.cxx
endif
ifeq ($(LIBCXX),yes)
CINTS2       := $(filter-out $(MODDIRSD)/gcc4strm.%,$(CINTS2))
CINTS2       += $(MODDIRSD)/libcxxstrm.cxx
endif
ifeq ($(CXXCMD),xlC)
ifeq ($(PLATFORM),macosx)
CINTS2       := $(filter-out $(MODDIRSD)/libstrm.%,$(CINTS2))
CINTS2       += $(MODDIRSD)/gcc3strm.cxx
endif
endif

CINTS        := $(CINTS1) $(CINTS2)
CINTO        := $(call stripsrc,$(CINTS1:.c=.o) $(CINTS2:.cxx=.o))
CINTTMPO     := $(subst loadfile.o,loadfile_tmp.o,$(CINTO))
CINTTMPINC   := -I$(CINTDIRDLLS) -I$(CINTDIRSTL) -I$(CINTDIRL)
CINTDEP      := $(CINTO:.o=.d)
CINTDEP      += $(call stripsrc,$(MODDIRS)/loadfile_tmp.d)

CINTLIB      := $(LPATH)/libCint.$(SOEXT)

##### cint #####
CINTEXES     := $(CINTDIRM)/cppmain.cxx
CINTEXEO     := $(call stripsrc,$(CINTEXES:.cxx=.o))
CINTEXEDEP   := $(CINTEXEO:.o=.d)
CINTTMP      := $(call stripsrc,$(CINTDIRM)/cint_tmp$(EXEEXT))
CINT         := bin/cint$(EXEEXT)

##### makecint #####
MAKECINTS    := $(CINTDIRT)/makecint.cxx
MAKECINTO    := $(call stripsrc,$(MAKECINTS:.cxx=.o))
MAKECINT     := bin/makecint$(EXEEXT)

##### iosenum.h #####
IOSENUM      := $(call stripsrc,$(MODDIR)/include/iosenum.h)
IOSENUMC     := $(CINTDIRIOSEN)/iosenum.cxx
ifneq ($(CLANG_MAJOR),)
IOSENUMA     := $(CINTDIRIOSEN)/iosenum.$(ARCH)3
else
ifeq ($(GCC_MAJOR),6)
IOSENUMA     := $(CINTDIRIOSEN)/iosenum.$(ARCH)3
else
ifeq ($(GCC_MAJOR),5)
IOSENUMA     := $(CINTDIRIOSEN)/iosenum.$(ARCH)3
else
ifeq ($(GCC_MAJOR),4)
IOSENUMA     := $(CINTDIRIOSEN)/iosenum.$(ARCH)3
else
ifeq ($(GCC_MAJOR),3)
IOSENUMA     := $(CINTDIRIOSEN)/iosenum.$(ARCH)3
else
IOSENUMA     := $(CINTDIRIOSEN)/iosenum.$(ARCH)
endif
endif
endif
endif
endif

# used in the main Makefile
ALLHDRS     += $(CINTHT) $(CINTINCLUDES)

CINTSIZEFLAGS :=
ifneq ($(CINTMAXSTRUCT),)
CINTSIZEFLAGS += -DG__MAXSTRUCT=$(CINTMAXSTRUCT)
endif
ifneq ($(CINTMAXTYPEDEF),)
CINTSIZEFLAGS += -DG__MAXTYPEDEF=$(CINTMAXTYPEDEF)
endif
ifneq ($(CINTLONGLINE),)
CINTSIZEFLAGS += -DG__LONGLINE=$(CINTLONGLINE)
endif
CXXFLAGS += $(CINTSIZEFLAGS)

CINTCXXFLAGS += -DG__HAVE_CONFIG -DG__NOMAKEINFO -DG__CINTBODY $(CINTSIZEFLAGS)
CINTCFLAGS += -DG__HAVE_CONFIG -DG__NOMAKEINFO -DG__CINTBODY $(CINTSIZEFLAGS)
ifneq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
CINTCXXFLAGS += -I$(call stripsrc,$(CINTDIRI))
CINTCFLAGS += -I$(call stripsrc,$(CINTDIRI))
endif
CINTCXXFLAGS += -I$(CINTDIRI) -I$(CINTDIRS) -I$(CINTDIRSD)
CINTCFLAGS += -I$(CINTDIRI) -I$(CINTDIRS) -I$(CINTDIRSD)

##### used by cintdlls.mk #####
CINTDLLDIRSTL    := $(CINTDIRSTL)
CINTDLLDIRDLLS   := $(CINTDIRDLLS)
CINTDLLDIRDLLSTL := $(CINTDIRDLLSTL)
CINTDLLDIRL      := $(CINTDIRL)
CINTDLLIOSENUM   := $(IOSENUM)
CINTDLLDICTVER   := $(CINTDIRI)/cintdictversion.h
ifneq ($(HOST),)
CINTDLLCINTTMP   := $(BUILDTOOLSDIR)/$(CINTTMP)
else
CINTDLLCINTTMP   := $(CINTTMP)
endif
CINTDLLROOTCINTTMP    = $(ROOTCINTTMP)
CINTDLLROOTCINTTMPDEP = $(ROOTCINTTMPDEP)
CINTDLLCFLAGS    := $(filter-out -DG__CINTBODY,$(CINTCFLAGS))
CINTDLLCXXFLAGS  := $(filter-out -DG__CINTBODY,$(CINTCXXFLAGS))

# include all dependency files
INCLUDEFILES += $(CINTDEP) $(CINTEXEDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h: $(call stripsrc,$(CINTDIRI))/%.h
		cp $< $@

include/%.h: $(CINTDIRI)/%.h
		cp $< $@

ifneq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
$(CINTDIRL):
		$(MAKEDIR)
		@$(RSYNC) --exclude '.svn' --exclude '*.o' --exclude '*.d' --exclude 'rootcint_*' --exclude 'G__cpp_*' --exclude 'G__c_*' --exclude 'mktypes' --exclude '*.dSYM' $(CINTDIR)/lib $(dir $@)
		@touch $(CINTDIRL)
$(CINTDIRDLLS): $(CINTDIRL)
		$(MAKEDIR)
		@$(RSYNC) --exclude '.svn' --exclude '*.o' --exclude '*.d' --exclude '*.dll' --exclude 'systypes.h' --exclude 'types.h' $(CINTDIR)/include $(dir $@)
		@touch $(CINTDIRDLLS)
$(CINTDIRSTL):  $(CINTDIRL)
		$(MAKEDIR)
		@$(RSYNC) --exclude '.svn' --exclude '*.o' --exclude '*.d' --exclude '*.dll' $(CINTDIR)/stl $(dir $@)
		@touch $(CINTDIRSTL)
endif

$(CINTLIB):     $(CINTO)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libCint.$(SOEXT) $@ "$^" "$(CINTLIBEXTRA)"

$(CINT):        $(CINTEXEO) $(CINTLIB)
		$(LD) $(LDFLAGS) -o $@ $(CINTEXEO) \
		   $(RPATH) $(CINTLIBS) $(CILIBS)

$(CINTTMP):     $(CINTEXEO) $(CINTTMPO)
		$(LD) $(LDFLAGS) -o $@ $(CINTEXEO) \
		   $(CINTTMPO) $(CILIBS)

$(MAKECINT):    $(MAKECINTO)
		$(LD) $(LDFLAGS) -o $@ $(MAKECINTO)

$(IOSENUM): 
		$(MAKEDIR)
		@(if [ ! -r $(IOSENUMA) ]; then \
			echo "Missing $(IOSENUMA), run: make $(IOSENUMA)"; \
			$(MAKE) $(IOSENUMA); \
		fi; \
		cp $(IOSENUMA) $@)

$(IOSENUMA):    $(CINTTMP)
		@(if [ ! -r $@ ]; then \
			echo "Making $@..."; \
			$(CINTTMP) $(CINTTMPINC) \
			        -I$(ROOT_SRCDIR)/cint/cint/inc -Iinclude \
				$(IOSENUMC) > /dev/null; \
			mv iosenum.h $@; \
		else \
			touch $@; \
		fi)

all-$(MODNAME): $(CINTLIB) $(CINTTMP) $(IOSENUM)

clean-$(MODNAME):
		@rm -f $(CINTTMPO) $(CINTO) $(CINTEXEO) $(MAKECINTO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(CINTDEP) $(CINTLIB) $(IOSENUM) $(CINTEXEDEP) \
		   $(CINT) $(CINTTMP) $(MAKECINT) $(CINTDIRM)/*.exp \
		   $(CINTDIRM)/*.lib \
		   $(call stripsrc,$(CINTDIRS)/loadfile_tmp.cxx) \
		   $(CINTDIRDLLS)/sys/types.h $(CINTDIRDLLS)/systypes.h \
		   $(CINTHT) $(CINTCONF)
ifneq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
		@rm -rf $(CINTINCLUDES)
endif

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(call stripsrc,$(CINTDIRSD)/libstrm.o):  CINTCXXFLAGS += -I$(CINTDIRL)/stream
$(call stripsrc,$(CINTDIRSD)/sun5strm.o): CINTCXXFLAGS += -I$(CINTDIRL)/sunstrm
$(call stripsrc,$(CINTDIRSD)/vcstrm.o):   CINTCXXFLAGS += -I$(CINTDIRL)/vcstream
$(call stripsrc,$(CINTDIRSD)/%strm.o):    CINTCXXFLAGS += -I$(CINTDIRL)/$(notdir $(basename $@))
ifeq ($(GCC_MAJOR),4)
$(call stripsrc,$(CINTDIRSD)/gcc4strm.o): CINTCXXFLAGS += -Wno-strict-aliasing
endif
ifeq ($(GCC_MAJOR),5)
$(call stripsrc,$(CINTDIRSD)/gcc4strm.o): CINTCXXFLAGS += -Wno-strict-aliasing
endif
ifeq ($(GCC_MAJOR),6)
$(call stripsrc,$(CINTDIRSD)/gcc4strm.o): CINTCXXFLAGS += -Wno-strict-aliasing
endif


$(MAKECINTO) $(CINTO): $(CINTCONF) $(ORDER_) $(CINTINCLUDES)

$(MAKECINTO): CXXFLAGS := $(CINTCXXFLAGS)

$(call stripsrc,$(CINTDIRS)/loadfile_tmp.cxx): $(CINTDIRS)/loadfile.cxx
	$(MAKEDIR)
	cp -f $< $@
$(call stripsrc,$(CINTDIRS)/loadfile_tmp.o): $(CINTCONF) $(ORDER_) $(CINTINCLUDES)
$(call stripsrc,$(CINTDIRS)/loadfile_tmp.o): CINTCXXFLAGS += -UR__HAVE_CONFIG -DROOTBUILD
$(call stripsrc,$(CINTDIRS)/loadfile_tmp.o) $(CINTO): OPT := $(filter-out -Wshadow,$(OPT))
$(call stripsrc,$(CINTDIRS)/loadfile_tmp.o) $(CINTO): CXXFLAGS:=$(filter-out -Wshadow,$(CXXFLAGS))
ifeq ($(MACOSX_TMPNAM_DEPRECATED),yes)
$(call stripsrc,$(CINTDIRS)/loadfile_tmp.o) $(CINTO): CINTCXXFLAGS += -Wno-deprecated-declarations
$(call stripsrc,$(CINTDIRS)/loadfile_tmp.o) $(CINTO): CINTCFLAGS += -Wno-deprecated-declarations
endif

$(call stripsrc,$(CINTDIRSD)/stdstrct.o):    CINTCXXFLAGS += -I$(CINTDIRL)/stdstrct

ifeq ($(ICC_MAJOR),12)
ifeq ($(ICC_MINOR),0)
$(call stripsrc,$(CINTDIRS)/val2a.o): OPT := -O0
endif
endif

ifneq ($(subst -ftest-coverage,,$(OPT)),$(OPT))
# we have coverage on - not interesting for dictionaries
$(call stripsrc,$(subst .cxx,.o,$(wildcard $(CINTDIRSD)/*.cxx))): override OPT := $(subst -fprofile-arcs,,$(subst -ftest-coverage,,$(OPT)))
endif

##### configcint.h
ifeq ($(CPPPREP),)
# cannot use "CPPPREP?=", as someone might set "CPPPREP="
ifeq ($(GCC_MAJOR),6)
  CPPPREP = $(CXX) -std=c++98 -E -C
else
  CPPPREP = $(CXX) -E -C
endif
endif

include $(CINTCONFMK)
##### configcint.h - END

##### cintdlls #####
include $(ROOT_SRCDIR)/cint/ROOT/cintdlls.mk
