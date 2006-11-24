# Module.mk for cint module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := cint
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

CINTDIR      := $(MODDIR)
CINTDIRS     := $(CINTDIR)/src
CINTDIRI     := $(CINTDIR)/inc
CINTDIRM     := $(CINTDIR)/main
CINTDIRT     := $(CINTDIR)/tool
CINTDIRL     := $(CINTDIR)/lib
CINTDIRDLLS  := $(CINTDIR)/include
CINTDIRSTL   := $(CINTDIR)/stl
CINTDIRDLLSTL:= $(CINTDIRL)/dll_stl

##### libCint #####
CINTH        := $(wildcard $(MODDIRI)/*.h)
CINTS1       := $(wildcard $(MODDIRS)/*.c)
CINTS2       := $(wildcard $(MODDIRS)/*.cxx)

CINTS1       += $(CINTDIRM)/G__setup.c

CINTALLO     := $(CINTS1:.c=.o) $(CINTS2:.cxx=.o)
CINTALLDEP   := $(CINTALLO:.o=.d)

CINTCONF     := include/configcint.h
CINTCONFMK   := $(MODDIR)/configcint.mk

CINTS1       := $(filter-out $(MODDIRS)/dlfcn.%,$(CINTS1))

CINTS2       := $(filter-out $(MODDIRS)/v6_sunos.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/v6_macos.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/v6_winnt.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/v6_newsos.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/v6_loadfile_tmp.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/allstrm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/kccstrm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/sunstrm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/sun5strm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/gcc3strm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/longif3.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/accstrm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/iccstrm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/libstrm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/fakestrm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/vcstrm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/vc7strm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/bcstrm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/vcstrmold.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/alphastrm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/Apiifold.%,$(CINTS2))

# strip off possible leading path from compiler command name
CXXCMD       := $(shell echo $(CXX) | sed s/".*\/"//)

ifeq ($(CXXCMD),KCC)
CINTS2       += $(MODDIRS)/kccstrm.cxx
CINTS2       := $(filter-out $(MODDIRS)/longif.%,$(CINTS2))
CINTS2       += $(MODDIRS)/longif3.cxx
else
ifeq ($(PLATFORM),linux)
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
ifeq ($(PLATFORM),hurd)
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
ifeq ($(PLATFORM),fbsd)
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
ifeq ($(PLATFORM),obsd)
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
ifeq ($(PLATFORM),hpux)
ifeq ($(ARCH),hpuxia64acc)
CINTS2       += $(MODDIRS)/accstrm.cxx
CINTS2       := $(filter-out $(MODDIRS)/longif.%,$(CINTS2))
CINTS2       += $(MODDIRS)/longif3.cxx
else
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
endif
ifeq ($(PLATFORM),solaris)
ifeq ($(SUNCC5),true)
CINTS2       := $(filter-out $(MODDIRS)/longif.%,$(CINTS2))
CINTS2       += $(MODDIRS)/longif3.cxx
ifeq ($(findstring $(CXXFLAGS),-library=iostream,no%Cstd),)
CINTS2       += $(MODDIRS)/sunstrm.cxx
#CINTS2       += $(MODDIRS)/sun5strm.cxx
else
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
else
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
endif
ifeq ($(PLATFORM),aix3)
CINTS1       += $(MODDIRS)/dlfcn.c
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
ifeq ($(PLATFORM),aix)
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
ifeq ($(PLATFORM),aix5)
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
ifeq ($(PLATFORM),sgi)
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
ifeq ($(PLATFORM),alpha)
CINTS2       += $(MODDIRS)/alphastrm.cxx
endif
ifeq ($(PLATFORM),alphagcc)
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
endif
ifeq ($(PLATFORM),sunos)
CINTS1       += $(MODDIRS)/sunos.c
endif
ifeq ($(PLATFORM),macos)
CINTS2       += $(MODDIRS)/v6_macos.cxx
CINTS2       += $(MODDIRS)/fakestrm.cxx
endif
ifeq ($(PLATFORM),macosx)
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
ifeq ($(PLATFORM),lynxos)
CINTS2       += $(MODDIRS)/fakestrm.cxx
endif
ifeq ($(PLATFORM),win32)
CINTS2       += $(MODDIRS)/v6_winnt.cxx
CINTS2       := $(filter-out $(MODDIRS)/longif.%,$(CINTS2))
CINTS2       += $(MODDIRS)/longif3.cxx
ifeq ($(VC_MAJOR),13)
 ifeq ($(VC_MINOR),10)
  CINTS2       += $(MODDIRS)/vc7strm.cxx
 else
  CINTS2       += $(MODDIRS)/iccstrm.cxx
 endif
else
 ifeq ($(VC_MAJOR),14)
  CINTS2       += $(MODDIRS)/vc7strm.cxx
 else
  CINTS2       += $(MODDIRS)/iccstrm.cxx
 endif
endif
endif
ifeq ($(PLATFORM),vms)
CINTS2       += $(MODDIRS)/fakestrm.cxx
endif
ifeq ($(CXXCMD),icc)
CINTS2       := $(filter-out $(MODDIRS)/libstrm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/longif.%,$(CINTS2))
ifeq ($(ICC_MAJOR),8)
ifneq ($(ICC_MINOR),0)
CINTS2       += $(MODDIRS)/gcc3strm.cxx
else
CINTS2       += $(MODDIRS)/iccstrm.cxx
endif
else
ifeq ($(ICC_MAJOR),9)
CINTS2       += $(MODDIRS)/gcc3strm.cxx
else
CINTS2       += $(MODDIRS)/iccstrm.cxx
endif
endif
CINTS2       += $(MODDIRS)/longif3.cxx
endif
ifeq ($(GCC_MAJOR),3)
CINTS2       := $(filter-out $(MODDIRS)/libstrm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/longif.%,$(CINTS2))
CINTS2       += $(MODDIRS)/gcc3strm.cxx
CINTS2       += $(MODDIRS)/longif3.cxx
endif
ifeq ($(GCC_MAJOR),4)
CINTS2       := $(filter-out $(MODDIRS)/libstrm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/longif.%,$(CINTS2))
CINTS2       += $(MODDIRS)/gcc3strm.cxx
CINTS2       += $(MODDIRS)/longif3.cxx
endif
ifeq ($(CXXCMD),xlC)
ifeq ($(PLATFORM),macosx)
CINTS2       := $(filter-out $(MODDIRS)/libstrm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/longif.%,$(CINTS2))
CINTS2       += $(MODDIRS)/gcc3strm.cxx
CINTS2       += $(MODDIRS)/longif3.cxx
endif
endif

CINTS        := $(CINTS1) $(CINTS2)
CINTO        := $(CINTS1:.c=.o) $(CINTS2:.cxx=.o)
CINTTMPO     := $(subst v6_loadfile.o,v6_loadfile_tmp.o,$(CINTO))
CINTTMPINC   := -I$(MODDIR)/include -I$(MODDIR)/stl -I$(MODDIR)/lib
CINTDEP      := $(CINTO:.o=.d)
CINTDEP      += $(MODDIRS)/v6_loadfile_tmp.d
CINTALLDEP   += $(MODDIRS)/v6_loadfile_tmp.d

CINTLIB      := $(LPATH)/libCint.$(SOEXT)

##### cint #####
CINTEXES     := $(CINTDIRM)/cppmain.cxx
CINTEXEO     := $(CINTEXES:.cxx=.o)
CINTEXEDEP   := $(CINTEXEO:.o=.d)
CINTTMP      := $(CINTDIRM)/cint_tmp$(EXEEXT)
CINT         := bin/cint$(EXEEXT)

##### makecint #####
MAKECINTS    := $(CINTDIRT)/makecint.cxx
MAKECINTO    := $(MAKECINTS:.cxx=.o)
MAKECINT     := bin/makecint$(EXEEXT)

##### iosenum.h #####
IOSENUM      := $(MODDIR)/include/iosenum.h
IOSENUMC     := $(MODDIR)/iosenum/iosenum.cxx
ifeq ($(GCC_MAJOR),4)
IOSENUMA     := $(MODDIR)/iosenum/iosenum.$(ARCH)3
else
ifeq ($(GCC_MAJOR),3)
IOSENUMA     := $(MODDIR)/iosenum/iosenum.$(ARCH)3
else
IOSENUMA     := $(MODDIR)/iosenum/iosenum.$(ARCH)
endif
endif

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(CINTH)) $(CINTCONF)

CINTCXXFLAGS += -DG__HAVE_CONFIG -DG__NOMAKEINFO -DG__CINTBODY -I$(CINTDIRS)

# include all dependency files
INCLUDEFILES += $(CINTDEP) $(CINTEXEDEP)

##### local rules #####
include/%.h:    $(CINTDIRI)/%.h
		cp $< $@

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

$(IOSENUM):     $(IOSENUMA)
		cp $< $@

$(IOSENUMA):    $(CINTTMP)
		@(if [ ! -r $@ ]; then \
			echo "Making $@..."; \
			$(CINTTMP) $(CINTTMPINC) $(IOSENUMC) > /dev/null; \
			mv iosenum.h $@; \
		else \
			touch $@; \
		fi)

all-cint:       $(CINTLIB) $(CINT) $(CINTTMP) $(MAKECINT) $(IOSENUM)

clean-cint:
		@rm -f $(CINTTMPO) $(CINTALLO) $(CINTEXEO) $(MAKECINTO)

clean::         clean-cint

distclean-cint: clean-cint
		@rm -f $(CINTALLDEP) $(CINTLIB) $(IOSENUM) $(CINTEXEDEP) \
		   $(CINT) $(CINTTMP) $(MAKECINT) $(CINTDIRM)/*.exp \
		   $(CINTDIRM)/*.lib $(CINTDIRS)/v6_loadfile_tmp.cxx

distclean::     distclean-cint

##### extra rules ######
$(CINTDIRS)/libstrm.o:  CINTCXXFLAGS += -I$(CINTDIRL)/stream
$(CINTDIRS)/sun5strm.o: CINTCXXFLAGS += -I$(CINTDIRL)/sunstrm
$(CINTDIRS)/vcstrm.o:   CINTCXXFLAGS += -I$(CINTDIRL)/vcstream
$(CINTDIRS)/%strm.o:    CINTCXXFLAGS += -I$(CINTDIRL)/$(notdir $(basename $@))

$(MAKECINTO) $(CINTALLO): $(CINTCONF)

$(CINTDIRS)/v6_stdstrct.o:     CINTCXXFLAGS += -I$(CINTDIRL)/stdstrct
$(CINTDIRS)/v6_loadfile_tmp.o: CINTCXXFLAGS += -UR__HAVE_CONFIG -DROOTBUILD

$(CINTDIRS)/v6_loadfile_tmp.cxx: $(CINTDIRS)/v6_loadfile.cxx
	cp -f $< $@

##### configcint.h
ifeq ($(CPPPREP),)
# cannot use "CPPPREP?=", as someone might set "CPPPREP="
  CPPPREP = $(CXX) -E -C
endif
include $(CINTCONFMK)
##### configcint.h - END
