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

##### libCint #####
CINTH        := $(wildcard $(MODDIRI)/*.h)
CINTS1       := $(wildcard $(MODDIRS)/*.c)
CINTS2       := $(wildcard $(MODDIRS)/*.cxx)

CINTS1       := $(filter-out $(MODDIRS)/sunos.%,$(CINTS1))
CINTS1       := $(filter-out $(MODDIRS)/dlfcn.%,$(CINTS1))
CINTS1       := $(filter-out $(MODDIRS)/macos.%,$(CINTS1))
CINTS1       := $(filter-out $(MODDIRS)/winnt.%,$(CINTS1))

CINTS2       := $(filter-out $(MODDIRS)/kccstrm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/sunstrm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/sun5strm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/libstrm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/fakestrm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/vcstrm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/bcstrm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/vcstrmold.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/Apiifold.%,$(CINTS2))

CINTS1       += $(CINTDIRM)/G__setup.c

ifeq ($(CXX),KCC)
CINTS2       += $(MODDIRS)/kccstrm.cxx
else
ifeq ($(PLATFORM),linux)
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
ifeq ($(PLATFORM),fbsd)
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
ifeq ($(PLATFORM),hpux)
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
ifeq ($(PLATFORM),solaris)
ifeq ($(SUNCC5),true)
ifeq ($(findstring $(CXXFLAGS),-library=iostream,no%Cstd),)
#CINTS2       += $(MODDIRS)/sunstrm.cxx
CINTS2       += $(MODDIRS)/sun5strm.cxx
else
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
else
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
endif
ifeq ($(PLATFORM),aix3)
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
ifeq ($(PLATFORM),aix)
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
ifeq ($(PLATFORM),sgi)
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
ifeq ($(PLATFORM),alpha)
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
ifeq ($(PLATFORM),alphaegcs)
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
endif

ifeq ($(PLATFORM),sunos)
CINTS1       += $(MODDIRS)/sunos.c
endif
ifeq ($(PLATFORM),aix3)
CINTS1       += $(MODDIRS)/dlfcn.c
endif
ifeq ($(PLATFORM),macos)
CINTS1       += $(MODDIRS)/macos.c
CINTS2       += $(MODDIRS)/fakestrm.cxx
endif
ifeq ($(PLATFORM),lynxos)
CINTS2       += $(MODDIRS)/fakestrm.cxx
endif
ifeq ($(PLATFORM),win32)
CINTS1       += $(MODDIRS)/winnt.c
CINTS2       += $(MODDIRS)/vcstrm.cxx
endif
ifeq ($(PLATFORM),vms)
CINTS2       += $(MODDIRS)/fakestrm.cxx
endif

CINTS        := $(CINTS1) $(CINTS2)
CINTO        := $(CINTS1:.c=.o) $(CINTS2:.cxx=.o)
CINTTMPO     := $(subst loadfile.o,loadfile_tmp.o,$(CINTO))
CINTTMPINC   := -Icint/include -Icint/stl -Icint/lib
CINTDEP      := $(CINTO:.o=.d)

CINTLIB      := $(LPATH)/libCint.$(SOEXT)

##### cint #####
CINTEXES     := $(CINTDIRM)/cppmain.cxx
CINTEXEO     := $(CINTEXES:.cxx=.o)
CINTEXEDEP   := $(CINTEXEO:.o=.d)
CINTTMP      := $(CINTDIRM)/cint_tmp$(EXEEXT)
CINT         := bin/cint$(EXEEXT)

##### makecint #####
MAKECINTS    := $(CINTDIRT)/makecint.c
MAKECINTO    := $(MAKECINTS:.c=.o)
MAKECINT     := bin/makecint$(EXEEXT)

##### iosenum.h #####
IOSENUMC     := $(MODDIR)/iosenum/iosenum.cxx
IOSENUMA     := $(MODDIR)/iosenum/iosenum.$(ARCH)
IOSENUM      := $(MODDIR)/include/iosenum.h

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(CINTH))

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

$(IOSENUMA):    $(CINTTMP) $(MAKEINFO)
		@(if [ ! -r $(IOSENUMA) ]; then \
			echo "Making $@..."; \
			$(CINTTMP) $(CINTTMPINC) $(IOSENUMC) > /dev/null; \
			mv iosenum.h $@; \
		else \
			touch $@; \
		fi)

all-cint:       $(CINTLIB) $(CINT) $(CINTTMP) $(MAKECINT) $(IOSENUM)

clean-cint:
		@rm -f $(CINTTMPO) $(CINTO) $(CINTEXEO) $(MAKECINTO)

clean::         clean-cint

distclean-cint: clean-cint
		@rm -f $(CINTDEP) $(CINTLIB) $(IOSENUM) $(CINTEXEDEP) \
		   $(CINT) $(CINTTMP) $(MAKECINT)

distclean::     distclean-cint

##### extra rules ######
$(CINTDIRS)/libstrm.o: $(CINTDIRS)/libstrm.cxx
	$(CXX) $(OPT) $(CINTCXXFLAGS) -I$(CINTDIRL)/stream -o $@ -c $<

$(CINTDIRS)/vcstrm.o: $(CINTDIRS)/vcstrm.cxx
	$(CXX) $(OPT) $(CINTCXXFLAGS) -I$(CINTDIRL)/vcstream -o $@ -c $<

$(CINTDIRS)/sunstrm.o: $(CINTDIRS)/sunstrm.cxx
	$(CXX) $(OPT) $(CINTCXXFLAGS) -I$(CINTDIRL)/snstream -o $@ -c $<

$(CINTDIRS)/sun5strm.o: $(CINTDIRS)/sun5strm.cxx
	$(CXX) $(OPT) $(CINTCXXFLAGS) -I$(CINTDIRL)/snstream -o $@ -c $<

$(CINTDIRS)/stdstrct.o: $(CINTDIRS)/stdstrct.c
	$(CC) $(OPT) $(CINTCFLAGS) -I$(CINTDIRL)/stdstrct -o $@ -c $<

$(CINTDIRS)/loadfile_tmp.o: $(CINTDIRS)/loadfile.c
	$(CC) $(OPT) $(CINTCFLAGS) -UHAVE_CONFIG -DROOTBUILD -o $@ -c $<

$(CINTDIRT)/makecint.o: $(CINTDIRT)/makecint.c
	$(CC) $(OPT) $(CINTCFLAGS) -o $@ -c $<

$(CINTDIRT)/makecint_tmp.o: $(CINTDIRT)/makecint.c
	$(CC) $(OPT) $(CINTCFLAGS) -UHAVE_CONFIG -DROOTBUILD -o $@ -c $<
