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

CINTS1       += $(CINTDIRM)/G__setup.c

CINTALLO     := $(CINTS1:.c=.o) $(CINTS2:.cxx=.o)
CINTALLDEP   := $(CINTALLO:.o=.d)

CINTS1       := $(filter-out $(MODDIRS)/sunos.%,$(CINTS1))
CINTS1       := $(filter-out $(MODDIRS)/dlfcn.%,$(CINTS1))
CINTS1       := $(filter-out $(MODDIRS)/macos.%,$(CINTS1))
CINTS1       := $(filter-out $(MODDIRS)/winnt.%,$(CINTS1))
CINTS1       := $(filter-out $(MODDIRS)/newsos.%,$(CINTS1))

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
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
ifeq ($(PLATFORM),alphaegcs)
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
endif
ifeq ($(PLATFORM),sunos)
CINTS1       += $(MODDIRS)/sunos.c
endif
ifeq ($(PLATFORM),macos)
CINTS1       += $(MODDIRS)/macos.c
CINTS2       += $(MODDIRS)/fakestrm.cxx
endif
ifeq ($(PLATFORM),macosx)
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
ifeq ($(PLATFORM),lynxos)
CINTS2       += $(MODDIRS)/fakestrm.cxx
endif
ifeq ($(PLATFORM),win32)
CINTS1       += $(MODDIRS)/winnt.c
CINTS2       := $(filter-out $(MODDIRS)/longif.%,$(CINTS2))
CINTS2       += $(MODDIRS)/longif3.cxx
ifeq ($(VC_MAJOR),13)
ifeq ($(VC_MINOR),10)
CINTS2       += $(MODDIRS)/vc7strm.cxx
else
CINTS2       += $(MODDIRS)/iccstrm.cxx
endif
else
CINTS2       += $(MODDIRS)/iccstrm.cxx
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
CINTTMPO     := $(subst loadfile.o,loadfile_tmp.o,$(CINTO))
CINTTMPINC   := -I$(MODDIR)/include -I$(MODDIR)/stl -I$(MODDIR)/lib
CINTDEP      := $(CINTO:.o=.d)
CINTDEP      += $(MODDIRS)/loadfile_tmp.d
CINTALLDEP   += $(MODDIRS)/loadfile_tmp.d

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
		   $(CINTDIRM)/*.lib

distclean::     distclean-cint

##### extra rules ######
$(CINTDIRS)/libstrm.o: $(CINTDIRS)/libstrm.cxx
	$(CXX) $(OPT) $(CINTCXXFLAGS) -I$(CINTDIRL)/stream -o $@ -c $<

$(CINTDIRS)/vcstrm.o: $(CINTDIRS)/vcstrm.cxx
	$(CXX) $(OPT) $(CINTCXXFLAGS) -I$(CINTDIRL)/vcstream -o $@ -c $<

$(CINTDIRS)/vc7strm.o: $(CINTDIRS)/vc7strm.cxx
	$(CXX) $(OPT) $(CINTCXXFLAGS) -I$(CINTDIRL)/vc7strm -o $@ -c $<

$(CINTDIRS)/sunstrm.o: $(CINTDIRS)/sunstrm.cxx
	$(CXX) $(OPT) $(CINTCXXFLAGS) -I$(CINTDIRL)/snstream -o $@ -c $<

$(CINTDIRS)/sun5strm.o: $(CINTDIRS)/sun5strm.cxx
	$(CXX) $(OPT) $(CINTCXXFLAGS) -I$(CINTDIRL)/snstream -o $@ -c $<

$(CINTDIRS)/gcc3strm.o: $(CINTDIRS)/gcc3strm.cxx
	$(CXX) $(OPT) $(CINTCXXFLAGS) -I$(CINTDIRL)/gcc3strm -o $@ -c $<

$(CINTDIRS)/iccstrm.o: $(CINTDIRS)/iccstrm.cxx
	$(CXX) $(OPT) $(CINTCXXFLAGS) -I$(CINTDIRL)/iccstrm -o $@ -c $<

$(CINTDIRS)/accstrm.o: $(CINTDIRS)/accstrm.cxx
	$(CXX) $(OPT) $(CINTCXXFLAGS) -I$(CINTDIRL)/accstrm -o $@ -c $<

$(CINTDIRS)/stdstrct.o: $(CINTDIRS)/stdstrct.c
	$(CC) $(OPT) $(CINTCFLAGS) -I$(CINTDIRL)/stdstrct -o $@ -c $<

$(CINTDIRS)/loadfile_tmp.o: $(CINTDIRS)/loadfile.c
	$(CC) $(OPT) $(CINTCFLAGS) -UHAVE_CONFIG -DROOTBUILD -o $@ -c $<

$(CINTDIRT)/makecint.o: $(CINTDIRT)/makecint.c
	$(CC) $(OPT) $(CINTCFLAGS) -o $@ -c $<

$(CINTDIRT)/makecint_tmp.o: $(CINTDIRT)/makecint.c
	$(CC) $(OPT) $(CINTCFLAGS) -UHAVE_CONFIG -DROOTBUILD -o $@ -c $<

$(CINTDIRS)/loadfile_tmp.d: $(CINTDIRS)/loadfile.c $(RMKDEP)
	@cp $(CINTDIRS)/loadfile.c $(CINTDIRS)/loadfile_tmp.c
	$(MAKEDEP) $@ "$(CFLAGS)" $(CINTDIRS)/loadfile_tmp.c > $@
	@rm -f $(CINTDIRS)/loadfile_tmp.c
