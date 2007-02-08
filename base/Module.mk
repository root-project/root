# Module.mk for base module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := base
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

BASEDIR      := $(MODDIR)
BASEDIRS     := $(BASEDIR)/src
BASEDIRI     := $(BASEDIR)/inc

##### libBase (part of libCore) #####
BASEL1       := $(MODDIRI)/LinkDef1.h
BASEL2       := $(MODDIRI)/LinkDef2.h
BASEL3       := $(MODDIRI)/LinkDef3.h
BASEDS1      := $(MODDIRS)/G__Base1.cxx
BASEDS2      := $(MODDIRS)/G__Base2.cxx
BASEDS3      := $(MODDIRS)/G__Base3.cxx
BASEDO1      := $(BASEDS1:.cxx=.o)
BASEDO2      := $(BASEDS2:.cxx=.o)
BASEDO3      := $(BASEDS3:.cxx=.o)

# ManualBase4 only needs to be regenerated (and then changed manually) when
# the dictionary interface changes
BASEL4       := $(MODDIRI)/LinkDef4.h
BASEDS4      := $(MODDIRS)/ManualBase4.cxx
BASEDO4      := $(BASEDS4:.cxx=.o)
BASEH4       := TDirectory.h

BASEDS       := $(BASEDS1) $(BASEDS2) $(BASEDS3) $(BASEDS4)
ifeq ($(PLATFORM),win32)
BASEDO       := $(BASEDO1) $(BASEDO2) $(BASEDO3) $(BASEDO4)
else
BASEDO       := $(BASEDO1) $(BASEDO2) $(BASEDO3)
endif
BASEDH       := $(BASEDS:.cxx=.h)

BASEH1       := $(wildcard $(MODDIRI)/T*.h)
BASEH3       := GuiTypes.h KeySymbols.h Buttons.h TTimeStamp.h TVirtualMutex.h \
                TVirtualPerfStats.h TVirtualX.h TParameter.h \
                TVirtualAuth.h TFileInfo.h TRedirectOutputGuard.h \
                TVirtualMonitoring.h TObjectSpy.h
BASEH3       := $(patsubst %,$(MODDIRI)/%,$(BASEH3))
BASEH1       := $(filter-out $(BASEH3),$(BASEH1))
BASEH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
ifeq ($(PLATFORM),win32)
BASES        := $(filter-out $(BASEDS4),$(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx)))
else
BASES        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
endif
BASEO        := $(BASES:.cxx=.o)

BASEDEP      := $(BASEO:.o=.d) $(BASEDO:.o=.d)

BASEO        := $(filter-out $(MODDIRS)/precompile.o,$(BASEO))

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(BASEH))

# include all dependency files
INCLUDEFILES += $(BASEDEP)

##### local rules #####
include/%.h:    $(BASEDIRI)/%.h
		cp $< $@

# Explicitely state this dependency.
# rmkdepend does not pick it up if $(COMPILEDATA) doesn't exist yet.
base/src/TSystem.d base/src/TSystem.o: $(COMPILEDATA)

$(BASEDS1):     $(BASEH1) $(BASEL1) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(BASEH1) $(BASEL1)
$(BASEDS2):     $(BASEH1) $(BASEL2) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(BASEH1) $(BASEL2)
$(BASEDS3):     $(BASEH3) $(BASEL3) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(BASEH3) $(BASEL3)
# pre-requisites intentionally not specified... should be called only
# on demand after deleting the file
$(BASEDS4):
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(BASEH4) $(BASEL4)

all-base:       $(BASEO) $(BASEDO)

clean-base:
		@rm -f $(BASEO) $(BASEDO) $(BASEDIRS)/precompile.o

clean::         clean-base

distclean-base: clean-base
		@rm -f $(BASEDEP) \
		   $(filter-out $(BASEDIRS)/ManualBase4.cxx, $(BASEDS)) \
		   $(filter-out $(BASEDIRS)/ManualBase4.h, $(BASEDH))

distclean::     distclean-base

##### extra rules ######
base/src/TPRegexp.o: $(PCREDEP)
base/src/TPRegexp.o: CXXFLAGS += $(PCREINC)

ifeq ($(ARCH),alphacxx6)
$(BASEDIRS)/TRandom.o: OPT = $(NOOPT)
endif

$(BASEDO1) $(BASEDO2): $(PCREDEP)
$(BASEDO1) $(BASEDO2): CXXFLAGS += $(PCREINC)
ifeq ($(ARCH),linuxicc)
$(BASEDO3):     CXXFLAGS += -wd191
endif
$(BASEDO4): OPT = $(NOOPT)
$(BASEDO4): CXXFLAGS += -I.
