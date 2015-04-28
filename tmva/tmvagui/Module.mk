# Module.mk for tmva module
# Copyright (c) 2006 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 20/6/2009

MODNAME      := tmvagui
MODDIR       := $(ROOT_SRCDIR)/tmva/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

TMVAGUIDIR      := $(MODDIR)
TMVAGUIDIRS     := $(TMVAGUIDIR)/src
TMVAGUIDIRI     := $(TMVAGUIDIR)/inc

##### libTMVAGUI #####
TMVAGUIL0       := $(MODDIRI)/LinkDef.h
#TMVAGUILS       := $(MODDIRI)/LinkDef1.h $(MODDIRI)/LinkDef2.h $(MODDIRI)/LinkDef3.h $(MODDIRI)/LinkDef4.h 
TMVAGUIDS       := $(call stripsrc,$(MODDIRS)/G__TMVAGui.cxx)
TMVAGUIDO       := $(TMVAGUIDS:.cxx=.o)
TMVAGUIDH       := $(TMVAGUIDS:.cxx=.h)

TMVAGUIH1       := annconvergencetest.h  deviations.h mvaeffs.h PlotFoams.h  TMVAGui.h\
	 BDTControlPlots.h  correlationscatters.h efficiencies.h  mvas.h probas.h \
	 BDT.h   correlationscattersMultiClass.h  likelihoodrefs.h  mvasMulticlass.h  regression_averagedevs.h  TMVAMultiClassGui.h\
	 BDT_Reg.h  correlations.h   mvaweights.h rulevisCorr.h  TMVARegGui.h\
	 BoostControlPlots.h correlationsMultiClass.h network.h rulevis.h   variables.h\
	 CorrGui.h  paracoor.h  rulevisHists.h variablesMultiClass.h\
	 compareanapp.h  CorrGuiMultiClass.h   MovieMaker.h tmvaglob.h

TMVAGUIH1       := $(patsubst %,$(MODDIRI)/TMVA/%,$(TMVAGUIH1))
TMVAGUIH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/TMVA/*.h))
TMVAGUIS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
TMVAGUIO        := $(call stripsrc,$(TMVAGUIS:.cxx=.o))

TMVAGUIDEP      := $(TMVAGUIO:.o=.d) $(TMVAGUIDO:.o=.d)

TMVAGUILIB      := $(LPATH)/libTMVAGui.$(SOEXT)
TMVAGUIMAP      := $(TMVAGUILIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/TMVA/%.h,include/TMVA/%.h,$(TMVAGUIH))
ALLLIBS      += $(TMVAGUILIB)
ALLMAPS      += $(TMVAGUIMAP)

# include all dependency files
INCLUDEFILES += $(TMVAGUIDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/TMVA/%.h: $(TMVAGUIDIRI)/TMVA/%.h
		@(if [ ! -d "include/TMVA" ]; then     \
		   mkdir -p include/TMVA;              \
		fi)
		cp $< $@

$(TMVAGUILIB):     $(TMVAGUIO) $(TMVAGUIDO) $(ORDER_) $(MAINLIBS) $(TMVAGUILIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libTMVAGui.$(SOEXT) $@ "$(TMVAGUIO) $(TMVAGUIDO)" \
		   "$(TMVAGUILIBEXTRA)"

$(call pcmrule,TMVAGUI)
	$(noop)

$(TMVAGUIDS):      $(TMVAGUIH) $(TMVAGUIL0) $(TMVAGUILS) $(ROOTCLINGEXE) $(call pcmdep,TMVAGUI)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,TMVAGUI) -c -writeEmptyRootPCM -I$(ROOT_SRCDIR) $(TMVAGUIH) $(TMVAGUIL0)

$(TMVAGUIMAP):     $(TMVAGUIH) $(TMVAGUIL0) $(TMVAGUILS) $(ROOTCLINGEXE) $(call pcmdep,TMVAGUI)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(TMVAGUIDS) $(call dictModule,TMVAGUI) -c -I$(ROOT_SRCDIR) $(TMVAGUIH) $(TMVAGUIL0)

all-$(MODNAME): $(TMVAGUILIB)

clean-$(MODNAME):
		@rm -f $(TMVAGUIDIRS)/*.o

clean::         clean-tmva

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(TMVAGUIDEP) $(TMVAGUIDS) $(TMVAGUIDH) $(TMVAGUILIB) $(TMVAGUIMAP)
		@rm -rf include/TMVA

distclean::     distclean-$(MODNAME)
